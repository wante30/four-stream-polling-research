"""Metadata-only timestamp replay feeding one serial RKNN worker."""
from __future__ import annotations

import argparse
import csv
import glob
import queue
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

try:
    from multistream_framework.scheduler import RoundRobinScheduler
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from multistream_framework.scheduler import RoundRobinScheduler


class DropOldestQueue:
    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self._items: list[tuple[int, int, float, int]] = []
        self._lock = threading.Lock()
        self.dropped = 0
        self.samples: list[int] = []

    def put_drop_oldest(self, item: tuple[int, int, float, int]):
        with self._lock:
            evicted = self._items.pop(0) if len(self._items) >= self.maxsize else None
            if evicted is not None:
                self.dropped += 1
            self._items.append(item)
            self.samples.append(len(self._items))
            return evicted

    def get_nowait(self):
        with self._lock:
            if not self._items:
                raise queue.Empty
            item = self._items.pop(0)
            self.samples.append(len(self._items))
            return item

    def qsize(self) -> int:
        with self._lock:
            return len(self._items)


def read_npu_load() -> str:
    try:
        return Path("/sys/kernel/debug/rknpu/load").read_text(encoding="utf-8").strip()
    except OSError:
        try:
            result = subprocess.run(["sudo", "-n", "cat", "/sys/kernel/debug/rknpu/load"], capture_output=True, text=True, timeout=1)
            return result.stdout.strip() if result.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            return ""


def sample_resources(rows: list[dict[str, object]], stop: threading.Event) -> None:
    import psutil
    while not stop.is_set():
        row: dict[str, object] = {"timestamp": time.time(), "npu_load": read_npu_load()}
        for index, value in enumerate(psutil.cpu_percent(percpu=True)):
            row[f"cpu_core_{index}_percent"] = value
        for path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
            try:
                value = float(Path(path).read_text(encoding="utf-8").strip())
                row[Path(path).parent.name] = value / 1000.0 if value > 1000 else value
            except (OSError, ValueError):
                pass
        rows.append(row); stop.wait(0.5)


def parse_npu_percent(value: object) -> float | None:
    import re
    match = re.search(r"(\d+(?:\.\d+)?)%", str(value))
    return float(match.group(1)) if match else None


def run(video_memmap: Path, timestamps_csv: Path, model: Path, streams: int, duration: float, output_dir: Path, depth: int = 3) -> None:
    from rknnlite.api import RKNNLite
    timestamps: list[float] = []
    with timestamps_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamps.append(float(row["timestamp_s"]))
    if not timestamps:
        raise RuntimeError("timestamp CSV is empty")
    mmap = np.load(str(video_memmap), mmap_mode="r")
    if mmap.ndim != 4 or mmap.shape[1:] != (320, 320, 3):
        raise ValueError(f"unexpected memmap shape {mmap.shape}")
    fps = 1.0 / max(np.median(np.diff(timestamps)) if len(timestamps) > 1 else 0.04, 1e-6)
    period = 1.0 / fps
    queues = {sid: DropOldestQueue(depth) for sid in range(streams)}
    work_queue: queue.Queue = queue.Queue(maxsize=1)
    post_queues = {sid: queue.Queue(maxsize=depth) for sid in range(streams)}
    rows: list[dict[str, object]] = []
    rows_lock = threading.Lock()
    resources: list[dict[str, object]] = []
    source_due = [0] * streams
    counters_lock = threading.Lock()
    stop = threading.Event(); producers_done = [False] * streams; done_lock = threading.Lock()
    deadline = time.perf_counter() + duration

    def producer(sid: int) -> None:
        frame_id = 0; next_tick = time.perf_counter()
        while time.perf_counter() < deadline:
            wait = next_tick - time.perf_counter()
            if wait > 0: time.sleep(wait)
            # Monotonic timestamp is used for latency arithmetic; the source PTS is
            # represented by the metadata offset/timestamp sequence and replay pace.
            timestamp = time.perf_counter()
            event = (sid, frame_id, timestamp, frame_id % len(timestamps))
            with counters_lock:
                source_due[sid] += 1
            evicted = queues[sid].put_drop_oldest(event)
            with rows_lock:
                if evicted is not None:
                    rows.append({"stream_id": evicted[0], "frame_id": evicted[1], "arrival_ts": evicted[2], "selected_ts": "", "infer_done_ts": "", "dropped": True})
            frame_id += 1; next_tick += period
        with done_lock: producers_done[sid] = True

    def npu_worker() -> None:
        rknn = RKNNLite(verbose=False)
        if rknn.load_rknn(str(model)) != 0 or rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
            raise RuntimeError(f"failed to initialize {model}")
        try:
            while not stop.is_set() or not work_queue.empty():
                try: event = work_queue.get(timeout=0.05)
                except queue.Empty: continue
                sid, fid, arrival, offset, selected = event
                tensor = np.asarray(mmap[offset:offset + 1])
                rknn.inference(inputs=[tensor])
                done = time.perf_counter()
                post_queues[sid].put((sid, fid, arrival, selected, done))
                work_queue.task_done()
        finally: rknn.release()

    def post_worker(sid: int) -> None:
        while not stop.is_set() or not post_queues[sid].empty():
            try: event = post_queues[sid].get(timeout=0.05)
            except queue.Empty: continue
            _, fid, arrival, selected, done = event
            with rows_lock:
                rows.append({"stream_id": sid, "frame_id": fid, "arrival_ts": arrival, "selected_ts": selected, "infer_done_ts": done, "dropped": False})
            post_queues[sid].task_done()

    producer_threads = [threading.Thread(target=producer, args=(sid,), daemon=True) for sid in range(streams)]
    for thread in producer_threads: thread.start()
    resource_thread = threading.Thread(target=sample_resources, args=(resources, stop), daemon=True); resource_thread.start()
    npu_thread = threading.Thread(target=npu_worker, daemon=True); npu_thread.start()
    post_threads = [threading.Thread(target=post_worker, args=(sid,), daemon=True) for sid in range(streams)]
    for thread in post_threads: thread.start()
    scheduler = RoundRobinScheduler(streams)
    start = time.perf_counter()
    while time.perf_counter() < deadline or not all(producers_done):
        sid = scheduler.select({i: {"queue_length": queues[i].qsize()} for i in range(streams)})
        if sid < 0:
            time.sleep(0.0005); continue
        event = queues[sid].get_nowait(); selected = time.perf_counter()
        work_queue.put((*event, selected))
    for thread in producer_threads: thread.join()
    while any(q.qsize() for q in queues.values()):
        sid = scheduler.select({i: {"queue_length": queues[i].qsize()} for i in range(streams)})
        if sid < 0: break
        event = queues[sid].get_nowait(); work_queue.put((*event, time.perf_counter()))
    work_queue.join(); stop.set(); npu_thread.join(timeout=5)
    for q in post_queues.values(): q.join()
    for thread in post_threads: thread.join(timeout=2)
    resource_thread.join(timeout=2)
    elapsed = time.perf_counter() - start
    fields = ["stream_id", "frame_id", "arrival_ts", "selected_ts", "infer_done_ts", "dropped"]
    output_dir.mkdir(parents=True, exist_ok=True)
    per_frame = output_dir / f"memmap_multistream_{streams}_per_frame.csv"
    with per_frame.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(sorted(rows, key=lambda r: (int(r["stream_id"]), int(r["frame_id"]))))
    resource_fields = list(resources[0]) if resources else ["timestamp", "npu_load"]
    with (output_dir / f"memmap_multistream_{streams}_resources.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=resource_fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(resources)
    metrics = []
    for sid in range(streams):
        sid_rows = [r for r in rows if int(r["stream_id"]) == sid]
        inferred = [r for r in sid_rows if not r["dropped"] and r["infer_done_ts"] != ""]
        e2e = [(float(r["infer_done_ts"]) - float(r["arrival_ts"])) * 1000 for r in inferred]
        wait = [(float(r["selected_ts"]) - float(r["arrival_ts"])) * 1000 for r in inferred]
        due = source_due[sid]
        # Every due event either entered the queue or was evicted at insertion.
        metrics.append({"stream_id": sid, "source_frames_due": due, "enqueued_frames": due, "queue_dropped_frames": queues[sid].dropped, "inferred_frames": len(inferred), "inferred_fps": len(inferred) / elapsed if elapsed else 0.0, "drop_rate": queues[sid].dropped / due if due else 0.0, "e2e_mean_ms": statistics.mean(e2e) if e2e else 0.0, "e2e_p95_ms": float(np.percentile(e2e, 95)) if e2e else 0.0, "wait_mean_ms": statistics.mean(wait) if wait else 0.0, "wait_p95_ms": float(np.percentile(wait, 95)) if wait else 0.0, "queue_avg": statistics.mean(queues[sid].samples) if queues[sid].samples else 0.0})
    npu_values = [parse_npu_percent(r.get("npu_load", "")) for r in resources]; npu_values = [v for v in npu_values if v is not None]
    system = {"streams": streams, "elapsed_s": elapsed, "system_throughput_fps": sum(m["inferred_frames"] for m in metrics) / elapsed if elapsed else 0.0, "npu_load_mean_percent": statistics.mean(npu_values) if npu_values else "unavailable"}
    with (output_dir / f"memmap_multistream_{streams}_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    with (output_dir / f"memmap_multistream_{streams}_summary.md").open("w", encoding="utf-8") as handle:
        handle.write(f"# Memmap multistream {streams}-stream RoundRobin\n\n- elapsed: {elapsed:.3f} s\n- system throughput: {system['system_throughput_fps']:.3f} FPS\n- NPU load mean: {system['npu_load_mean_percent']}\n\n")
        handle.write("| stream | due | enqueued | queue dropped | inferred | fps | drop rate | e2e mean/p95 ms | wait mean/p95 ms | queue avg |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for m in metrics:
            handle.write(f"| {m['stream_id']} | {m['source_frames_due']} | {m['enqueued_frames']} | {m['queue_dropped_frames']} | {m['inferred_frames']} | {m['inferred_fps']:.3f} | {m['drop_rate']:.3%} | {m['e2e_mean_ms']:.3f}/{m['e2e_p95_ms']:.3f} | {m['wait_mean_ms']:.3f}/{m['wait_p95_ms']:.3f} | {m['queue_avg']:.3f} |\n")
    print(system); print(*metrics, sep="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memmap", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--streams", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=Path("memmap_multistream_output"))
    args = parser.parse_args()
    for streams in args.streams:
        run(args.memmap, args.timestamps, args.model, streams, args.duration, args.output_dir)


if __name__ == "__main__":
    main()
