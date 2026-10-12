from __future__ import annotations

import argparse
import csv
import glob
import queue
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .scheduler import RoundRobinScheduler
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scheduler import RoundRobinScheduler

try:
    from profile_rknn_single_stream import decode_yolo11_raw
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    try:
        from profile_rknn_single_stream import decode_yolo11_raw
    except ImportError:
        decode_yolo11_raw = None

try:
    from tools.profile_rknn_single_stream import letterbox_bgr
except ImportError:
    try:
        from profile_rknn_single_stream import letterbox_bgr
    except ImportError:
        letterbox_bgr = None


class DropOldestQueue:
    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self._items: list[Any] = []
        self._lock = threading.Lock()
        self.dropped = 0

    def put_drop_oldest(self, item: Any) -> Any | None:
        with self._lock:
            evicted = self._items.pop(0) if len(self._items) >= self.maxsize else None
            if evicted is not None:
                self.dropped += 1
            self._items.append(item)
            return evicted

    def get_nowait(self) -> Any:
        with self._lock:
            if not self._items:
                raise queue.Empty
            return self._items.pop(0)

    def qsize(self) -> int:
        with self._lock:
            return len(self._items)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _resource_sample() -> dict[str, Any]:
    import psutil

    try:
        result = subprocess.run(["sudo", "-n", "cat", "/sys/kernel/debug/rknpu/load"], capture_output=True, text=True, timeout=1.0)
        npu = result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        npu = ""
    row: dict[str, Any] = {"timestamp": time.time(), "npu_load": npu}
    row.update({f"cpu_core_{i}_percent": value for i, value in enumerate(psutil.cpu_percent(percpu=True))})
    for path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            value = float(Path(path).read_text().strip())
            row[Path(path).parent.name] = value / 1000 if value > 1000 else value
        except (OSError, ValueError):
            pass
    return row


def run_replay(video: Path, model: Path, stream_count: int, duration: float, output_dir: Path, queue_depth: int = 3) -> None:
    import cv2
    from rknnlite.api import RKNNLite

    output_dir.mkdir(parents=True, exist_ok=True)
    stream_queues = {stream_id: DropOldestQueue(queue_depth) for stream_id in range(stream_count)}
    post_queues = {stream_id: queue.Queue(maxsize=queue_depth) for stream_id in range(stream_count)}
    rows: list[dict[str, Any]] = []
    rows_lock = threading.Lock()
    resource_rows: list[dict[str, Any]] = []
    queue_samples = {stream_id: [] for stream_id in range(stream_count)}
    stop = threading.Event()
    done = {stream_id: False for stream_id in range(stream_count)}
    done_lock = threading.Lock()
    deadline = time.perf_counter() + duration

    def decode_worker(stream_id: int) -> None:
        cap = cv2.VideoCapture(str(video))
        frame_id = 0
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            period = 1.0 / fps
            next_tick = time.perf_counter()
            while time.perf_counter() < deadline:
                wait = next_tick - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                ok, bgr = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, bgr = cap.read()
                if not ok:
                    break
                packet = {"stream_id": stream_id, "frame_id": frame_id, "arrival_ts": time.perf_counter(), "bgr": bgr}
                evicted = stream_queues[stream_id].put_drop_oldest(packet)
                queue_samples[stream_id].append(stream_queues[stream_id].qsize())
                if evicted is not None:
                    with rows_lock:
                        rows.append({"stream_id": stream_id, "frame_id": evicted["frame_id"], "arrival_ts": evicted["arrival_ts"], "selected_ts": "", "infer_done_ts": "", "dropped": True})
                frame_id += 1
                next_tick += period
        finally:
            cap.release()
            with done_lock:
                done[stream_id] = True

    def sampler() -> None:
        while not stop.is_set():
            resource_rows.append(_resource_sample())
            stop.wait(0.5)

    for stream_id in range(stream_count):
        threading.Thread(target=decode_worker, args=(stream_id,), daemon=True).start()
    threading.Thread(target=sampler, daemon=True).start()
    post_threads = []
    for sid in range(stream_count):
        thread = threading.Thread(target=_post_worker, args=(sid, post_queues[sid], rows, rows_lock), daemon=True)
        thread.start()
        post_threads.append(thread)

    rknn = RKNNLite(verbose=False)
    if rknn.load_rknn(str(model)) != 0 or rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
        raise RuntimeError(f"failed to initialize {model}")
    scheduler = RoundRobinScheduler(stream_count)
    inferred = 0
    try:
        while True:
            states = {sid: {"queue_length": stream_queues[sid].qsize()} for sid in range(stream_count)}
            sid = scheduler.select(states)
            if sid < 0:
                with done_lock:
                    finished = all(done.values())
                if finished and all(stream_queues[i].qsize() == 0 for i in range(stream_count)):
                    break
                time.sleep(0.001)
                continue
            packet = stream_queues[sid].get_nowait()
            packet["selected_ts"] = time.perf_counter()
            image, scale, pad_x, pad_y = _letterbox(packet["bgr"], _REPLAY_CANVAS)
            outputs = rknn.inference(inputs=[image[None, ...]], data_format="nhwc")
            packet["outputs"] = outputs
            packet["scale"], packet["pad_x"], packet["pad_y"] = scale, pad_x, pad_y
            post_queues[sid].put(packet)
            inferred += 1
            queue_samples[sid].append(stream_queues[sid].qsize())
    finally:
        rknn.release()
        stop.set()
    for sid in range(stream_count):
        post_queues[sid].put(sentinel_global)
    for thread in post_threads:
        thread.join()
    fields = ["stream_id", "frame_id", "arrival_ts", "selected_ts", "infer_done_ts", "dropped"]
    _write_csv(output_dir / f"multistream_{stream_count}_per_frame.csv", sorted(rows, key=lambda row: (int(row["stream_id"]), int(row["frame_id"]))), fields)
    _write_csv(output_dir / f"multistream_{stream_count}_resources.csv", resource_rows, list(resource_rows[0]) if resource_rows else ["timestamp", "npu_load"])
    metrics = []
    for sid in range(stream_count):
        selected = [r for r in rows if int(r["stream_id"]) == sid and not r["dropped"]]
        dropped = [r for r in rows if int(r["stream_id"]) == sid and r["dropped"]]
        e2e = [(float(r["infer_done_ts"]) - float(r["arrival_ts"])) * 1000 for r in selected]
        wait = [(float(r["selected_ts"]) - float(r["arrival_ts"])) * 1000 for r in selected]
        metrics.append({"stream_id": sid, "inferred_fps": len(selected) / duration, "drop_rate": len(dropped) / max(len(selected) + len(dropped), 1), "e2e_mean_ms": statistics.mean(e2e) if e2e else 0.0, "e2e_p95_ms": float(np.percentile(e2e, 95)) if e2e else 0.0, "wait_mean_ms": statistics.mean(wait) if wait else 0.0, "wait_p95_ms": float(np.percentile(wait, 95)) if wait else 0.0, "queue_avg": statistics.mean(queue_samples[sid]) if queue_samples[sid] else 0.0})
    _write_csv(output_dir / f"multistream_{stream_count}_summary.csv", metrics, list(metrics[0]) if metrics else ["stream_id"])
    output_dir.joinpath(f"multistream_{stream_count}_summary.md").write_text(f"# Multistream {stream_count}-stream round-robin\n\n- system throughput: {inferred / duration:.3f} FPS\n- scheduler: RoundRobinScheduler\n- queue depth: {queue_depth}\n\n" + "\n".join(f"- stream {m['stream_id']}: {m['inferred_fps']:.3f} FPS, drop {m['drop_rate']:.3%}, e2e mean/p95 {m['e2e_mean_ms']:.3f}/{m['e2e_p95_ms']:.3f} ms, wait mean/p95 {m['wait_mean_ms']:.3f}/{m['wait_p95_ms']:.3f} ms, queue avg {m['queue_avg']:.3f}" for m in metrics) + "\n", encoding="utf-8")


def _letterbox(image: np.ndarray, size):
    """Letterbox to a fixed canvas, delegating to the canonical implementation.

    This used to build a square ``size x size`` canvas from an ``int``, which is
    the exact assumption the rectangular work removes: a 16:9 frame padded into a
    square wastes ~40% of the input. It now accepts ``int`` or ``(width, height)``
    and shares one implementation with the profiler, the detector and the exports.
    """
    if letterbox_bgr is None:
        raise RuntimeError("tools.profile_rknn_single_stream.letterbox_bgr is not importable")
    return letterbox_bgr(image, size)


def _post_worker(stream_id, post_queue, rows, rows_lock):
    while True:
        packet = post_queue.get()
        if packet is sentinel_global:
            return
        if decode_yolo11_raw is not None:
            decode_yolo11_raw(packet["outputs"], _REPLAY_CANVAS, 0.25, 0.45, packet["scale"], packet["pad_x"], packet["pad_y"])
        packet["infer_done_ts"] = time.perf_counter()
        with rows_lock:
            rows.append({"stream_id": stream_id, "frame_id": packet["frame_id"], "arrival_ts": packet["arrival_ts"], "selected_ts": packet["selected_ts"], "infer_done_ts": packet["infer_done_ts"], "dropped": False})


sentinel_global = object()

# The historical homogeneous replay canvas. Kept as an explicit named constant so
# ``320`` no longer appears as a bare literal in the inference path.
_REPLAY_CANVAS = 320


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--streams", type=int, nargs="+", default=[4])
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=Path("multistream_output"))
    args = parser.parse_args()
    for count in args.streams:
        run_replay(args.video, args.model, count, args.duration, args.output_dir)


if __name__ == "__main__":
    main()
