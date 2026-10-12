"""Heterogeneous four-stream replay on the RK3566.

Unlike ``replay.py`` (one video, one square canvas, uniform round-robin), this
harness supports a different video, RKNN model, canvas and target FPS per stream,
which is what CH0=FRONT / CH1=LEFT / CH2=RIGHT / CH3=BACK actually needs.

Design:

* **Per stream**: a decode thread that reads its own video at its own requested
  FPS and pushes into a bounded drop-oldest queue; a postprocess thread that
  decodes YOLO raw outputs and records the timing row.
* **One inference thread** owns the NPU. It asks the scheduler which stream gets
  the next slot, letterboxes that frame to that stream's canvas, runs its model and
  hands the raw outputs to the stream's post queue.
* **Scheduler**: ``weighted`` (interleaved weighted round-robin) or ``rate``
  (per-stream deadline). Both are deterministic; see ``scheduler.py``.
* **Backpressure**: queues are bounded and drop the oldest frame, so a slow
  consumer degrades freshness instead of growing memory or stalling other streams.

Everything is recorded: requested vs actual FPS per stream, drop rate, queue depth,
service interval, NPU-only latency, end-to-end latency, CPU and RSS samples.

Config file schema (JSON)::

    {
      "label": "CONFIG-B",
      "duration_s": 300,
      "queue_depth": 3,
      "scheduler": {"kind": "rate", "priority": {"0": 3, "1": 2, "2": 2, "3": 1}},
      "streams": [
        {"stream_id": 0, "role": "FRONT", "video": "front.mp4", "model": "m_512x288.rknn",
         "width": 512, "height": 288, "target_fps": 10}
      ]
    }
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE.parent, _HERE.parent / "tools", _HERE.parent / "deployment" / "rknn", _HERE):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from competition_ai.input_shape import validate_model_shape  # noqa: E402
from scheduler import (  # noqa: E402
    RateDeadlineScheduler,
    WeightedRoundRobinScheduler,
)

try:
    from decode_yolo_output import decode_outputs
except ImportError:  # pragma: no cover - depends on the board layout
    decode_outputs = None

try:
    from tools.profile_rknn_single_stream import letterbox_bgr
except ImportError:  # pragma: no cover - flat board layout
    try:
        from profile_rknn_single_stream import letterbox_bgr
    except ImportError:
        letterbox_bgr = None

SENTINEL = object()


class DropOldestQueue:
    """Bounded FIFO that evicts the oldest item instead of blocking the producer."""

    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be positive")
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


def load_config(path: Path) -> dict:
    """Read and validate a heterogeneous replay config."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    streams = config.get("streams") or []
    if not streams:
        raise SystemExit(f"{path}: no streams defined")
    seen = set()
    for stream in streams:
        for field in ("stream_id", "video", "model", "width", "height", "target_fps"):
            if field not in stream:
                raise SystemExit(f"{path}: stream {stream.get('stream_id')} is missing {field!r}")
        sid = int(stream["stream_id"])
        if sid in seen:
            raise SystemExit(f"{path}: duplicate stream_id {sid}")
        seen.add(sid)
        try:
            width, height = validate_model_shape((stream["width"], stream["height"]))
        except ValueError as exc:
            raise SystemExit(f"{path}: stream {sid}: {exc}") from exc
        if float(stream["target_fps"]) <= 0:
            raise SystemExit(f"{path}: stream {sid}: target_fps must be positive")
        stream["stream_id"] = sid
        stream["width"], stream["height"] = width, height
        stream["target_fps"] = float(stream["target_fps"])
    config["queue_depth"] = int(config.get("queue_depth", 3))
    config["duration_s"] = float(config.get("duration_s", 60))
    return config


def make_scheduler(config: dict):
    """Build the configured scheduler for the stream set."""
    spec = config.get("scheduler") or {"kind": "rate"}
    kind = spec.get("kind", "rate")
    ids = [int(stream["stream_id"]) for stream in config["streams"]]
    priority = {int(key): int(value) for key, value in (spec.get("priority") or {}).items()}
    if kind == "weighted":
        weights = {int(key): int(value) for key, value in (spec.get("weights") or {}).items()}
        if not weights:
            weights = {0: 5, 1: 2, 2: 2, 3: 1}
        return WeightedRoundRobinScheduler({sid: weights.get(sid, 1) for sid in ids})
    if kind == "rate":
        fps = {int(stream["stream_id"]): float(stream["target_fps"]) for stream in config["streams"]}
        return RateDeadlineScheduler(fps, priority=priority)
    if kind == "round_robin":
        from scheduler import RoundRobinScheduler

        return RoundRobinScheduler(len(ids))
    raise SystemExit(f"unknown scheduler kind {kind!r}")


def resource_sample() -> dict:
    """One CPU / NPU / thermal sample."""
    row: dict[str, Any] = {"timestamp": time.time()}
    try:
        import psutil

        row["cpu_percent"] = psutil.cpu_percent(interval=None)
        row["rss_mb"] = round(psutil.Process().memory_info().rss / 1024 / 1024, 3)
    except ImportError:
        pass
    try:
        result = subprocess.run(
            ["sudo", "-n", "cat", "/sys/kernel/debug/rknpu/load"], capture_output=True, text=True, timeout=1.0
        )
        row["npu_load"] = result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        row["npu_load"] = ""
    return row


def run(config: dict, output_dir: Path) -> dict:
    """Run one heterogeneous replay and write the per-frame, resource and summary artifacts."""
    if decode_outputs is None or letterbox_bgr is None:
        raise SystemExit("decode_yolo_output.decode_outputs and letterbox_bgr must be importable")

    import cv2
    from rknnlite.api import RKNNLite

    streams = {stream["stream_id"]: stream for stream in config["streams"]}
    stream_ids = sorted(streams)
    duration = config["duration_s"]
    queue_depth = config["queue_depth"]
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_queues = {sid: DropOldestQueue(queue_depth) for sid in stream_ids}
    post_queues = {sid: DropOldestQueue(queue_depth) for sid in stream_ids}
    rows: list[dict] = []
    rows_lock = threading.Lock()
    queue_samples: dict[int, list[int]] = {sid: [] for sid in stream_ids}
    resource_rows: list[dict] = []
    stop = threading.Event()
    decode_done = {sid: False for sid in stream_ids}
    decode_lock = threading.Lock()
    deadline = time.perf_counter() + duration

    def decode_worker(stream_id: int) -> None:
        stream = streams[stream_id]
        capture = cv2.VideoCapture(str(stream["video"]))
        frame_id = 0
        try:
            if not capture.isOpened():
                raise RuntimeError(f"cannot open {stream['video']}")
            period = 1.0 / stream["target_fps"]
            next_tick = time.perf_counter()
            while time.perf_counter() < deadline and not stop.is_set():
                wait = next_tick - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                ok, bgr = capture.read()
                if not ok:  # loop the file instead of ending the stream early
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, bgr = capture.read()
                if not ok:
                    break
                packet = {
                    "stream_id": stream_id,
                    "frame_id": frame_id,
                    "arrival_ts": time.perf_counter(),
                    "bgr": bgr,
                    "target_fps": stream["target_fps"],
                }
                evicted = frame_queues[stream_id].put_drop_oldest(packet)
                queue_samples[stream_id].append(frame_queues[stream_id].qsize())
                if evicted is not None:
                    with rows_lock:
                        rows.append(
                            {
                                "stream_id": stream_id,
                                "frame_id": evicted["frame_id"],
                                "dropped": True,
                                "arrival_ts": evicted["arrival_ts"],
                                "selected_ts": "",
                                "infer_done_ts": "",
                                "post_done_ts": "",
                                "npu_ms": "",
                                "canvas": f"{stream['width']}x{stream['height']}",
                            }
                        )
                frame_id += 1
                next_tick += period
        except BaseException as exc:  # a stream must never take the process down
            with rows_lock:
                rows.append({"stream_id": stream_id, "frame_id": -1, "dropped": True, "arrival_ts": time.perf_counter(),
                             "selected_ts": "", "infer_done_ts": "", "post_done_ts": "", "npu_ms": "",
                             "canvas": "", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            capture.release()
            with decode_lock:
                decode_done[stream_id] = True

    def post_worker(stream_id: int) -> None:
        stream = streams[stream_id]
        while True:
            try:
                packet = post_queues[stream_id].get_nowait()
            except queue.Empty:
                # The worker outlives gaps in its own queue; it only stops once the
                # run is over and there is nothing left to drain. Polling without
                # this guard killed the thread on the very first empty poll.
                if stop.is_set() and post_queues[stream_id].qsize() == 0:
                    return
                time.sleep(0.001)
                continue
            if packet is SENTINEL:
                return
            decode_outputs(packet["outputs"], 0.25, 0.45)
            packet["post_done_ts"] = time.perf_counter()
            with rows_lock:
                rows.append(
                    {
                        "stream_id": stream_id,
                        "frame_id": packet["frame_id"],
                        "dropped": False,
                        "arrival_ts": packet["arrival_ts"],
                        "selected_ts": packet["selected_ts"],
                        "infer_done_ts": packet["infer_done_ts"],
                        "post_done_ts": packet["post_done_ts"],
                        "npu_ms": packet["npu_ms"],
                        "canvas": f"{stream['width']}x{stream['height']}",
                    }
                )
            while True:  # drain anything that arrived while we were busy
                try:
                    nxt = post_queues[stream_id].get_nowait()
                except queue.Empty:
                    break
                if nxt is SENTINEL:
                    return
                decode_outputs(nxt["outputs"], 0.25, 0.45)
                nxt["post_done_ts"] = time.perf_counter()
                with rows_lock:
                    rows.append(
                        {
                            "stream_id": stream_id,
                            "frame_id": nxt["frame_id"],
                            "dropped": False,
                            "arrival_ts": nxt["arrival_ts"],
                            "selected_ts": nxt["selected_ts"],
                            "infer_done_ts": nxt["infer_done_ts"],
                            "post_done_ts": nxt["post_done_ts"],
                            "npu_ms": nxt["npu_ms"],
                            "canvas": f"{stream['width']}x{stream['height']}",
                        }
                    )

    def sampler() -> None:
        while not stop.is_set():
            resource_rows.append(resource_sample())
            stop.wait(0.5)

    threads = [threading.Thread(target=decode_worker, args=(sid,), daemon=True) for sid in stream_ids]
    threads += [threading.Thread(target=post_worker, args=(sid,), daemon=True) for sid in stream_ids]
    for thread in threads:
        thread.start()
    threading.Thread(target=sampler, daemon=True).start()

    runners: dict[int, Any] = {}
    for sid in stream_ids:
        runner = RKNNLite(verbose=False)
        if runner.load_rknn(str(streams[sid]["model"])) != 0 or runner.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
            raise RuntimeError(f"failed to initialize {streams[sid]['model']}")
        runners[sid] = runner

    scheduler = make_scheduler(config)
    served = 0
    idle_spins = 0
    started = time.perf_counter()
    try:
        while True:
            now = time.perf_counter()
            states = {sid: {"queue_length": frame_queues[sid].qsize()} for sid in stream_ids}
            sid = scheduler.select(states, now)
            if sid < 0:
                with decode_lock:
                    finished = all(decode_done.values())
                if finished and all(frame_queues[i].qsize() == 0 for i in stream_ids):
                    break
                if now >= deadline and all(frame_queues[i].qsize() == 0 for i in stream_ids):
                    break
                idle_spins += 1
                time.sleep(0.001)
                continue
            stream = streams[sid]
            try:
                packet = frame_queues[sid].get_nowait()
            except queue.Empty:
                continue
            packet["selected_ts"] = time.perf_counter()
            canvas, scale, pad_x, pad_y = letterbox_bgr(packet["bgr"], (stream["width"], stream["height"]))
            npu_started = time.perf_counter()
            outputs = runners[sid].inference(inputs=[canvas[None, ...]], data_format="nhwc")
            packet["npu_ms"] = (time.perf_counter() - npu_started) * 1000
            packet["outputs"] = outputs
            packet["infer_done_ts"] = time.perf_counter()
            post_queues[sid].put_drop_oldest(packet)
            served += 1
            queue_samples[sid].append(frame_queues[sid].qsize())
    finally:
        for runner in runners.values():
            runner.release()
        stop.set()
    for sid in stream_ids:
        post_queues[sid].put_drop_oldest(SENTINEL)
    for thread in threads:
        thread.join(timeout=10)
    elapsed = time.perf_counter() - started

    fields = ["stream_id", "frame_id", "dropped", "arrival_ts", "selected_ts", "infer_done_ts", "post_done_ts", "npu_ms", "canvas", "error"]
    with (output_dir / "per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (int(row["stream_id"]), int(row["frame_id"]))))
    if resource_rows:
        with (output_dir / "resources.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(resource_rows[0]))
            writer.writeheader()
            writer.writerows(resource_rows)

    stream_errors = sorted(
        {str(row.get("error")) for row in rows if row.get("error")} | {""}
    )
    stream_errors = [message for message in stream_errors if message]
    summary_rows = []
    for sid in stream_ids:
        stream = streams[sid]
        selected = [row for row in rows if int(row["stream_id"]) == sid and not row["dropped"]]
        dropped = [row for row in rows if int(row["stream_id"]) == sid and row["dropped"]]
        e2e = [(float(row["post_done_ts"]) - float(row["arrival_ts"])) * 1000 for row in selected if row["post_done_ts"]]
        wait = [(float(row["selected_ts"]) - float(row["arrival_ts"])) * 1000 for row in selected if row["selected_ts"]]
        services = [float(row["selected_ts"]) for row in selected if row["selected_ts"]]
        intervals = np.diff(sorted(services)) * 1000 if len(services) > 1 else np.asarray([])
        npu = [float(row["npu_ms"]) for row in selected if row["npu_ms"]]
        summary_rows.append(
            {
                "stream_id": sid,
                "role": stream.get("role", f"CH{sid}"),
                "canvas": f"{stream['width']}x{stream['height']}",
                "target_fps": stream["target_fps"],
                "actual_fps": len(selected) / elapsed if elapsed else 0.0,
                "frames": len(selected),
                "dropped": len(dropped),
                "drop_rate": len(dropped) / max(len(selected) + len(dropped), 1),
                "queue_avg": statistics.fmean(queue_samples[sid]) if queue_samples[sid] else 0.0,
                "queue_max": max(queue_samples[sid]) if queue_samples[sid] else 0,
                "npu_mean_ms": statistics.fmean(npu) if npu else 0.0,
                "npu_p95_ms": float(np.percentile(npu, 95)) if npu else 0.0,
                "e2e_mean_ms": statistics.fmean(e2e) if e2e else 0.0,
                "e2e_p95_ms": float(np.percentile(e2e, 95)) if e2e else 0.0,
                "wait_mean_ms": statistics.fmean(wait) if wait else 0.0,
                "service_interval_mean_ms": float(intervals.mean()) if intervals.size else 0.0,
                "service_interval_p95_ms": float(np.percentile(intervals, 95)) if intervals.size else 0.0,
            }
        )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    aggregate = sum(row["actual_fps"] for row in summary_rows)
    report = {
        "label": config.get("label", output_dir.name),
        "scheduler": (config.get("scheduler") or {}).get("kind", "rate"),
        "queue_depth": queue_depth,
        "duration_s": elapsed,
        "aggregate_actual_fps": aggregate,
        "total_requested_fps": sum(row["target_fps"] for row in summary_rows),
        "inferences_served": served,
        "idle_spins": idle_spins,
        "inferences_per_second": served / elapsed if elapsed else 0.0,
        "stream_errors": stream_errors,
        "streams": summary_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, help="override the config duration")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.duration:
        config["duration_s"] = args.duration
    # Resolve stream paths relative to the config file so the bundle is relocatable.
    base = args.config.parent
    for stream in config["streams"]:
        for field in ("video", "model"):
            candidate = Path(stream[field])
            if not candidate.is_absolute():
                stream[field] = str(base / candidate)
    report = run(config, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
