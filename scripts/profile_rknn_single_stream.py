"""Single-stream RKNN latency decomposition profiler for YOLO11 raw-head models."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

if __package__ in (None, ""):  # allow `python tools/profile_rknn_single_stream.py` from the repo root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from competition_ai.input_shape import (
    normalize_shape as canonical_normalize_shape,
    validate_model_shape as canonical_validate_model_shape,
)


COCO_CLASSES = 80
DFL_BINS = 16


def gstreamer_pipeline(video: str | Path) -> str:
    """Return the RK3566 MPP H.264 file-decoding pipeline for OpenCV."""
    location = str(video).replace("\\", "/")
    return (
        f"filesrc location={location} ! qtdemux ! h264parse ! mppvideodec ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink drop=false sync=false"
    )


def opencv_has_gstreamer() -> bool:
    import cv2

    return any("GStreamer" in line and "YES" in line for line in cv2.getBuildInformation().splitlines())


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Return fixed precision statistics for one non-empty latency series."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty series")
    return {
        "mean": round(float(array.mean()), 3),
        "median": round(float(np.median(array)), 3),
        "p95": round(float(np.percentile(array, 95)), 3),
        "std": round(float(array.std()), 3),
    }


def normalize_input_shape(size: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Return a model input as ``(width, height)``.

    Delegates to the canonical contract in :mod:`competition_ai.input_shape` so the
    board profiler, the competition detector and the PC evaluators can never drift
    apart on what a "320" input means.
    """
    return canonical_normalize_shape(size)


def validate_model_shape(shape: int | str | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Return ``(width, height)`` after enforcing the RKNN graph constraints.

    Accepts ``320``, ``"320"``, ``"512x288"`` (also ``X``/``×``/``*`` separators)
    or a two-item sequence. Both sides must be at least 32 and divisible by 32,
    which every YOLO stride requires from a rectangular input.
    """
    return canonical_validate_model_shape(shape)


def letterbox_bgr(image: np.ndarray, size: int | tuple[int, int] | list[int]) -> tuple[np.ndarray, float, int, int]:
    """Resize BGR input to an RGB NHWC model input without cropping.

    ``size`` is either a square integer or ``(width, height)``. Padding and
    returned coordinates are in model-space pixels.
    """
    import cv2

    model_width, model_height = normalize_input_shape(size)
    height, width = image.shape[:2]
    scale = min(model_width / width, model_height / height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((model_height, model_width, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (model_width - resized_width) // 2, (model_height - resized_height) // 2
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    # ONNX from Ultralytics consumes RGB. RKNN input remains uint8 NHWC.
    return np.ascontiguousarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)), scale, pad_x, pad_y


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    import cv2

    if boxes.size == 0:
        return []
    xywh = np.column_stack(
        [boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]]
    )
    indices = cv2.dnn.NMSBoxes(
        xywh.tolist(), scores.astype(float).tolist(), score_threshold=0.0, nms_threshold=iou_threshold
    )
    return np.asarray(indices, dtype=np.int32).reshape(-1).tolist()


def _as_nchw(output: np.ndarray) -> np.ndarray:
    # Keep the RKNN output dtype; cast only the small survivor subset below.
    array = np.asarray(output)
    if array.ndim != 4:
        raise ValueError(f"raw head output must be 4-D, got {array.shape}")
    if array.shape[1] in (DFL_BINS * 4, COCO_CLASSES):
        return array
    if array.shape[-1] in (DFL_BINS * 4, COCO_CLASSES):
        return array.transpose(0, 3, 1, 2)
    raise ValueError(f"cannot identify channel dimension in {array.shape}")


def decode_yolo11_raw(
    outputs: Sequence[np.ndarray],
    input_size: int | tuple[int, int] | list[int],
    confidence_threshold: float,
    iou_threshold: float,
    scale: float,
    pad_x: int,
    pad_y: int,
) -> list[list[float]]:
    """CPU DFL/decode, sigmoid, filtering and class-wise NMS for six raw outputs."""
    if len(outputs) != 6:
        raise ValueError(f"expected six raw head outputs, got {len(outputs)}")
    model_width, model_height = normalize_input_shape(input_size)
    candidates: list[np.ndarray] = []
    dfl_weights = np.arange(DFL_BINS, dtype=np.float32)
    confidence_logit = math.log(confidence_threshold / (1.0 - confidence_threshold))
    for reg_raw, cls_raw in zip(outputs[0::2], outputs[1::2]):
        reg, classes = _as_nchw(reg_raw), _as_nchw(cls_raw)
        _, _, grid_h, grid_w = reg.shape
        if classes.shape != (1, COCO_CLASSES, grid_h, grid_w):
            raise ValueError(f"incompatible raw outputs: {reg.shape}, {classes.shape}")
        if model_height % grid_h or model_width % grid_w:
            raise ValueError(f"grid {grid_w}x{grid_h} does not divide input {(model_width, model_height)}")
        stride_x, stride_y = model_width / grid_w, model_height / grid_h
        class_logits = np.asarray(classes[0])
        max_logits = class_logits.max(axis=0)
        ys, xs = np.where(max_logits >= confidence_logit)
        if not len(xs):
            continue
        selected_logits = class_logits[:, ys, xs]
        class_ids = selected_logits.argmax(axis=0)
        confidences = 1.0 / (1.0 + np.exp(-selected_logits[class_ids, np.arange(len(xs))].astype(np.float32)))
        # With paired advanced indices NumPy already returns [n_candidates, 64].
        # Transposing here swaps candidate and channel axes; the subsequent
        # reshape then mixes four DFL edges across different anchors and
        # produces grossly oversized boxes.  Keep the candidate-major layout.
        selected_reg = np.asarray(reg[0, :, ys, xs], dtype=np.float32)
        distances = selected_reg.reshape(-1, 4, DFL_BINS)
        distances -= distances.max(axis=2, keepdims=True)
        distances = np.exp(distances)
        distances /= distances.sum(axis=2, keepdims=True)
        distances = (distances * dfl_weights.reshape(1, 1, DFL_BINS)).sum(axis=2)
        x_center = (xs.astype(np.float32) + 0.5) * stride_x
        y_center = (ys.astype(np.float32) + 0.5) * stride_y
        xyxy = np.column_stack(
            [
                x_center - distances[:, 0] * stride_x,
                y_center - distances[:, 1] * stride_y,
                x_center + distances[:, 2] * stride_x,
                y_center + distances[:, 3] * stride_y,
            ]
        )
        # Restore coordinates from letterboxed model space to the 1080p BGR frame.
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale
        candidates.append(np.column_stack([xyxy, confidences, class_ids]))
    if not candidates:
        return []
    detections = np.concatenate(candidates)
    result: list[list[float]] = []
    for class_id in np.unique(detections[:, 5].astype(np.int32)):
        same_class = detections[detections[:, 5] == class_id]
        for index in _nms(same_class[:, :4], same_class[:, 4], iou_threshold):
            result.append([float(value) for value in same_class[index]])
    return sorted(result, key=lambda row: row[4], reverse=True)


def _read_npu_load() -> str:
    try:
        return Path("/sys/kernel/debug/rknpu/load").read_text(encoding="utf-8").strip().replace("\n", " | ")
    except (OSError, PermissionError):
        try:
            result = subprocess.run(
                ["sudo", "-n", "cat", "/sys/kernel/debug/rknpu/load"],
                check=False,
                capture_output=True,
                text=True,
                timeout=0.2,
            )
            return result.stdout.strip().replace("\n", " | ") if result.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            return ""


def _read_temperatures() -> dict[str, float]:
    readings: dict[str, float] = {}
    for path_string in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            value = float(Path(path_string).read_text(encoding="utf-8").strip())
            readings[Path(path_string).parent.name] = value / 1000 if value > 1000 else value
        except (OSError, ValueError):
            continue
    return readings


class ResourceSampler:
    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.rows: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._phase = ""

    def start(self, phase: str) -> None:
        import psutil

        self._phase = phase
        self._started = time.perf_counter()
        psutil.cpu_percent(percpu=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import psutil

        while not self._stop.is_set():
            row: dict[str, Any] = {
                "elapsed_s": round(time.perf_counter() - self._started, 3),
                "phase": self._phase,
                "npu_load": _read_npu_load(),
            }
            row.update({f"cpu_core_{index}_percent": value for index, value in enumerate(psutil.cpu_percent(percpu=True))})
            row.update({f"{zone}_c": value for zone, value in _read_temperatures().items()})
            self.rows.append(row)
            self._stop.wait(self.interval_seconds)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 1)


class QueueStats:
    def __init__(self, names: Sequence[str]) -> None:
        self._totals = {name: [0, 0] for name in names}
        self._lock = threading.Lock()

    def observe(self, name: str, value: int) -> None:
        with self._lock:
            self._totals[name][0] += value
            self._totals[name][1] += 1

    def rows(self) -> list[dict[str, float | str]]:
        with self._lock:
            return [{"queue": name, "average_length": round(total / max(count, 1), 3), "samples": count} for name, (total, count) in self._totals.items()]


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = ("decode_ms", "preprocess_ms", "npu_infer_ms", "postprocess_ms")
    lines = ["# RK3566 YOLO11n INT8 Single-Stream Profile", "", "| resolution | frames | " + " | ".join(f"{field} mean/median/p95/std (ms)" for field in fields) + " |", "| --- | ---: | " + " | ".join("---" for _ in fields) + " |"]
    for resolution in sorted({int(row["resolution"]) for row in rows}):
        subset = [row for row in rows if int(row["resolution"]) == resolution]
        statistics_text = []
        for field in fields:
            values = [float(row[field]) for row in subset]
            stats = summarize(values)
            statistics_text.append(f"{stats['mean']:.3f}/{stats['median']:.3f}/{stats['p95']:.3f}/{stats['std']:.3f}")
        lines.append(f"| {resolution} | {len(subset)} | " + " | ".join(statistics_text) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_thermal_svg(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    values = [float(row["npu_infer_ms"]) for row in rows]
    if not values:
        return
    width, height, margin = 900, 360, 45
    low, high = min(values), max(values)
    span = max(high - low, 1.0)
    points = []
    for index, row in enumerate(rows):
        x = margin + index * (width - 2 * margin) / max(len(rows) - 1, 1)
        y = height - margin - (float(row["npu_infer_ms"]) - low) * (height - 2 * margin) / span
        points.append(f"{x:.1f},{y:.1f}")
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/><line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/><polyline fill="none" stroke="#1877b8" stroke-width="1" points="{' '.join(points)}"/><text x="{margin}" y="22" font-size="16">640 INT8 NPU inference latency over time</text><text x="5" y="{margin}" font-size="12">{high:.1f} ms</text><text x="5" y="{height-margin}" font-size="12">{low:.1f} ms</text></svg>'''
    path.write_text(svg, encoding="utf-8")


def run_profile(
    video: Path,
    model_paths: dict[int, Path],
    frames: int,
    output_dir: Path,
    thermal_seconds: int,
    switch_frames: int,
) -> None:
    import cv2
    from rknnlite.api import RKNNLite

    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    resources = ResourceSampler()
    resources.start("profile")
    try:
        for resolution, model_path in sorted(model_paths.items()):
            rknn = RKNNLite(verbose=False)
            try:
                if rknn.load_rknn(str(model_path)) != 0 or rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
                    raise RuntimeError(f"failed to initialize {model_path}")
                capture = cv2.VideoCapture(str(video))
                if not capture.isOpened():
                    raise RuntimeError(f"cannot open video {video}")
                for frame_id in range(frames):
                    started = time.perf_counter()
                    ok, bgr = capture.read()
                    decode_ms = (time.perf_counter() - started) * 1000
                    if not ok:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        started = time.perf_counter()
                        ok, bgr = capture.read()
                        decode_ms = (time.perf_counter() - started) * 1000
                    if not ok:
                        raise RuntimeError("video cannot supply a frame")
                    started = time.perf_counter()
                    image, scale, pad_x, pad_y = letterbox_bgr(bgr, resolution)
                    preprocess_ms = (time.perf_counter() - started) * 1000
                    started = time.perf_counter()
                    outputs = rknn.inference(inputs=[image[None, ...]], data_format="nhwc")
                    npu_infer_ms = (time.perf_counter() - started) * 1000
                    started = time.perf_counter()
                    _ = decode_yolo11_raw(outputs, resolution, 0.25, 0.45, scale, pad_x, pad_y)
                    postprocess_ms = (time.perf_counter() - started) * 1000
                    all_rows.append({"frame_id": frame_id, "resolution": resolution, "decode_ms": round(decode_ms, 3), "preprocess_ms": round(preprocess_ms, 3), "npu_infer_ms": round(npu_infer_ms, 3), "postprocess_ms": round(postprocess_ms, 3)})
                capture.release()
            finally:
                rknn.release()
    finally:
        resources.stop()
    _write_csv(output_dir / "per_frame_latency.csv", all_rows, ["frame_id", "resolution", "decode_ms", "preprocess_ms", "npu_infer_ms", "postprocess_ms"])
    _write_summary(output_dir / "summary.md", all_rows)
    _write_csv(output_dir / "resource_samples.csv", resources.rows)
    if switch_frames:
        _run_model_switch(video, model_paths[320], model_paths[640], output_dir, switch_frames)
    if thermal_seconds:
        _run_thermal(video, model_paths[640], output_dir, thermal_seconds)


def _run_model_switch(video: Path, model_320: Path, model_640: Path, output_dir: Path, cycles: int) -> None:
    """Measure alternating inference with both RKNN contexts resident."""
    import cv2
    from rknnlite.api import RKNNLite

    runners: dict[int, Any] = {}
    capture = cv2.VideoCapture(str(video))
    rows: list[dict[str, Any]] = []
    try:
        for resolution, model_path in ((320, model_320), (640, model_640)):
            runner = RKNNLite(verbose=False)
            if runner.load_rknn(str(model_path)) != 0 or runner.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
                raise RuntimeError(f"failed to initialize {model_path}")
            runners[resolution] = runner
        for switch_index in range(cycles):
            resolution = 320 if switch_index % 2 == 0 else 640
            ok, bgr = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, bgr = capture.read()
            if not ok:
                raise RuntimeError("video cannot supply a frame during model switch test")
            image, *_ = letterbox_bgr(bgr, resolution)
            started = time.perf_counter()
            runners[resolution].inference(inputs=[image[None, ...]], data_format="nhwc")
            rows.append({
                "switch_index": switch_index,
                "resolution": resolution,
                "npu_infer_ms": round((time.perf_counter() - started) * 1000, 3),
            })
    finally:
        capture.release()
        for runner in runners.values():
            runner.release()
    _write_csv(output_dir / "model_switch_320_640.csv", rows, ["switch_index", "resolution", "npu_infer_ms"])
    summary_rows = []
    for resolution in (320, 640):
        values = [float(row["npu_infer_ms"]) for row in rows if row["resolution"] == resolution]
        stats = summarize(values)
        summary_rows.append({"resolution": resolution, **stats})
    _write_csv(output_dir / "model_switch_summary.csv", summary_rows, ["resolution", "mean", "median", "p95", "std"])


def _run_thermal(video: Path, model_path: Path, output_dir: Path, seconds: int) -> None:
    import cv2
    from rknnlite.api import RKNNLite

    rknn = RKNNLite(verbose=False)
    rows: list[dict[str, Any]] = []
    sampler = ResourceSampler()
    try:
        if rknn.load_rknn(str(model_path)) != 0 or rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
            raise RuntimeError(f"failed to initialize {model_path}")
        capture = cv2.VideoCapture(str(video))
        sampler.start("thermal_640")
        deadline, frame_id = time.perf_counter() + seconds, 0
        while time.perf_counter() < deadline:
            ok, bgr = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, bgr = capture.read()
            image, *_ = letterbox_bgr(bgr, 640)
            started = time.perf_counter()
            rknn.inference(inputs=[image[None, ...]], data_format="nhwc")
            rows.append({"frame_id": frame_id, "elapsed_s": round(time.perf_counter() - sampler._started, 3), "npu_infer_ms": round((time.perf_counter() - started) * 1000, 3)})
            frame_id += 1
        capture.release()
    finally:
        sampler.stop()
        rknn.release()
    _write_csv(output_dir / "thermal_640_npu_latency.csv", rows)
    _write_thermal_svg(output_dir / "thermal_640_npu_latency.svg", rows)
    _write_csv(output_dir / "thermal_resource_samples.csv", sampler.rows)


def _run_pipeline(
    video: Path,
    model_path: Path,
    output_dir: Path,
    frames: int,
    resolution: int | str | tuple[int, int] = 640,
    queue_size: int = 4,
) -> None:
    """Measure one stream through four bounded worker queues.

    ``resolution`` is a square size or a ``(width, height)`` pair, so a 16:9
    input such as 512x288 runs through exactly the same bounded pipeline and
    only changes the letterbox canvas and the decode strides.
    """
    import cv2
    from rknnlite.api import RKNNLite

    model_width, model_height = validate_model_shape(resolution)
    input_shape = (model_width, model_height)
    label = f"{model_width}x{model_height}"

    queues = [queue.Queue(maxsize=queue_size) for _ in range(3)]
    stats = QueueStats(("decode_to_preprocess", "preprocess_to_npu", "npu_to_postprocess"))
    errors: queue.Queue[BaseException] = queue.Queue()
    rows: list[dict[str, Any]] = []
    stop = object()

    def put(index: int, item: Any) -> None:
        queues[index].put(item)
        stats.observe(("decode_to_preprocess", "preprocess_to_npu", "npu_to_postprocess")[index], queues[index].qsize())

    def decode_worker() -> None:
        capture = cv2.VideoCapture(str(video))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"cannot open video {video}")
            for frame_id in range(frames):
                started = time.perf_counter()
                ok, bgr = capture.read()
                if not ok:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError("video cannot supply a frame")
                put(0, {"frame_id": frame_id, "bgr": bgr, "decode_ms": (time.perf_counter() - started) * 1000, "started": started})
        except BaseException as exc:
            errors.put(exc)
        finally:
            capture.release()
            put(0, stop)

    def preprocess_worker() -> None:
        try:
            while True:
                item = queues[0].get()
                if item is stop:
                    put(1, stop)
                    return
                started = time.perf_counter()
                image, scale, pad_x, pad_y = letterbox_bgr(item["bgr"], input_shape)
                item.update({"image": image, "scale": scale, "pad_x": pad_x, "pad_y": pad_y, "preprocess_ms": (time.perf_counter() - started) * 1000})
                del item["bgr"]
                put(1, item)
        except BaseException as exc:
            errors.put(exc)

    def npu_worker() -> None:
        rknn = RKNNLite(verbose=False)
        try:
            if rknn.load_rknn(str(model_path)) != 0 or rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
                raise RuntimeError(f"failed to initialize {model_path}")
            while True:
                item = queues[1].get()
                if item is stop:
                    put(2, stop)
                    return
                started = time.perf_counter()
                item["outputs"] = rknn.inference(inputs=[item["image"][None, ...]], data_format="nhwc")
                item["npu_infer_ms"] = (time.perf_counter() - started) * 1000
                del item["image"]
                put(2, item)
        except BaseException as exc:
            errors.put(exc)
        finally:
            rknn.release()

    def postprocess_worker() -> None:
        try:
            while True:
                item = queues[2].get()
                if item is stop:
                    return
                started = time.perf_counter()
                decode_yolo11_raw(item["outputs"], input_shape, 0.25, 0.45, item["scale"], item["pad_x"], item["pad_y"])
                item["postprocess_ms"] = (time.perf_counter() - started) * 1000
                item.pop("outputs", None)
                item["end_to_end_ms"] = (time.perf_counter() - item["started"]) * 1000
                rows.append({key: round(item[key], 3) for key in ("frame_id", "decode_ms", "preprocess_ms", "npu_infer_ms", "postprocess_ms", "end_to_end_ms")})
        except BaseException as exc:
            errors.put(exc)

    sampler = ResourceSampler()
    sampler.start(f"pipeline_{label}")
    workers = [threading.Thread(target=decode_worker), threading.Thread(target=preprocess_worker), threading.Thread(target=npu_worker), threading.Thread(target=postprocess_worker)]
    started = time.perf_counter()
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    elapsed = time.perf_counter() - started
    sampler.stop()
    if not errors.empty():
        raise errors.get()
    rows.sort(key=lambda row: row["frame_id"])
    _write_csv(output_dir / f"pipeline_{label}_per_frame_latency.csv", rows, ["frame_id", "decode_ms", "preprocess_ms", "npu_infer_ms", "postprocess_ms", "end_to_end_ms"])
    _write_csv(output_dir / "pipeline_queue_stats.csv", stats.rows(), ["queue", "average_length", "samples"])
    _write_csv(output_dir / f"pipeline_{label}_resource_samples.csv", sampler.rows)
    output_dir.joinpath("pipeline_summary.md").write_text(
        f"# {label} pipeline profile\n\n- frames: {len(rows)}\n- wall time: {elapsed:.3f} s\n- throughput: {len(rows) / elapsed:.3f} FPS\n\n" + "\n".join(f"- {row['queue']}: average queue length {row['average_length']:.3f}" for row in stats.rows()) + "\n", encoding="utf-8"
    )


def make_raw_head_onnx(source: Path, destination: Path) -> None:
    """Expose YOLO11's three raw DFL/class head pairs and remove decoded output."""
    import onnx
    from onnx import TensorProto, helper, shape_inference

    model = shape_inference.infer_shapes(onnx.load(source))
    graph = model.graph
    value_infos = {value.name: value for value in (*graph.input, *graph.value_info, *graph.output)}
    names = [
        "/model.23/cv2.0/cv2.0.2/Conv_output_0", "/model.23/cv3.0/cv3.0.2/Conv_output_0",
        "/model.23/cv2.1/cv2.1.2/Conv_output_0", "/model.23/cv3.1/cv3.1.2/Conv_output_0",
        "/model.23/cv2.2/cv2.2.2/Conv_output_0", "/model.23/cv3.2/cv3.2.2/Conv_output_0",
    ]
    if any(name not in value_infos for name in names):
        raise ValueError("unexpected YOLO11 ONNX graph: raw head tensors were not found")
    del graph.output[:]
    for index, name in enumerate(names):
        dims = [dimension.dim_value for dimension in value_infos[name].type.tensor_type.shape.dim]
        graph.output.append(helper.make_tensor_value_info(f"raw_head_{index}", TensorProto.FLOAT, dims))
        graph.node.append(helper.make_node("Identity", [name], [f"raw_head_{index}"], name=f"profile_raw_head_{index}"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--model", action="append", default=[], metavar="SIZE=PATH")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=Path("profile_output"))
    parser.add_argument("--thermal-seconds", type=int, default=300)
    parser.add_argument("--switch-frames", type=int, default=200, help="Alternating 320/640 inferences; 0 disables")
    parser.add_argument("--pipeline-frames", type=int, default=0, help="Run bounded four-thread pipeline; 0 disables")
    parser.add_argument(
        "--pipeline-size",
        default="640",
        metavar="SIZE|WxH",
        help="pipeline input: 320, 416, 512x288, ... (default 640)",
    )
    parser.add_argument(
        "--pipeline-model",
        type=Path,
        help="RKNN model for --pipeline-size when it is not one of the 320/416/640 models",
    )
    parser.add_argument("--make-raw-onnx", type=Path, metavar="SOURCE")
    parser.add_argument("--raw-onnx-output", type=Path, metavar="DESTINATION")
    args = parser.parse_args()
    if args.make_raw_onnx:
        if not args.raw_onnx_output:
            parser.error("--raw-onnx-output is required with --make-raw-onnx")
        make_raw_head_onnx(args.make_raw_onnx, args.raw_onnx_output)
        return
    if not args.video or len(args.model) != 3:
        parser.error("--video and exactly three --model SIZE=PATH arguments are required")
    model_paths: dict[int, Path] = {}
    for value in args.model:
        size_text, path_text = value.split("=", 1)
        model_paths[int(size_text)] = Path(path_text)
    if sorted(model_paths) != [320, 416, 640]:
        parser.error("models must be supplied for 320, 416 and 640")
    if args.frames < 1 or args.switch_frames < 0 or args.thermal_seconds < 0 or args.pipeline_frames < 0:
        parser.error("--frames must be positive; optional run lengths must be non-negative")
    run_profile(args.video, model_paths, args.frames, args.output_dir, args.thermal_seconds, args.switch_frames)
    if args.pipeline_frames:
        try:
            pipeline_shape = validate_model_shape(args.pipeline_size)
        except ValueError as exc:
            parser.error(str(exc))
        if args.pipeline_model:
            pipeline_model = args.pipeline_model
        elif pipeline_shape[0] == pipeline_shape[1] and pipeline_shape[0] in model_paths:
            pipeline_model = model_paths[pipeline_shape[0]]
        else:
            parser.error(
                f"--pipeline-model is required for {pipeline_shape[0]}x{pipeline_shape[1]}: "
                "only 320/416/640 are covered by --model"
            )
        _run_pipeline(args.video, pipeline_model, args.output_dir, args.pipeline_frames, pipeline_shape)


if __name__ == "__main__":
    main()
