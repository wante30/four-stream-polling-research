"""Frozen MOT17 evaluation helpers for full-rate-oracle comparisons."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

IOU_THRESHOLD = 0.5
ORACLE_CONFIDENCE = 0.25
VISIBILITY_THRESHOLD = 0.25


def load_memmap_manifest(path: Path) -> dict[int, dict[str, Any]]:
    """Load per-stream image dimensions and letterbox metadata.

    The manifest is the single source of truth for coordinate conversion;
    evaluators must not reconstruct scale/padding from hard-coded 1920x1080
    assumptions.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload if isinstance(payload, list) else payload.get("sequences", payload)
    if not isinstance(entries, list):
        raise ValueError(f"invalid MOT17 manifest: {path}")
    result: dict[int, dict[str, Any]] = {}
    for stream_id, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"invalid manifest entry {stream_id}")
        result[stream_id] = entry
    return result


def model_bbox_to_original(box: Iterable[float], transform: dict[str, Any]) -> list[float]:
    """Convert xyxy from model-square coordinates using manifest metadata."""
    x1, y1, x2, y2 = [float(x) for x in box][:4]
    scale = float(transform["scale"])
    if scale <= 0:
        raise ValueError("manifest letterbox scale must be positive")
    px, py = float(transform["pad_x"]), float(transform["pad_y"])
    return [(x1 - px) / scale, (y1 - py) / scale,
            (x2 - px) / scale, (y2 - py) / scale]


def clip_and_validate_bbox(box: Iterable[float], width: float, height: float) -> tuple[list[float], bool]:
    """Clip decoded boxes to image bounds, then assert the invariant.

    DFL regression can produce sub-pixel overshoot at image borders.  Clipping
    is part of post-processing; the returned box is guaranteed to be in the
    original-image coordinate system, and the boolean records whether clipping
    occurred for diagnostics.
    """
    values = [float(x) for x in box][:4]
    clipped = [min(max(values[0], 0.0), width), min(max(values[1], 0.0), height),
               min(max(values[2], 0.0), width), min(max(values[3], 0.0), height)]
    changed = any(abs(a - b) > 1e-6 for a, b in zip(values, clipped))
    if not (0.0 <= clipped[0] <= width and 0.0 <= clipped[2] <= width and
            0.0 <= clipped[1] <= height and 0.0 <= clipped[3] <= height):
        raise ValueError(f"bbox outside original image after conversion: {values}")
    return clipped, changed


def box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a][:4]
    bx1, by1, bx2, by2 = [float(x) for x in b][:4]
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def read_mot_gt(path: Path, *, stats: dict[str, int] | None = None) -> dict[int, list[dict[str, float | int]]]:
    """Read only marked pedestrian rows (mark=1, class=1)."""
    result: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    before_tracks: set[int] = set(); after_tracks: set[int] = set()
    before_boxes = after_boxes = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in handle:
            fields = raw.strip().split(",")
            if len(fields) < 9:
                continue
            frame, track = int(float(fields[0])), int(float(fields[1]))
            x, y, width, height = [float(v) for v in fields[2:6]]
            mark, cls, visibility = int(float(fields[6])), int(float(fields[7])), float(fields[8])
            before_tracks.add(track); before_boxes += 1
            if mark != 1 or cls != 1:
                continue
            after_tracks.add(track); after_boxes += 1
            result[frame].append({"track_id": track, "x1": x, "y1": y,
                                  "x2": x + width, "y2": y + height,
                                  "class_id": cls, "visibility": visibility})
    if stats is not None:
        stats.update({"tracks_before": len(before_tracks), "boxes_before": before_boxes,
                      "tracks_after": len(after_tracks), "boxes_after": after_boxes})
    return dict(result)


def match_boxes_to_tracks(boxes: list[list[float]], tracks: list[dict[str, float | int]], threshold: float = IOU_THRESHOLD) -> dict[int, tuple[list[float], float]]:
    """Maximum-IoU one-to-one assignment using Hungarian matching."""
    if not boxes or not tracks:
        return {}
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError("scipy is required for frozen Hungarian matching") from exc
    matrix = np.asarray([[box_iou(box, [t["x1"], t["y1"], t["x2"], t["y2"]]) for box in boxes] for t in tracks], dtype=np.float64)
    track_indices, box_indices = linear_sum_assignment(-matrix)
    return {int(tracks[ti]["track_id"]): (boxes[bi], float(matrix[ti, bi]))
            for ti, bi in zip(track_indices, box_indices) if matrix[ti, bi] >= threshold}


def load_detections(path: Path) -> dict[tuple[int, int], list[list[float]]]:
    detections: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if float(row.get("conf", 0.0)) < ORACLE_CONFIDENCE:
                continue
            stream = int(row.get("stream_id", row.get("stream", 0)))
            detections[(stream, int(row["frame_id"]))].append(
                [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]),
                 float(row.get("conf", 0.0)), float(row.get("cls", 0))])
    return dict(detections)


def percentile(values: Iterable[float], q: float) -> float | None:
    vals = sorted(float(v) for v in values)
    return float(np.percentile(vals, q)) if vals else None


def jain(values: Iterable[int | float]) -> float:
    numbers = [float(v) for v in values]
    denom = len(numbers) * sum(v * v for v in numbers)
    return sum(numbers) ** 2 / denom if denom else 0.0


def _summary(values: list[float], scale: float = 1.0) -> dict[str, float | None]:
    vals = [v * scale for v in values]
    return {"mean": float(np.mean(vals)) if vals else None, "p50": percentile(vals, 50),
            "p95": percentile(vals, 95), "max": max(vals) if vals else None}


def evaluate(oracle: dict[tuple[int, int], list[list[float]]], scheduled: dict[tuple[int, int], list[list[float]]],
             gt_by_stream_frame: dict[tuple[int, int], list[dict[str, float | int]]],
             selected_frames: dict[int, list[tuple[int, float]]], target_fps: float, duration: float,
             *, loop_starts: dict[int, set[int]] | None = None,
             frame_period_ms: float | None = None) -> dict[str, Any]:
    """Evaluate scheduling loss against oracle detections.

    GT keys must use replay-frame IDs.  If loop namespaces are needed, the
    caller should use distinct track IDs for each loop in the GT records.
    """
    loop_starts = loop_starts or {}
    period_ms = frame_period_ms if frame_period_ms is not None else 1000.0 / target_fps
    oracle_matches: dict[tuple[int, int], dict[int, tuple[list[float], float]]] = {}
    scheduled_matches: dict[tuple[int, int], dict[int, tuple[list[float], float]]] = {}
    oracle_objects: set[tuple[int, int]] = set(); first_oracle: dict[tuple[int, int], int] = {}
    first_scheduled: dict[tuple[int, int], int] = {}
    oracle_pairs = covered_pairs = 0
    for key, tracks in gt_by_stream_frame.items():
        om = match_boxes_to_tracks(oracle.get(key, []), tracks)
        sm = match_boxes_to_tracks(scheduled.get(key, []), tracks)
        oracle_matches[key], scheduled_matches[key] = om, sm
        stream, frame = key
        for tid in om:
            obj = (stream, int(tid)); oracle_objects.add(obj)
            first_oracle[obj] = min(frame, first_oracle.get(obj, frame)); oracle_pairs += 1
            if tid in sm and box_iou(om[tid][0], sm[tid][0]) >= IOU_THRESHOLD:
                covered_pairs += 1
        for tid in sm:
            obj = (stream, int(tid)); first_scheduled[obj] = min(frame, first_scheduled.get(obj, frame))
    kinds = {obj: ("startup" if first in loop_starts.get(obj[0], set()) else "arrival")
             for obj, first in first_oracle.items()}
    delays = {"arrival": [], "startup": []}
    for obj in oracle_objects:
        if obj in first_scheduled:
            delays[kinds[obj]].append(first_scheduled[obj] - first_oracle[obj])
    missed = [obj for obj in oracle_objects if obj not in first_scheduled]
    counts = [len(v) for v in selected_frames.values()]
    intervals = {stream: _summary([(stamps[i] - stamps[i - 1]) for i in range(1, len(stamps))], 1000.0)
                 for stream, stamps in ((s, sorted(ts for _, ts in values)) for s, values in selected_frames.items())}
    all_delays = delays["arrival"] + delays["startup"]
    return {
        "protocol": {"iou_threshold": IOU_THRESHOLD, "oracle_confidence": ORACLE_CONFIDENCE,
                     "visibility_threshold": VISIBILITY_THRESHOLD, "matching": "hungarian_max_iou",
                     "gt_filter": "mark==1 and class==1", "fdd_split": "per_loop_startup_vs_arrival"},
        "objects_oracle": len(oracle_objects), "objects_hit": len(oracle_objects) - len(missed),
        "per_object_miss_rate": len(missed) / len(oracle_objects) if oracle_objects else 0.0,
        "fdd": {kind: {"frames": _summary(vals), "ms": _summary(vals, period_ms)} for kind, vals in delays.items()},
        "first_detection_delay_frames_mean": float(np.mean(all_delays)) if all_delays else None,
        "first_detection_delay_frames_p95": percentile(all_delays, 95),
        "first_detection_delay_ms_mean": float(np.mean(all_delays) * period_ms) if all_delays else None,
        "fdd_censored_rate": len(missed) / len(oracle_objects) if oracle_objects else 0.0,
        "object_frame_recall": covered_pairs / oracle_pairs if oracle_pairs else 0.0,
        "per_stream_service_interval": intervals, "jain_fairness": jain(counts),
        "inferred_total": sum(counts), "system_throughput_fps": sum(counts) / duration if duration else 0.0,
    }
