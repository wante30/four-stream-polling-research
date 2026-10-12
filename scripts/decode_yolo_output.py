"""Decode split YOLO [xywh, class scores] outputs into JSON-friendly boxes."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_current = (boxes[current, 2] - boxes[current, 0]) * (boxes[current, 3] - boxes[current, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = intersection / np.maximum(area_current + area_rest - intersection, 1e-6)
        order = rest[iou <= iou_threshold]
    return keep


def decode_outputs(
    outputs: Sequence[np.ndarray],
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    score_logits: bool = False,
) -> list[dict[str, object]]:
    """Decode RKNN outputs shaped [1,4,N] and [1,C,N].

    ``score_logits=True`` for models whose scores branch was cut before the
    sigmoid (see split_yolo_output.py --scores-mode logits): the scores tensor
    then holds log-odds and is converted to probabilities here.
    """
    if len(outputs) != 2:
        raise ValueError(f"expected two outputs, got {len(outputs)}")
    boxes, scores = (np.asarray(output, dtype=np.float32) for output in outputs)
    if boxes.ndim != 3 or boxes.shape[0] != 1 or boxes.shape[1] != 4:
        raise ValueError(f"expected boxes [1,4,N], got {boxes.shape}")
    if scores.ndim != 3 or scores.shape[0] != 1 or scores.shape[2] != boxes.shape[2]:
        raise ValueError(f"expected scores [1,C,{boxes.shape[2]}], got {scores.shape}")

    xywh = boxes[0].T
    xyxy = np.column_stack(
        [xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
         xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2]
    )
    class_scores = scores[0].T
    if score_logits:
        class_scores = 1.0 / (1.0 + np.exp(-class_scores))
    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
    valid = np.flatnonzero(confidences >= confidence_threshold)
    detections: list[dict[str, object]] = []
    for class_id in np.unique(class_ids[valid]):
        class_valid = valid[class_ids[valid] == class_id]
        kept = _nms(xyxy[class_valid], confidences[class_valid], iou_threshold)
        for index in class_valid[kept]:
            detections.append({
                "class_id": int(class_ids[index]),
                "confidence": float(confidences[index]),
                "bbox": [float(value) for value in xyxy[index]],
            })
    return sorted(detections, key=lambda item: float(item["confidence"]), reverse=True)
