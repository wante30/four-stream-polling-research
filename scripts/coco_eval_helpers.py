"""Pure helpers for a COCO-style evaluation of the frozen BDD100K holdout.

Kept free of Ultralytics and pycocotools imports so the mapping, the ground-truth
construction and the metric extraction can be unit tested without a GPU or a
model file. ``tools/evaluate_bdd_coco.py`` wires these to a real checkpoint.

The class mapping is deliberately the one the project has always used, so the new
COCO numbers describe the same detector the historical recall numbers did:

    pedestrian -> person      bicycle -> bicycle     car -> car
    motorcycle -> motorcycle  bus -> bus             truck -> truck
    traffic light -> traffic light

BDD100K's ``rider``, ``traffic sign``, ``train``, ``trailer``, ``other vehicle``
and ``other person`` are dropped, exactly as before. ``rider`` is a person on a
two-wheeler; folding it into ``person`` would inflate pedestrian recall and break
comparability with the frozen reports, so it stays out and is reported as dropped.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

# The seven competition classes, in the order used by every report in this repo.
CLASSES = ["car", "truck", "bus", "person", "motorcycle", "bicycle", "traffic light"]

# BDD100K class name -> competition class. Anything absent is dropped.
BDD_MAP = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "pedestrian": "person",
    "person": "person",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "traffic light": "traffic light",
}

# DETECTOR class name (what the model emits) -> competition class.
MODEL_MAP = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "traffic_light": "traffic light",
    "traffic light": "traffic light",
}

# COCO category ids are 1-based and dense; id 0 is reserved by pycocotools.
CATEGORY_IDS = {name: index + 1 for index, name in enumerate(CLASSES)}

# COCO area ranges in original-pixel area.
SMALL_AREA = 32 * 32
MEDIUM_AREA = 96 * 96
AREA_RANGES = [[0.0, 1e10], [0.0, SMALL_AREA], [SMALL_AREA, MEDIUM_AREA], [MEDIUM_AREA, 1e10]]
AREA_INDEX = {"all": 0, "small": 1, "medium": 2, "large": 3}

SKIPPED_BDD_CLASSES = (
    "rider",
    "traffic sign",
    "train",
    "trailer",
    "other vehicle",
    "other person",
)


def box_area(box: list[float]) -> float:
    """Area of an ``[x1, y1, x2, y2]`` box, never negative."""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def size_bucket(area: float) -> str:
    """COCO object-size bucket for an area in original pixels."""
    if area < SMALL_AREA:
        return "small"
    return "medium" if area < MEDIUM_AREA else "large"


def read_bdd_rows(csv_path: Path) -> list[dict]:
    """Read the BDD100K validation CSV rows verbatim."""
    with Path(csv_path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_ground_truth(
    rows: list[dict],
    image_names: list[str],
    width: int,
    height: int,
) -> tuple[dict, dict]:
    """Build a COCO ground-truth dict for exactly ``image_names``.

    Returns ``(coco_gt, stats)``. Image ids are the 1-based index into
    ``image_names`` so the same 1000 frozen frames always produce the same ids.
    Boxes are emitted as COCO ``[x, y, w, h]`` in original-pixel coordinates.
    """
    image_ids = {Path(name).name: index + 1 for index, name in enumerate(image_names)}
    images = [{"id": image_id, "file_name": name, "width": width, "height": height} for name, image_id in image_ids.items()]
    annotations: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    dropped = 0
    outside = 0
    for row in rows:
        name = Path(row["image_path"]).name
        image_id = image_ids.get(name)
        if image_id is None:
            outside += 1
            continue
        target = BDD_MAP.get(row["class_name"])
        if target is None:
            dropped += 1
            continue
        x1, y1 = float(row["xmin"]), float(row["ymin"])
        x2, y2 = float(row["xmax"]), float(row["ymax"])
        box = [x1, y1, x2 - x1, y2 - y1]
        area = box_area([x1, y1, x2, y2])
        if box[2] <= 0 or box[3] <= 0:
            dropped += 1
            continue
        annotations.append(
            {
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": CATEGORY_IDS[target],
                "bbox": box,
                "area": area,
                "iscrowd": 0,
                "bucket": size_bucket(area),
            }
        )
        counts[target] += 1
    categories = [{"id": CATEGORY_IDS[name], "name": name} for name in CLASSES]
    coco_gt = {"images": images, "categories": categories, "annotations": annotations}
    stats = {
        "images": len(images),
        "annotations": len(annotations),
        "per_class_instances": {name: counts.get(name, 0) for name in CLASSES},
        "dropped_rows": dropped,
        "rows_outside_holdout": outside,
        "dropped_classes": list(SKIPPED_BDD_CLASSES),
    }
    return coco_gt, stats


def strip_internal_fields(coco_gt: dict) -> dict:
    """Drop the helper ``bucket`` key so the dict is valid COCO JSON."""
    return {
        "images": coco_gt["images"],
        "categories": coco_gt["categories"],
        "annotations": [
            {key: value for key, value in annotation.items() if key != "bucket"}
            for annotation in coco_gt["annotations"]
        ],
    }


def predictions_to_coco(
    per_image: list[tuple[str, list[tuple[int, float, float, float, float, float]]]],
    image_names: list[str],
    model_names: dict[int, str],
    score_floor: float,
) -> tuple[list[dict], dict]:
    """Convert ``(image_name, [(class_id, x1, y1, x2, y2, score)])`` to COCO detections.

    Predictions whose model class is outside the competition taxonomy are dropped
    and counted, so an unmapped class can never silently vanish.
    """
    image_ids = {Path(name).name: index + 1 for index, name in enumerate(image_names)}
    detections: list[dict] = []
    dropped_unmapped = 0
    below_floor = 0
    per_class_predictions: dict[str, int] = defaultdict(int)
    for name, boxes in per_image:
        image_id = image_ids.get(Path(name).name)
        if image_id is None:
            continue
        for class_id, x1, y1, x2, y2, score in boxes:
            if score < score_floor:
                below_floor += 1
                continue
            target = MODEL_MAP.get(model_names.get(int(class_id), ""))
            if target is None:
                dropped_unmapped += 1
                continue
            if x2 <= x1 or y2 <= y1:
                dropped_unmapped += 1
                continue
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": CATEGORY_IDS[target],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
            per_class_predictions[target] += 1
    stats = {
        "detections": len(detections),
        "dropped_unmapped_or_degenerate": dropped_unmapped,
        "below_score_floor": below_floor,
        "per_class_detections": {name: per_class_predictions.get(name, 0) for name in CLASSES},
    }
    return detections, stats


def operating_point(
    detections: list[dict],
    coco_gt: dict,
    conf: float,
    iou_threshold: float = 0.5,
) -> dict:
    """Per-class precision/recall/F1 at one confidence threshold.

    Uses greedy one-to-one matching per class at IoU >= ``iou_threshold``, which is
    the protocol the historical reports used, so these numbers are directly
    comparable with them. They are NOT mAP and must never be quoted as mAP.
    """
    gt_per_image: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    gt_buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    for annotation in coco_gt["annotations"]:
        key = (annotation["image_id"], annotation["category_id"])
        x, y, w, h = annotation["bbox"]
        gt_per_image[key].append([x, y, x + w, y + h])
        gt_buckets[key].append(annotation["bucket"])
    pred_per_image: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    pred_scores: dict[tuple[int, int], list[float]] = defaultdict(list)
    for detection in detections:
        if detection["score"] < conf:
            continue
        key = (detection["image_id"], detection["category_id"])
        x, y, w, h = detection["bbox"]
        pred_per_image[key].append([x, y, x + w, y + h])
        pred_scores[key].append(detection["score"])

    totals = {name: {"tp": 0, "fp": 0, "fn": 0} for name in CLASSES}
    buckets = {(name, bucket): {"gt": 0, "matched": 0} for name in CLASSES for bucket in ("small", "medium", "large")}
    keys = set(gt_per_image) | set(pred_per_image)
    for key in keys:
        name = None
        for candidate in CLASSES:
            if CATEGORY_IDS[candidate] == key[1]:
                name = candidate
                break
        if name is None:
            continue
        truth = gt_per_image.get(key, [])
        predictions = pred_per_image.get(key, [])
        order = sorted(range(len(predictions)), key=lambda index: -pred_scores[key][index])
        matched_pred: set[int] = set()
        matched_gt: set[int] = set()
        for pred_index in order:
            best_iou, best_gt = 0.0, -1
            for gt_index, gt_box in enumerate(truth):
                if gt_index in matched_gt:
                    continue
                score = iou(predictions[pred_index], gt_box)
                if score > best_iou:
                    best_iou, best_gt = score, gt_index
            if best_gt >= 0 and best_iou >= iou_threshold:
                matched_pred.add(pred_index)
                matched_gt.add(best_gt)
        totals[name]["tp"] += len(matched_pred)
        totals[name]["fp"] += len(predictions) - len(matched_pred)
        totals[name]["fn"] += len(truth) - len(matched_gt)
        for gt_index, bucket in enumerate(gt_buckets.get(key, [])):
            buckets[(name, bucket)]["gt"] += 1
            buckets[(name, bucket)]["matched"] += int(gt_index in matched_gt)

    for name, value in totals.items():
        precision = value["tp"] / (value["tp"] + value["fp"]) if value["tp"] + value["fp"] else 0.0
        recall = value["tp"] / (value["tp"] + value["fn"]) if value["tp"] + value["fn"] else 0.0
        value["precision"] = precision
        value["recall"] = recall
        value["f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    macro_recall = sum(value["recall"] for value in totals.values()) / len(CLASSES)
    macro_f1 = sum(value["f1"] for value in totals.values()) / len(CLASSES)
    bucket_recall = {}
    for bucket in ("small", "medium", "large"):
        gt_total = sum(buckets[(name, bucket)]["gt"] for name in CLASSES)
        matched = sum(buckets[(name, bucket)]["matched"] for name in CLASSES)
        bucket_recall[bucket] = {"gt": gt_total, "matched": matched, "recall": matched / gt_total if gt_total else 0.0}
    return {
        "conf": conf,
        "iou_match": iou_threshold,
        "classes": totals,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "size_buckets": bucket_recall,
    }


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = box_area(a)
    area_b = box_area(b)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def average_precision(precision: "object", iou_index: int, area_index: int, category_index: int | None = None) -> float:
    """Mean precision over the 101 recall thresholds, ignoring unset entries.

    ``precision`` is the COCOeval ``eval['precision']`` array with shape
    ``[T, R, K, A, M]``. ``-1`` marks recall levels with no detections and is
    excluded, which is exactly what ``COCOeval.accumulate`` relies on.
    """
    import numpy as np

    values = precision[iou_index, :, :, area_index, -1] if category_index is None else precision[iou_index, :, category_index, area_index, -1]
    values = np.asarray(values, dtype=np.float64)
    values = values[values > -1]
    return float(values.mean()) if values.size else 0.0


def average_precision_over_iou(
    precision: "object",
    iou_indices: range,
    area_index: int,
    category_index: int | None = None,
) -> float:
    """Mean of :func:`average_precision` across several IoU thresholds.

    ``range(10)`` reproduces COCO's ``mAP@0.5:0.95``; ``range(1)`` is AP50.
    """
    indices = list(iou_indices)
    if not indices:
        raise ValueError("iou_indices must not be empty")
    return sum(average_precision(precision, index, area_index, category_index) for index in indices) / len(indices)


def summarize(statistics: list[float], per_class: dict[str, dict[str, float]], extra: dict) -> dict:
    """Assemble the metric block from ``COCOeval.stats`` plus per-class APs.

    ``stats`` order is fixed by pycocotools:
    ``[AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]``.
    """
    return {
        "mAP50_95": statistics[0],
        "mAP50": statistics[1],
        "mAP75": statistics[2],
        "mAP_small": statistics[3],
        "mAP_medium": statistics[4],
        "mAP_large": statistics[5],
        "AR_1": statistics[6],
        "AR_10": statistics[7],
        "AR_100": statistics[8],
        "per_class": per_class,
        **extra,
    }
