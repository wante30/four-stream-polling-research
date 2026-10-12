"""Compare detection dumps from different backends, and convert board outputs to one.

Phase 7 asks whether the ONNX export and the INT8 quantization damaged the model.
"Loads and runs" is not an answer; the comparison has to be on the same road frames
with the same confidence, class map and coordinate system.

Sources are reduced to one dump schema first (:func:`board_dir_to_dump` turns the
board's raw ``.npz`` outputs into it), then matched per image and class with greedy
one-to-one IoU matching. Reported per candidate:

* detection count ratio against the reference;
* matched / missed / spurious boxes;
* mean IoU and mean confidence delta of matched boxes;
* a ``gross_mismatch`` verdict when a matched fraction collapses, which is the
  signal to stop and fix the export or the quantization instead of benchmarking.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):  # allow `python tools/compare_detection_dumps.py` from the repo root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coco_eval_helpers import CLASSES, MODEL_MAP

# Guard rail: below this matched fraction the pipeline is considered broken rather
# than merely degraded, and benchmarking must not proceed.
GROSS_MISMATCH_MATCHED_FRACTION = 0.80


def _iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def load_dump(path: Path) -> dict:
    """Read a detection dump produced by any of the backends."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "per_image" not in payload:
        raise ValueError(f"{path} is not a detection dump")
    return payload


def board_dir_to_dump(
    board_dir: Path,
    manifest_path: Path,
    conf: float,
    iou_threshold: float,
    scores_logits: bool = False,
) -> dict:
    """Decode the board's raw RKNN outputs into the common dump schema."""
    import numpy as np

    from decode_yolo_output import decode_outputs

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    per_image = []
    for entry in manifest["entries"]:
        archive = Path(board_dir) / f"{entry['stem']}_out.npz"
        if not archive.is_file():
            continue
        with np.load(archive) as data:
            outputs = [data[key] for key in sorted(data.files)]
        decoded = decode_outputs(outputs, conf, iou_threshold, score_logits=scores_logits)
        scale, pad_x, pad_y = float(entry["scale"]), float(entry["pad_x"]), float(entry["pad_y"])
        detections = []
        for item in decoded:
            x1, y1, x2, y2 = item["bbox"]
            detections.append(
                [
                    int(item["class_id"]),
                    float(item["confidence"]),
                    (x1 - pad_x) / scale,
                    (y1 - pad_y) / scale,
                    (x2 - pad_x) / scale,
                    (y2 - pad_y) / scale,
                ]
            )
        per_image.append({"image": entry.get("image", entry["stem"]), "detections": detections})
    return {
        "backend": "rknn-int8",
        "canvas": manifest["canvas"],
        "conf": conf,
        "iou_nms": iou_threshold,
        "image_count": len(per_image),
        "input_convention": "uint8 RGB NHWC 0..255",
        "per_image": per_image,
    }


def class_name(class_id: int) -> str:
    """Map a model class id to the competition class, or ``class_<id>`` when unmapped."""
    return MODEL_MAP.get(str(class_id), MODEL_MAP.get(f"class_{class_id}", f"class_{class_id}"))


def match_dumps(reference: dict, candidate: dict, iou_threshold: float = 0.5) -> dict:
    """Greedy one-to-one matching of candidate detections against the reference."""
    ref = {entry["image"]: entry["detections"] for entry in reference["per_image"]}
    cand = {entry["image"]: entry["detections"] for entry in candidate["per_image"]}
    shared = sorted(set(ref) & set(cand))
    matched = missed = spurious = 0
    ious: list[float] = []
    confidence_deltas: list[float] = []
    reference_classes: Counter = Counter()
    candidate_classes: Counter = Counter()
    for name in shared:
        ref_rows = ref[name]
        cand_rows = cand[name]
        for row in ref_rows:
            reference_classes[str(int(row[0]))] += 1
        for row in cand_rows:
            candidate_classes[str(int(row[0]))] += 1
        used = set()
        for ref_row in ref_rows:
            best_iou, best_index = 0.0, -1
            for index, cand_row in enumerate(cand_rows):
                if index in used or int(cand_row[0]) != int(ref_row[0]):
                    continue
                score = _iou(ref_row[2:6], cand_row[2:6])
                if score > best_iou:
                    best_iou, best_index = score, index
            if best_index >= 0 and best_iou >= iou_threshold:
                used.add(best_index)
                matched += 1
                ious.append(best_iou)
                confidence_deltas.append(abs(float(cand_rows[best_index][1]) - float(ref_row[1])))
            else:
                missed += 1
        spurious += len(cand_rows) - len(used)
    total_reference = matched + missed
    return {
        "images_compared": len(shared),
        "reference_detections": total_reference,
        "candidate_detections": sum(len(rows) for rows in cand.values()),
        "matched": matched,
        "missed": missed,
        "spurious": spurious,
        "matched_fraction": matched / total_reference if total_reference else 0.0,
        "mean_iou_matched": sum(ious) / len(ious) if ious else 0.0,
        "mean_abs_conf_delta": sum(confidence_deltas) / len(confidence_deltas) if confidence_deltas else 0.0,
        "max_abs_conf_delta": max(confidence_deltas) if confidence_deltas else 0.0,
        "reference_class_histogram": {class_name(int(key)): value for key, value in reference_classes.items()},
        "candidate_class_histogram": {class_name(int(key)): value for key, value in candidate_classes.items()},
        "gross_mismatch": (matched / total_reference if total_reference else 0.0) < GROSS_MISMATCH_MATCHED_FRACTION,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# INT8 / export correctness: candidate versus PyTorch FP32 reference",
        "",
        f"- reference: `{report['reference']['backend']}` on {report['reference']['image_count']} images "
        f"at {report['reference']['canvas']}",
        f"- matching: same class, greedy one-to-one, IoU >= {report['iou_threshold']}",
        f"- gross-mismatch threshold: matched fraction < {report['gross_mismatch_threshold']}",
        "",
        "| candidate | images | ref det | cand det | matched | missed | spurious | matched frac | mean IoU | mean |dconf| | verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for label, entry in report["candidates"].items():
        verdict = "GROSS MISMATCH" if entry["gross_mismatch"] else "consistent"
        lines.append(
            f"| {label} | {entry['images_compared']} | {entry['reference_detections']} | "
            f"{entry['candidate_detections']} | {entry['matched']} | {entry['missed']} | {entry['spurious']} | "
            f"{entry['matched_fraction']:.4f} | {entry['mean_iou_matched']:.4f} | "
            f"{entry['mean_abs_conf_delta']:.4f} | {verdict} |"
        )
    for label, entry in report["candidates"].items():
        lines += ["", f"## {label} class histogram", "", "| class | reference | candidate |", "| --- | ---: | ---: |"]
        for name in CLASSES:
            lines.append(
                f"| {name} | {entry['reference_class_histogram'].get(name, 0)} | "
                f"{entry['candidate_class_histogram'].get(name, 0)} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], metavar="LABEL=DUMP.json")
    parser.add_argument(
        "--board",
        action="append",
        default=[],
        metavar="LABEL=BOARD_DIR|MANIFEST.json",
        help="board outputs; '|' separates the two paths because Windows drive letters contain ':'",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--scores-logits",
        action="store_true",
        help="board model exposes pre-sigmoid logits (split_yolo_output.py --scores-mode logits)",
    )
    parser.add_argument("--iou-nms", type=float, default=0.45)
    parser.add_argument("--iou-match", type=float, default=0.5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    reference = load_dump(args.reference)
    candidates: dict[str, dict] = {}
    for item in args.candidate:
        label, _, path = item.partition("=")
        candidates[label] = match_dumps(reference, load_dump(Path(path)), args.iou_match)
    for item in args.board:
        label, _, spec = item.partition("=")
        board_dir, _, manifest = spec.partition("|")
        if not manifest:
            parser.error(f"--board expects LABEL=BOARD_DIR|MANIFEST.json, got {item!r}")
        dump = board_dir_to_dump(Path(board_dir), Path(manifest), args.conf, args.iou_nms, args.scores_logits)
        candidates[label] = match_dumps(reference, dump, args.iou_match)

    report = {
        "reference": {key: reference[key] for key in ("backend", "canvas", "conf", "image_count", "model") if key in reference},
        "iou_threshold": args.iou_match,
        "gross_mismatch_threshold": GROSS_MISMATCH_MATCHED_FRACTION,
        "candidates": candidates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    table = render_markdown(report)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nwrote {args.out}")
    if any(entry["gross_mismatch"] for entry in candidates.values()):
        raise SystemExit("gross mismatch detected: fix the export or the quantization before benchmarking")


if __name__ == "__main__":
    main()
