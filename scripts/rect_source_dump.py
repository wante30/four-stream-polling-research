"""Dump detections from a PC backend on a fixed rectangular canvas.

Two backends, same output contract, so the results can be diffed directly against
each other and against the board's RKNN output:

``pytorch``  the exported checkpoint through Ultralytics, which is the reference;
``onnx``     the static split-output ONNX through onnxruntime, which isolates
             "the export broke it" from "the quantization broke it".

Both write original-image ``xyxy`` boxes. The two backends use different, and
easily confused, input conventions, so each is asserted rather than assumed:

* PyTorch/ONNX take float32 **RGB, NCHW, 0..1**. Feeding 0..255 to this graph
  produces confident nonsense (class 67 at confidence 1.0), which is why it is
  checked here and recorded in the output.
* The board's RKNN graph takes uint8 **RGB, NHWC, 0..255** with
  ``mean=0, std=255`` baked in, which is the same normalisation expressed
  differently.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # allow `python tools/rect_source_dump.py` from the repo root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from competition_ai.input_shape import validate_model_shape
from profile_rknn_single_stream import letterbox_bgr

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(directory: Path, limit: int) -> list[Path]:
    """Deterministic image list from a directory."""
    paths = sorted(path for path in Path(directory).rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    return paths[:limit] if limit > 0 else paths


def restore_boxes(rows: list[list[float]], scale: float, pad_x: float, pad_y: float) -> list[list[float]]:
    """Convert model-space xyxy rows back to original-image coordinates."""
    if scale <= 0:
        raise ValueError("letterbox scale must be positive")
    out = []
    for row in rows:
        x1, y1, x2, y2 = (float(value) for value in row[:4])
        out.append([(x1 - pad_x) / scale, (y1 - pad_y) / scale, (x2 - pad_x) / scale, (y2 - pad_y) / scale])
    return out


def infer_pytorch(model_path: Path, image_path: Path, width: int, height: int, device: str, conf: float) -> list[tuple[int, float, list[float]]]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    result = model.predict(
        source=str(image_path), imgsz=(height, width), conf=conf, device=device, rect=False, verbose=False
    )[0]
    rows = []
    for box, cls, score in zip(
        result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()
    ):
        rows.append((int(cls), float(score), [float(value) for value in box]))
    return rows


def _decode_split(outputs, conf: float, iou: float):
    from deployment.rknn.decode_yolo_output import decode_outputs

    return decode_outputs(outputs, conf, iou)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("pytorch", "onnx"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        width, height = validate_model_shape((args.width, args.height))
    except ValueError as exc:
        parser.error(str(exc))

    import cv2

    images = list_images(args.images, args.limit)
    if not images:
        raise SystemExit(f"no images under {args.images}")

    session = None
    if args.backend == "onnx":
        import onnxruntime as ort

        session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

    per_image = []
    started = time.perf_counter()
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        if args.backend == "pytorch":
            rows = infer_pytorch(args.model, path, width, height, args.device, args.conf)
        else:
            canvas, scale, pad_x, pad_y = letterbox_bgr(frame, (width, height))
            tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1))[None, ...].astype(np.float32) / 255.0
            outputs = session.run(None, {input_name: tensor})
            decoded = _decode_split(outputs, args.conf, args.iou)
            boxes = restore_boxes([item["bbox"] for item in decoded], scale, pad_x, pad_y)
            rows = [(int(item["class_id"]), float(item["confidence"]), box) for item, box in zip(decoded, boxes)]
        per_image.append(
            {
                "image": path.name,
                "detections": [[cls, score, *box] for cls, score, box in rows],
            }
        )
    elapsed = time.perf_counter() - started

    payload = {
        "backend": args.backend,
        "model": str(args.model),
        "canvas": f"{width}x{height}",
        "canvas_pixels": width * height,
        "conf": args.conf,
        "iou_nms": args.iou,
        "images_dir": str(args.images),
        "image_count": len(per_image),
        "input_convention": (
            "float32 RGB NCHW 0..1" if args.backend in ("pytorch", "onnx") else "uint8 RGB NHWC 0..255"
        ),
        "wall_ms_per_image": elapsed * 1000 / len(per_image) if per_image else None,
        "per_image": per_image,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload), encoding="utf-8")
    total = sum(len(entry["detections"]) for entry in per_image)
    print(
        f"{args.backend} @ {width}x{height}: {len(per_image)} images, {total} detections, "
        f"{payload['wall_ms_per_image']:.2f} ms/image -> {args.out}"
    )


if __name__ == "__main__":
    main()
