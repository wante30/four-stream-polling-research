"""Canonical input-shape contract for detection models.

Every place in this repository that turns a configured model input into an actual
tensor goes through here, so "the input is square" can never be an implicit
assumption again. A square configuration stays valid and keeps its old behaviour -
``320`` means 320x320 - while a rectangular one has to be written explicitly.

    normalize_shape(320)              -> (320, 320)
    normalize_shape((512, 288))       -> (512, 288)   # (width, height)
    validate_model_shape("512x288")   -> (512, 288)
    validate_model_shape("320")       -> (320, 320)

``normalize_shape`` is pure: it only checks that the value is a positive int or a
two-item ``(width, height)``. ``validate_model_shape`` adds the graph policy that
every YOLO stride needs: both sides at least 32 and divisible by 32. Keeping the
two apart is what lets unit-scale tensors (a 1x1 grid, say) still be decoded by
the raw-head decoder while a real model input is held to the stride rule.
"""
from __future__ import annotations

MIN_SIDE = 32
STRIDE = 32

ShapeLike = "int | str | tuple[int, int] | list[int]"


def normalize_shape(size) -> tuple[int, int]:
    """Return ``(width, height)`` for an int or a two-item sequence."""
    if isinstance(size, bool):
        raise ValueError(f"input shape must be int or (width, height), got {size!r}")
    if isinstance(size, int):
        width = height = size
    else:
        try:
            values = list(size)
        except TypeError as exc:
            raise ValueError(f"input shape must be int or (width, height), got {size!r}") from exc
        if len(values) != 2:
            raise ValueError(f"input shape must be int or (width, height), got {size!r}")
        try:
            width, height = int(values[0]), int(values[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"input shape must contain integers, got {size!r}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"input shape must be positive, got {(width, height)}")
    return width, height


def parse_shape_text(text: str) -> int | tuple[int, int]:
    """Parse ``"320"``, ``"512x288"`` (also ``X``/``×``/``*``/``,``) into a shape value."""
    clean = text.strip().lower().replace("×", "x").replace("*", "x").replace(",", "x")
    for separator in ("x",):
        if separator in clean:
            parts = [part.strip() for part in clean.split(separator)]
            if len(parts) != 2 or not all(part.lstrip("+-").isdigit() for part in parts):
                break
            return (int(parts[0]), int(parts[1]))
    if clean.lstrip("+-").isdigit():
        return int(clean)
    raise ValueError(f"cannot parse model shape {text!r}; expected e.g. 512x288 or 320")


def validate_model_shape(shape) -> tuple[int, int]:
    """Return ``(width, height)`` after enforcing the >=32 / multiple-of-32 rule."""
    resolved = parse_shape_text(shape) if isinstance(shape, str) else shape
    width, height = normalize_shape(resolved)
    if width < MIN_SIDE or height < MIN_SIDE or width % STRIDE or height % STRIDE:
        raise ValueError(
            f"model input must be >={MIN_SIDE} and divisible by {STRIDE}, got {(width, height)}"
        )
    return width, height


def to_ultralytics_imgsz(shape) -> int | tuple[int, int]:
    """Convert to the ``imgsz`` Ultralytics expects: an int, or ``(height, width)``."""
    width, height = validate_model_shape(shape)
    return height if width == height else (height, width)


def format_shape(shape) -> str:
    """Render ``(width, height)`` as ``WxH``; a square value stays a single number."""
    width, height = validate_model_shape(shape)
    return str(width) if width == height else f"{width}x{height}"


def format_tensor_shape(tensor_shape) -> str:
    """Render an observed tensor shape (NCHW or NHWC) as ``WxH (Cch)``."""
    values = [int(value) for value in tensor_shape]
    if len(values) == 4:
        if values[1] in (1, 3, 4) and values[-1] not in (1, 3, 4):
            return f"{values[3]}x{values[2]} ({values[1]}ch)"
        return f"{values[2]}x{values[1]} ({values[3]}ch)"
    if len(values) == 3:
        return f"{values[2]}x{values[1]} ({values[0]}ch)"
    return "x".join(str(value) for value in values)
