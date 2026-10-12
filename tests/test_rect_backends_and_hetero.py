"""Behavior checks for the rectangular backends and the heterogeneous stream config.

Covers the Phase 19 checklist items that were not yet tested: rectangular
coordinate restoration, cross-backend routing by image key, heterogeneous stream
configuration, and per-stream scheduler construction.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).parents[1]


def _load(name: str, relative: str):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    for path in (REPO, REPO / "src", REPO / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def source_dump_module():
    return _load("rect_source_dump", "scripts/rect_source_dump.py")


@pytest.fixture(scope="module")
def compare_module():
    return _load("compare_dumps", "scripts/compare_detection_dumps.py")


@pytest.fixture(scope="module")
def hetero_module():
    return _load("hetero_replay", "src/multistream_framework/hetero_replay.py")


# ------------------------------------------------------- rectangular restoration


def test_restore_boxes_undoes_letterbox_for_a_rectangle(source_dump_module) -> None:
    # a 1280x720 frame letterboxed to 512x288 needs no padding: scale 0.4, pad 0
    rows = [[0.0, 0.0, 512.0, 288.0]]
    restored = source_dump_module.restore_boxes(rows, scale=0.4, pad_x=0.0, pad_y=0.0)
    assert restored == [[0.0, 0.0, 1280.0, 720.0]]


def test_restore_boxes_accounts_for_vertical_padding_on_a_320x192_canvas(source_dump_module) -> None:
    # 1280x720 -> 320x180 content inside a 320x192 canvas: scale 0.25, pad_y 6
    restored = source_dump_module.restore_boxes([[0.0, 6.0, 320.0, 186.0]], scale=0.25, pad_x=0.0, pad_y=6.0)
    assert restored == [[0.0, 0.0, 1280.0, 720.0]]


def test_restore_boxes_rejects_a_non_positive_scale(source_dump_module) -> None:
    with pytest.raises(ValueError):
        source_dump_module.restore_boxes([[1.0, 2.0, 3.0, 4.0]], scale=0.0, pad_x=0.0, pad_y=0.0)


def test_image_listing_is_deterministic_and_capped(tmp_path: Path, source_dump_module) -> None:
    for name in ("b.jpg", "a.jpg", "c.png", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")

    found = source_dump_module.list_images(tmp_path, 2)

    assert [path.name for path in found] == ["a.jpg", "b.jpg"]


# ------------------------------------------------------- cross-backend routing


def _board_bundle(tmp_path: Path) -> tuple[Path, Path]:
    """A minimal board output directory plus its canvas manifest."""
    board = tmp_path / "board"
    board.mkdir()
    # two canvases, each holding one identical car box in model space
    for stem in ("frame_000", "frame_001"):
        # the split decoder expects boxes [1,4,N] and scores [1,80,N]
        boxes = np.zeros((1, 4, 1), dtype=np.float32)
        boxes[0, :, 0] = [100.0, 100.0, 40.0, 20.0]
        scores = np.full((1, 80, 1), -10.0, dtype=np.float32)
        scores[0, 2, 0] = 10.0
        np.savez_compressed(board / f"{stem}_out.npz", out0=boxes, out1=scores)
    manifest = tmp_path / "canvas.json"
    manifest.write_text(
        json.dumps(
            {
                "canvas": "512x288",
                "layout": "NHWC",
                "channel_order": "RGB",
                "array_shape": [1, 288, 512, 3],
                "entries": [
                    {"stem": "frame_000", "image": "a.jpg", "scale": 0.4, "pad_x": 0, "pad_y": 0},
                    {"stem": "frame_001", "image": "b.jpg", "scale": 0.4, "pad_x": 0, "pad_y": 0},
                ],
            }
        ),
        encoding="utf-8",
    )
    return board, manifest


def test_board_outputs_are_routed_by_image_key_not_by_index(tmp_path: Path, compare_module) -> None:
    board, manifest = _board_bundle(tmp_path)

    dump = compare_module.board_dir_to_dump(board, manifest, conf=0.0, iou_threshold=0.7)

    assert [entry["image"] for entry in dump["per_image"]] == ["a.jpg", "b.jpg"]
    assert dump["canvas"] == "512x288"


def test_match_dumps_reports_spurious_missed_and_gross_mismatch(compare_module) -> None:
    reference = {
        "backend": "pytorch",
        "canvas": "512x288",
        "image_count": 1,
        "per_image": [{"image": "a.jpg", "detections": [[2, 0.9, 0.0, 0.0, 100.0, 50.0]]}],
    }
    candidate = {
        "backend": "rknn-int8",
        "canvas": "512x288",
        "image_count": 1,
        # shifted far enough to miss at IoU >= 0.5, plus an extra box
        "per_image": [{"image": "a.jpg", "detections": [[2, 0.9, 400.0, 400.0, 500.0, 450.0]]}],
    }

    report = compare_module.match_dumps(reference, candidate, iou_threshold=0.5)

    assert report["matched"] == 0
    assert report["missed"] == 1
    assert report["spurious"] == 1
    assert report["gross_mismatch"] is True


def test_match_dumps_accepts_a_faithful_candidate(compare_module) -> None:
    detections = [[2, 0.9, 0.0, 0.0, 100.0, 50.0], [0, 0.8, 10.0, 10.0, 30.0, 60.0]]
    reference = {"backend": "pytorch", "canvas": "512x288", "image_count": 1, "per_image": [{"image": "a.jpg", "detections": detections}]}
    candidate = {"backend": "onnx", "canvas": "512x288", "image_count": 1, "per_image": [{"image": "a.jpg", "detections": detections}]}

    report = compare_module.match_dumps(reference, candidate, iou_threshold=0.5)

    assert report["matched"] == 2
    assert report["missed"] == 0
    assert report["spurious"] == 0
    assert report["matched_fraction"] == 1.0
    assert report["gross_mismatch"] is False
    assert report["mean_iou_matched"] == pytest.approx(1.0)


# ------------------------------------------------- heterogeneous stream config


def _config_payload(tmp_path: Path) -> dict:
    return {
        "label": "TEST",
        "duration_s": 5,
        "queue_depth": 3,
        "scheduler": {"kind": "rate", "priority": {"0": 3, "1": 2, "2": 2, "3": 1}},
        "streams": [
            {"stream_id": 0, "role": "FRONT", "video": "a.mp4", "model": "m512.rknn", "width": 512, "height": 288, "target_fps": 10},
            {"stream_id": 1, "role": "LEFT", "video": "b.mp4", "model": "m320.rknn", "width": 320, "height": 192, "target_fps": 6},
        ],
    }


def test_load_config_accepts_a_heterogeneous_stream_set(tmp_path: Path, hetero_module) -> None:
    path = tmp_path / "c.json"
    path.write_text(json.dumps(_config_payload(tmp_path)), encoding="utf-8")

    config = hetero_module.load_config(path)

    assert [stream["stream_id"] for stream in config["streams"]] == [0, 1]
    assert config["streams"][0]["width"] == 512 and config["streams"][0]["height"] == 288
    assert config["streams"][1]["target_fps"] == 6.0


def test_load_config_rejects_an_illegal_canvas(tmp_path: Path, hetero_module) -> None:
    payload = _config_payload(tmp_path)
    payload["streams"][0]["height"] = 360  # not a multiple of 32
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit):
        hetero_module.load_config(path)


def test_load_config_rejects_duplicate_stream_ids(tmp_path: Path, hetero_module) -> None:
    payload = _config_payload(tmp_path)
    payload["streams"][1]["stream_id"] = 0
    path = tmp_path / "dup.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit):
        hetero_module.load_config(path)


def test_load_config_rejects_a_non_positive_rate(tmp_path: Path, hetero_module) -> None:
    payload = _config_payload(tmp_path)
    payload["streams"][0]["target_fps"] = 0
    path = tmp_path / "zero.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit):
        hetero_module.load_config(path)


def test_make_scheduler_builds_the_configured_kind(tmp_path: Path, hetero_module) -> None:
    # hetero_replay imports ``scheduler`` as a top-level module, so the class is
    # identified by name and behaviour rather than by isinstance against a second
    # importlib copy of the same file.
    payload = _config_payload(tmp_path)
    payload["scheduler"] = {"kind": "weighted", "weights": {"0": 5, "1": 2}}
    config = hetero_module.load_config(_write(tmp_path, payload))

    scheduler = hetero_module.make_scheduler(config)

    assert type(scheduler).__name__ == "WeightedRoundRobinScheduler"
    assert scheduler.weights == {0: 5, 1: 2}
    # FRONT gets five slots for every two LEFT slots
    states = {0: {"queue_length": 100}, 1: {"queue_length": 100}}
    picks = [scheduler.select(states) for _ in range(7)]
    assert picks.count(0) == 5 and picks.count(1) == 2


def test_make_scheduler_rate_uses_per_stream_targets(tmp_path: Path, hetero_module) -> None:
    config = hetero_module.load_config(_write(tmp_path, _config_payload(tmp_path)))

    scheduler = hetero_module.make_scheduler(config)

    assert type(scheduler).__name__ == "RateDeadlineScheduler"
    assert scheduler.target_fps == {0: 10.0, 1: 6.0}
    assert scheduler.priority[0] == 3


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_shipped_configs_are_loadable_and_keep_front_heaviest() -> None:
    """The generated CONFIG-A..D and PROFILE configs must satisfy the priority rule."""
    generator = _load("make_hetero_configs", "scripts/make_hetero_configs.py")

    for name, payload in generator.PLANS.items():
        by_role = {stream["role"]: stream for stream in payload["streams"]}
        assert set(by_role) == {"FRONT", "LEFT", "RIGHT", "BACK"}, name
        # FRONT must never be the cheapest stream and must carry the highest rate
        assert by_role["FRONT"]["target_fps"] >= by_role["BACK"]["target_fps"], name
        front_pixels = by_role["FRONT"]["width"] * by_role["FRONT"]["height"]
        assert front_pixels >= by_role["LEFT"]["width"] * by_role["LEFT"]["height"], name
        assert front_pixels >= by_role["BACK"]["width"] * by_role["BACK"]["height"], name
