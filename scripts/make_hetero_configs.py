"""Generate the heterogeneous four-stream replay configs (CONFIG A-D, Profiles 1-3)."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("D:/Datasets/adas_edge_training/phase10")
MODELS = "../rect_models/"
# Paths are resolved relative to the config file, which lives in configs/.
VIDEOS = "../videos/"
ROLES = [
    ("FRONT", f"{VIDEOS}ch0_front.mp4"),
    ("LEFT", f"{VIDEOS}ch1_left.mp4"),
    ("RIGHT", f"{VIDEOS}ch2_right.mp4"),
    ("BACK", f"{VIDEOS}ch3_back.mp4"),
]
RATE = {"kind": "rate", "priority": {"0": 3, "1": 2, "2": 2, "3": 1}}
Q320, Q416, Q512, Q640 = (320, 192), (416, 256), (512, 288), (640, 352)


def rknn(width: int, height: int) -> str:
    return f"{MODELS}yolo11n_rk3566_int8_split_{width}x{height}.rknn"


def config(label, canvases, fps, scheduler, duration=300, queue_depth=3):
    streams = []
    for index, (role, video) in enumerate(ROLES):
        width, height = canvases[index]
        streams.append(
            {
                "stream_id": index,
                "role": role,
                "video": video,
                "model": rknn(width, height),
                "width": width,
                "height": height,
                "target_fps": fps[index],
            }
        )
    return {
        "label": label,
        "duration_s": duration,
        "queue_depth": queue_depth,
        "scheduler": scheduler,
        "streams": streams,
    }


PLANS = {
    # A: every stream on the cheapest canvas - the conservative speed floor.
    "CONFIG-A": config("CONFIG-A", [Q320] * 4, [10, 6, 6, 6], RATE),
    # B: the recommended candidate - detail only where it matters.
    "CONFIG-B": config("CONFIG-B", [Q512, Q320, Q320, Q320], [10, 6, 6, 6], RATE),
    # C: quality candidate - the side cameras get a middle canvas too.
    "CONFIG-C": config("CONFIG-C", [Q512, Q416, Q416, Q320], [10, 6, 6, 6], RATE),
    # D: front at the high-quality diagnostic canvas.
    "CONFIG-D": config("CONFIG-D", [Q640, Q320, Q320, Q320], [10, 6, 6, 6], RATE),
    "PROFILE-1": config("PROFILE-1 (on CONFIG-B)", [Q512, Q320, Q320, Q320], [10, 6, 6, 6], RATE),
    "PROFILE-2": config("PROFILE-2 (on CONFIG-B)", [Q512, Q320, Q320, Q320], [12, 6, 6, 5], RATE),
    "PROFILE-3": config("PROFILE-3 (on CONFIG-B)", [Q512, Q320, Q320, Q320], [12, 8, 8, 6], RATE),
    # The recommended candidate, run for the longer acceptance duration.
    "CONFIG-B-10MIN": config("CONFIG-B 10 min", [Q512, Q320, Q320, Q320], [10, 6, 6, 6], RATE, duration=600),
    # Same allocation as B but on the weighted round-robin scheduler, so the two
    # schedulers can be compared on identical stream definitions.
    "CONFIG-B-WRR": config(
        "CONFIG-B weighted RR", [Q512, Q320, Q320, Q320], [10, 6, 6, 6],
        {"kind": "weighted", "weights": {"0": 5, "1": 2, "2": 2, "3": 1}},
    ),
    # The same scheduler with weights proportional to the target FPS instead of an
    # arbitrary priority ladder. A 5:2:2:1 ladder entitles BACK to only ~10% of
    # slots, which is less than its 6 FPS target needs, so it is the one stream
    # that still drops frames under weighted round-robin.
    "CONFIG-B-WRR-FAIR": config(
        "CONFIG-B weighted RR, weights matched to targets", [Q512, Q320, Q320, Q320], [10, 6, 6, 6],
        {"kind": "weighted", "weights": {"0": 10, "1": 6, "2": 6, "3": 6}},
    ),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, payload in PLANS.items():
        (OUT / f"{name}.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
        allocation = " ".join(
            f"{s['role']}={s['width']}x{s['height']}@{s['target_fps']}" for s in payload["streams"]
        )
        print(f"{name:<14} {payload['scheduler']['kind']:<10} {allocation}")


if __name__ == "__main__":
    main()
