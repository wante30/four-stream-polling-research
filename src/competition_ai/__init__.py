"""Competition AI package.

Imports are resolved lazily (PEP 562). ``competition_ai.input_shape`` is a pure,
dependency-free module that board tools and calibration generators need, and it
must not drag in OpenCV, NumPy or RKNN Lite just because a sibling module wants
them. ``from competition_ai import CompetitionDetector`` keeps working exactly as
before; the heavy module is only imported at the moment the name is used.
"""
from __future__ import annotations

import importlib

__all__ = [
    "CompetitionDetector",
    "Detection",
    "FrameResult",
    "TrafficLightState",
    "classify_traffic_light",
    "classify_traffic_lights",
    "SceneInput",
    "LaneType",
    "NavigationCommand",
    "VehicleCommand",
    "CompetitionAdas",
    "CompetitionCommand",
]

_MODULE_BY_NAME = {
    "CompetitionDetector": "detector",
    "Detection": "types",
    "FrameResult": "types",
    "TrafficLightState": "traffic_light",
    "classify_traffic_light": "traffic_light",
    "classify_traffic_lights": "traffic_light",
    "SceneInput": "scene",
    "LaneType": "scene",
    "NavigationCommand": "scene",
    "VehicleCommand": "scene",
    "CompetitionAdas": "adas",
    "CompetitionCommand": "adas",
}


def __getattr__(name: str):
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache so later lookups skip the import
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
