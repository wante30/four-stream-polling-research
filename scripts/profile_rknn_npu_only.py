"""Measure RKNN inference service capacity with no decode/pre/post work."""
from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np


def read_npu_load() -> str:
    paths = ["/sys/kernel/debug/rknpu/load"]
    for path in paths:
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            pass
    try:
        result = subprocess.run(["sudo", "-n", "cat", paths[0]], capture_output=True, text=True, timeout=1)
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return ""


def sample_resources(rows: list[dict[str, object]], stop: threading.Event) -> None:
    import psutil

    while not stop.is_set():
        row: dict[str, object] = {"timestamp": time.time(), "npu_load": read_npu_load()}
        for index, value in enumerate(psutil.cpu_percent(percpu=True)):
            row[f"cpu_core_{index}_percent"] = value
        for path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
            try:
                value = float(Path(path).read_text(encoding="utf-8").strip())
                row[Path(path).parent.name] = value / 1000.0 if value > 1000 else value
            except (OSError, ValueError):
                pass
        rows.append(row)
        stop.wait(0.5)


def run(model: Path, resolution: int, iterations: int, output_dir: Path, resource_sampler: bool = True) -> None:
    from rknnlite.api import RKNNLite

    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep 100 prepared tensors in memory. The loop below does no work besides inference and timing.
    tensors = [np.zeros((1, resolution, resolution, 3), dtype=np.uint8) for _ in range(100)]
    rknn = RKNNLite(verbose=False)
    if rknn.load_rknn(str(model)) != 0:
        raise RuntimeError(f"load_rknn failed: {model}")
    if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
        raise RuntimeError("init_runtime failed")
    rows: list[dict[str, object]] = []
    resources: list[dict[str, object]] = []
    stop = threading.Event()
    sampler = threading.Thread(target=sample_resources, args=(resources, stop), daemon=True)
    if resource_sampler:
        sampler.start()
    started = time.perf_counter()
    try:
        for index in range(iterations):
            t0 = time.perf_counter()
            rknn.inference(inputs=[tensors[index % len(tensors)]])
            t1 = time.perf_counter()
            rows.append({"iteration": index, "resolution": resolution, "npu_infer_ms": (t1 - t0) * 1000.0})
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        if resource_sampler:
            sampler.join(timeout=2.0)
        rknn.release()
    values = [float(row["npu_infer_ms"]) for row in rows]
    summary = {
        "resolution": resolution,
        "iterations": len(values),
        "wall_seconds": elapsed,
        "mean_ms": statistics.mean(values) if values else 0.0,
        "median_ms": statistics.median(values) if values else 0.0,
        "p95_ms": float(np.percentile(values, 95)) if values else 0.0,
        "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "fps": len(values) / elapsed if elapsed else 0.0,
    }
    prefix = output_dir / f"npu_only_{resolution}"
    with (prefix.with_name(prefix.name + "_per_frame.csv")).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iteration", "resolution", "npu_infer_ms"])
        writer.writeheader(); writer.writerows(rows)
    resource_fields = list(resources[0]) if resources else ["timestamp", "npu_load"]
    with (prefix.with_name(prefix.name + "_resources.csv")).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=resource_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(resources)
    with (prefix.with_name(prefix.name + "_summary.csv")).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader(); writer.writerow(summary)
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--resolution", type=int, choices=(320, 416, 640), required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("npu_only_output"))
    parser.add_argument("--no-resource-sampler", action="store_true", help="run a pure inference loop without the resource sampler thread")
    args = parser.parse_args()
    run(args.model, args.resolution, args.iterations, args.output_dir, resource_sampler=not args.no_resource_sampler)


if __name__ == "__main__":
    main()
