# Provenance and exclusions

This repository was assembled from the local workspace on 2026-10-12.

Primary source project:

`C:\Users\JingyuWang\adas-edge`

Board result source directory:

`D:\Datasets\adas_edge_training\phase10`

MOT17/RKNN result source directory:

`C:\Users\JingyuWang\adas-edge\benchmarks\mot17_e2_board`

The copied source reports retain their original experiment dates and hardware
descriptions. The compact JSON/CSV files under `data/` are copies of local
outputs, not recomputed values.

The following remain outside GitHub because they are large, sensitive, or
require separate data licensing: road videos, MOT17 frames, BDD holdout images
and labels, model weights, ONNX/RKNN binaries, NPZ output dumps, telemetry
streams, and board runtime wheels/libraries. Their absence is intentional.

The four-stream road videos are unlabelled. Their results support service
capacity, queue age, drop behavior, and stability only. The MOT17 files use a
fixed oracle/GT trace and support polling utility analysis only. These evidence
labels must remain attached when the results are reused in a manuscript.
