# Experiment catalog

This catalog distinguishes what each local experiment can support in a paper.

## Group A: RK3566 four-stream service experiments

Source reports: `docs/source_reports/heterogeneous_multistream.md`,
`multistream_budget.md`, and `rectangular_rk3566_benchmark.md`.

| experiment | independent variable | result kept for paper | evidence type |
| --- | --- | --- | --- |
| CONFIG-A | all streams 320×192, rate scheduler | 25.72 aggregate FPS; queues saturated; FRONT 8.69 FPS | board, unlabelled road video |
| CONFIG-B | FRONT 512×288, sides 320×192, rate scheduler | 23.86 aggregate FPS; 11.2% FRONT drop and 18.1% side drop | board, unlabelled road video |
| CONFIG-C | FRONT 512×288, sides 416×256, rate scheduler | 22.91 aggregate FPS; quality pixels cost service capacity | board, unlabelled road video |
| CONFIG-D | FRONT 640×352, sides 320×192, rate scheduler | 22.42 aggregate FPS; highest front cost | board, unlabelled road video |
| CONFIG-B-WRR | same canvases and targets as B, weights 5:2:2:1 | 27.54 aggregate FPS; BACK remains under-served | board, unlabelled road video |
| CONFIG-B-WRR-FAIR | same canvases and targets, weights 10:6:6:6 | 27.91 aggregate FPS; 0.997 aggregate attainment | board, unlabelled road video |
| WRR-FAIR-D1 | same fair weights, queue depth 1 | 27.41 FPS in 300 s; p95 freshness falls sharply | board, unlabelled road video |
| LONG20 / WRR-FAIR-D1 | same as D1, 1,200 s telemetry run | 27.30 FPS; no thermal decay or scheduler drift | board, unlabelled road video |

### Interpretation

The scheduler is a first-order factor. Rate/deadline selection creates bursts
when several streams become overdue; weighted round-robin spreads service and
reduces queue age. Weights should track the requested FPS. Queue depth is a
freshness knob: depth 1 reduces end-to-end p95 at the cost of a small increase
in drops.

Rectangular input is a separate resource result. Single-stream INT8 means were
16.12 ms (320×192), 23.86 ms (416×256), 30.42 ms (512×288), and 47.66 ms
(640×352). Four-stream measurements must use the higher concurrent values,
because CPU-side inference calls are stretched by decode and postprocess
threads.

## Group B: GT-backed polling experiments

These runs use fixed MOT17/RKNN oracle traces and answer a different question:
how polling policy affects object-frame recall, FDD censoring, and fairness when
the service is overloaded. The compact files under `data/mot17/` are copied
from the local benchmark tree.

The useful paper-facing result is the operating-point transition: at 4–6 FPS
target load, round-robin keeps object-frame recall near 0.99 with Jain fairness
near 1.0; at 8 FPS the system crosses its capacity boundary and recall drops
to about 0.887 for the RR trace. This is a scheduler stress result, not a
claim about the four unlabelled road videos.

## Group C: INT8 quality checks

The 100-image per-class checks compare FP32 and RKNN INT8 at 320×192 and
512×288. They are compact sanity checks for quantization drift, not a final
accuracy benchmark. The small-object classes remain weak, and the files do not
replace the frozen BDD holdout evaluation.
