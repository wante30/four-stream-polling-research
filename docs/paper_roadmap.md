# Paper continuation roadmap

## Candidate contribution

The local evidence supports a systems paper about freshness-aware multi-stream
edge inference rather than a generic detector paper:

> A weighted polling and bounded drop-oldest pipeline for heterogeneous
> multi-camera inference on a resource-limited NPU.

The contribution should be framed around three controlled factors:

1. heterogeneous per-stream compute cost from rectangular input canvases;
2. scheduler choice (rate/deadline versus weighted round-robin);
3. queue depth as the throughput/freshness trade-off.

## Proposed evaluation matrix

| axis | minimum comparison |
| --- | --- |
| scheduler | rate, uniform RR, weighted RR, FPS-proportional WRR |
| queue | depth 1, 2, 3 with the same drop-oldest rule |
| canvas | all 320×192; FRONT 512×288; FRONT 640×352; mixed 416×256 |
| load | 4×6, 10+6+6+6, and overload above the measured capacity |
| evidence | board latency/throughput + GT-backed trace utility |
| duration | 300 s per cell; 1,200 s for the selected configuration |

## Missing evidence before submission

- repeat the decisive scheduler cells with the same video set and fixed warm-up;
- report confidence intervals across independent runs, not one wall-clock run;
- add a labelled multi-camera or trace-derived utility experiment for the road
  streams;
- publish exact config JSON, commit SHA, model hash, runtime version, and board
  identity for every table row;
- separate algorithmic scheduler claims from RKNN implementation claims;
- add a failure taxonomy for queue drops, late frames, decode stalls, and
  runtime errors.

## Suggested paper structure

1. Problem and hardware constraints.
2. Heterogeneous service model and freshness metrics.
3. Polling policies and bounded queue design.
4. Experimental protocol and reproducibility boundary.
5. RK3566 results: canvas cost, scheduler, queue depth, long-run stability.
6. GT-backed stress results and operating point.
7. Limitations and next experiments.
