# Four-stream video polling research

This repository is the continuation workspace for the local four-camera video
polling experiments on RK3566. It collects the scheduler implementation,
replay harness, compact result files, and the reports that define the
measurement protocol.

The main question is:

> How should one RK3566 NPU serve four camera streams so that aggregate
> throughput, per-stream freshness, and detection utility remain usable at the
> same time?

The repository is an experiment archive and a paper starting point. It does
not contain camera videos, model weights, raw frame dumps, RKNN binaries,
board runtime libraries, or the frozen BDD holdout.

## What has already been measured

The strongest local result is the 20-minute `WRR-FAIR-D1` run:

| item | measured value |
| --- | ---: |
| streams | FRONT 512×288 @ 10 FPS; LEFT/RIGHT/BACK 320×192 @ 6 FPS |
| scheduler | weighted round-robin, weights 10:6:6:6 |
| queue | depth 1, drop-oldest |
| duration | 1,196.4 s |
| aggregate service | 27.30 / 28 FPS (97.5%) |
| FRONT service | 9.85 FPS; 1.8% dropped; end-to-end p95 153.7 ms |
| side/back service | 5.70–5.93 FPS; 1.5–5.3% dropped; end-to-end p95 175.7–194.4 ms |
| failures | no stream errors, crash, restart, or id/frame corruption |

The controlled scheduler comparison is also preserved: on identical stream
definitions, weighted round-robin reached 27.54 FPS while the rate/deadline
policy reached 23.86 FPS. Matching weights to requested FPS raised the 300 s
run to 27.91 FPS and removed the under-served BACK stream seen with 5:2:2:1.

These numbers are board measurements. The four road videos are unlabelled, so
they establish service, freshness, and stability; they do not establish object
detection accuracy.

## Layout

- `src/multistream_framework/`: schedulers and heterogeneous replay runner.
- `scripts/`: copied launchers and analysis helpers from the local project.
- `data/phase10/`: compact JSON summaries for the RK3566 four-stream runs.
- `data/mot17/`: compact CSV summaries for the GT-backed polling experiments.
- `docs/source_reports/`: source reports copied from the local ADAS project.
- `docs/experiment_catalog.md`: the paper-facing inventory and evidence labels.
- `docs/metrics_protocol.md`: definitions for throughput, drop, freshness, FDD,
  recall, and fairness.
- `docs/paper_roadmap.md`: a proposed paper structure and the next experiments.

## Reproducing the software layer

The board commands need the external RKNN runtime and the local video/model
paths. The scheduler tests run without those assets:

```powershell
python -m pytest -q
```

For a board replay, copy a generated JSON config beside the runner, replace
the video and RKNN paths, then run:

```bash
python3 src/multistream_framework/hetero_replay.py \
  --config configs/CONFIG-B-WRR-FAIR.json \
  --output-dir results/CONFIG-B-WRR-FAIR \
  --duration 300
```

The original absolute paths, dates, hardware/runtime versions, and excluded
raw artifacts are recorded in `docs/provenance.md`.
