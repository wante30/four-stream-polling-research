# Metrics protocol

All future runs must state the data source, scheduler, queue policy, input
canvas, requested FPS, duration, and whether labels exist.

## Service metrics

- `aggregate_actual_fps = sum(selected frames across streams) / wall time`.
- `attainment = aggregate_actual_fps / total_requested_fps`.
- `drop_rate = dropped arrivals / (selected + dropped arrivals)` per stream.
- `wait_ms = selected_ts - arrival_ts`.
- `e2e_ms = post_done_ts - arrival_ts`; this includes queue wait and excludes
  decode time that happened before `arrival_ts`.
- `service_interval_ms` is the difference between consecutive selected times
  for one stream.
- `queue_depth` is the maximum number of pending frames allowed before the
  drop-oldest policy removes the oldest pending frame.

## Scheduling metrics

- `object_frame_recall` is used only when a fixed GT/oracle universe exists.
- `per_object_miss_rate` is the miss rate over oracle object occurrences.
- `fdd_arrival_censor_rate` counts GT/object events whose first eligible service
  was outside the evaluation window.
- `Jain fairness = (sum x_i)^2 / (n * sum x_i^2)` for the selected per-stream
  service shares or the explicitly documented utility vector.

Do not compare a board throughput number with a PC wall-clock number. Do not
call an unlabelled road replay an accuracy evaluation. Do not mix the MOT17
oracle universe with the four-camera road-video universe.
