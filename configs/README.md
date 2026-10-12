# Configuration placeholders

Board configs are intentionally not copied with user-specific absolute paths.
Use `scripts/make_hetero_configs.py` in the original project or create a JSON
with the fields below, then replace `video` and `model` paths on the board.

```json
{
  "label": "CONFIG-B-WRR-FAIR",
  "duration_s": 300,
  "queue_depth": 3,
  "scheduler": {"kind": "weighted", "weights": {"0": 10, "1": 6, "2": 6, "3": 6}},
  "streams": [
    {"stream_id": 0, "role": "FRONT", "width": 512, "height": 288, "target_fps": 10, "video": "...", "model": "..."},
    {"stream_id": 1, "role": "LEFT", "width": 320, "height": 192, "target_fps": 6, "video": "...", "model": "..."},
    {"stream_id": 2, "role": "RIGHT", "width": 320, "height": 192, "target_fps": 6, "video": "...", "model": "..."},
    {"stream_id": 3, "role": "BACK", "width": 320, "height": 192, "target_fps": 6, "video": "...", "model": "..."}
  ]
}
```
