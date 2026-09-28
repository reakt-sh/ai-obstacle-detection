# Bundled Model

`reaktrailai-yolo26m-seg.pt` is the unified ReaktRailAi model used by the live
pipeline. It segments both the rail area and the trained object classes. It
therefore replaces the previous approach, which used a separate rail model and
a separate object detector.

File information:

- Architecture: Ultralytics YOLO26m Segmentation
- Task: joint rail and object segmentation
- Pipeline input size: 640 × 640 pixels
- SHA-256: `3f1e0c05d1da5c971a112200a84a7fa95052988df27c94d27afe29c4ee26285c`
- Ultralytics license stored in the checkpoint: `AGPL-3.0`

To try another compatible model, set `MODEL_PATH` in `.env` to the model's host
path. Compose also mounts a host file with a different name at the expected
container path, so `MODEL` does not need to be changed.
