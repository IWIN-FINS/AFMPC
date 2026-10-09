# Data interfaces

## Vision JSONL

`vision/src/depth_estimation/demo_video.py` can write:

- `--result-jsonl`: processed detections and 3D positions with timestamps;
- `--frame-jsonl`: one timestamp record for every captured stereo frame;
- `--record-raw`: unmodified side-by-side frames;
- `--record-monitor`: annotated monitoring video.

The controller tails the result JSONL and rejects stale or implausible samples.
Every downstream dataset should retain both capture and processing timestamps.

## Coordinate conventions

- Camera: `[X right, Y down, Z forward]`, metres.
- Vehicle body: `[forward, right, down]`, metres.
- Yaw: positive nose-right rotation about the body down axis, radians.
- Force: `[forward, right, down]`, newtons.
- Yaw moment: newton-metres.

## Firmware protocol

The current host and firmware use protocol version 5. The Python protocol
implementation and firmware structs must change together. Protocol tests in
both trees should be run before flashing or operating hardware.

