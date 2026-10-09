# Active parameter summary

This page is a human-readable index. Machine-readable values and source
hashes are under `parameters/`; runtime files remain authoritative.

## Shared experiment frame and limits

| Item | Value |
|---|---|
| Body frame | forward, right, down (FRD) |
| Camera frame | right, down, forward (OpenCV) |
| Calibrated target in body frame | `[0.857634, -0.055545, -0.120815] m` |
| Control period | `0.05 s` |
| Expected stereo update period | `0.10 s` |
| Translation channel limit | `±0.20` |
| Positive force envelope | `[4.730162, 4.997534, 7.063140] N` |
| Negative force envelope | `[5.050680, 4.783308, 6.867140] N` |
| Positive/negative yaw envelope | `+1.807854 / -2.041126 N·m` |
| Per-update force slew | `±[1.20, 0.80, 1.00] N` |

代码中保存的通信地址是示例值，不属于控制器实验参数。

## Fusion MPC

| Group | Value |
|---|---|
| Effective mass diagonal | `[24.82, 26.26, 26.257826] kg` |
| Linear damping diagonal | `[9.589, 14.964789, 11.290874] N·s/m` |
| Restoring force | `[0, 0, 0.80729] N` |
| Horizon | `15` at `0.10 s` model sample time |
| Position weights | `[1200, 350, 900]` |
| Velocity weights | `[150, 150, 200]` |
| Force weights | `[0.6, 0.5, 0.8]` |
| Delta-force weights | `[5.0, 0.5, 3.0]` |
| Terminal scale | `2.0` |
| Actuator delay/time constant | `0.08 / 0.15 s` |
| Fusion window/prediction horizon | `8 / 5` |
| Forgetting factor | `0.8` |
| Weight update rate | `0.1` |
| Initial model-1 weight | `[0, 0, 0]` |

The complete solver, Kalman, FOV, fusion, actuator, and camera-transform
settings are in `parameters/control/mpc_fusion.json`.

## Rotation-aware MPC

| Item | Value |
|---|---|
| Effective yaw inertia | `0.33453415 kg·m²` |
| Linear yaw damping | `0.32251723 N·m/(rad/s)` |
| Rotation on/off/emergency | `6.0° / 1.2° / 8.0°` |
| Trigger/settle frames | `3 / 5` |
| Outer PID `Kp, Ki, Kd` | `1.5, 0, 0.2` |
| Inner PID `Kp, Ki, Kd` | `0.6, 0.02, 0` |
| Yaw moment slew | `±0.5 N·m` per update |
| Thruster force-limit scale | `0.20` |

## PID

| Axis order | forward, right, down |
|---|---|
| `Kp` | `[28, 35, 42]` |
| `Ki` | `[0.10, 0.15, 0]` |
| `Kd` | `[0, 0, 0]` |
| Derivative filter time constant | `0.35 s` |
| Integral limits | `[2.0, 1.5, 1.2]` |
| Yaw `Kp, Ki, Kd` | `1.8, 0.12, 0.55` |

PID uses the same force and per-update slew envelope as MPC. The generated
snapshot includes the complete eight-thruster constraint matrix.

## SMC

| Item | Value |
|---|---|
| Target/approach/hard-min distance | `0.60 / 0.80 / 0.30 m` |
| Maximum approach force | `3.20 N` |
| Minimum retreat force | `0.75 N` |
| Startup/reacquire confirmations | `3 / 5` |
| Maximum depth NIS | `25` |
| Yaw authority | lower-controller local hold |

Per-axis sliding-mode gains, boundary layers, rate filters, and input limits
are in `parameters/control/smc.json`.

## Stereo vision

| Item | Value |
|---|---|
| Calibrated resolution | `640 × 480` |
| Rectified focal length | `715.130774 px` |
| Principal point | `(319.177116, 256.594992) px` |
| Stereo baseline | `0.059459063 m` |
| Detector confidence | `0.30` |
| Detector input size | `960 px` |
| Stereo image scale / iterations | `0.20 / 4` |
| Temporal depth filter | sequence confidence Kalman |

The full camera matrices, distortion, tracker, detection, temporal filter,
and output settings are in `parameters/vision/stereo_pipeline.yaml`.
