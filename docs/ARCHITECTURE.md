# Architecture

```text
stereo cameras
      |
      v
vision/demo_video.py -- timestamped 3D target JSONL
      |
      v
vision gate + camera-to-body transform
      |
      +--> dual-model translational MPC + optional yaw MPC
      +--> three-axis PID + yaw PID
      +--> full-vehicle SMC
      |
      v
force/yaw adapter + eight-thruster allocation
      |
      v
v5 command protocol over UDP/TCP/USART
      |
      v
STM32 firmware: validation, attitude hold, mixer, DShot output, telemetry
```

All controller coordinates use body FRD: forward, right, down. Vision emits
OpenCV camera coordinates: right, down, forward. The calibrated rigid transform
in the runtime configuration is the only supported conversion for hardware
experiments.

The pool-top camera is not part of this repository and must never be used as
the real-time control measurement source.

