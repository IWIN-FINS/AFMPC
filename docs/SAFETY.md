# Hardware safety

This repository can command physical thrusters. Treat every hardware run as a
hazardous operation.

Before enabling output:

1. Confirm the vehicle and firmware revision match the configuration.
2. Verify the command/telemetry protocol in disarmed mode.
3. Verify thruster order, signs, force limits, camera transform, and IMU signs.
4. Keep people, cables, and loose objects away from every propeller.
5. Use a working physical emergency stop and a separate operator.
6. Start with propellers removed or the vehicle mechanically secured.
7. Review preflight blockers; do not bypass missing calibration evidence.

Runtime invariants:

- startup sends a disarmed zero command;
- hardware execution requires an explicit `--execute` flag;
- stale telemetry, rejected commands, failsafe state, or invalid vision causes
  a zero/disarmed transition according to the guarded runtime;
- shutdown sends disarmed zero;
- direct calibration and autonomous command envelopes are distinct.

配置文件中的通信目标是示例地址。实机连接前应根据当前设备填写并重新检查。
