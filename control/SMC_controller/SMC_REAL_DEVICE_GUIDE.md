# FineSUB 全车滑模控制实机链路

该入口只替换上位机的 MPC/QP 控制计算，以下链路保持不变：

- `PipelineJsonlTail` 与视觉质量/创新/时延/范围门禁；
- `FineSUBConnection` 的 UDP、会话、CRC、确认和遥测；
- 失联、failsafe、ACK 过期和执行反馈检查；
- `FineSUBHardwareAdapter` 的非对称力/力矩换算；
- 下位机最终八电机混控，以及本地 roll/pitch 稳定。

SMC 输出四个高层量 `[Fx, Fy, Fz, N]`，再沿原有 FineSUB 适配器发送。平移输入采用目标相对坐标；偏航直接由上位机 SMC 控制。

SMC 状态估计直接复用活动 MPC 的 `FixedLinearDampingRelativeModel`、Kalman 噪声配置和
`RelativePositionKalmanFilter`；不再维护一套独立的差分/EMA 速度参数。视觉重捕获时，
估计器与 SMC 历史一起清零，并使用下位机回传的上一拍实际力重新开始。

## 相机 forward 安全距离

SMC 不使用旧原型的单目框宽测距，而是读取与 MPC 相同的立体视觉 JSONL
三维坐标。`smc_parameters.reference_position` 必须与活动 MPC 的
`controller.reference_position` 完全一致；当前通过同一相机外参换算后为相机
forward `0.6000 m`。

在该目标周围启用相机 forward 安全包络：从 `0.80 m` 开始限制接近力，达到
`0.60 m` 时不允许继续产生相机 forward 正向力，低于 `0.30 m` 时强制施加
退离方向的最小力。安全层修改输出后会同步覆盖 SMC 的上一拍 slew 状态，避免
残留正向力继续推入目标。视觉输入范围保持为 `0.30--1.40 m`；因此
`0.30--1.40 m` 的有效立体视觉进入控制器，低于 `0.30 m` 才触发视觉
重捕获和停桨。

SMC 实验 profile 的平移/偏航通道权限保持为授权的 `±0.20`。启动视觉需连续
3 帧，重捕获需连续 5 帧，深度 NIS 上限为 25。
当前 forward 接近力上限为 `3.20 N`，仅用于距离大于目标的追赶段；在
`0.60--0.80 m` 内仍按距离线性收敛到零，在 `0.60 m` 处不允许继续产生正向
接近力。这只扩大安全包络内的远距离追赶能力，不改变 `0.60 m` 目标、
`0.30 m` 硬下限或 `±0.20` 通道权限。力输出在减小幅值或反向时使用更快的
制动 slew，避免正向力跨过目标后仍持续累积。

SMC 的视觉状态先按图像采集时间更新，再用实际测得的采集到控制延迟向前预测；
这避免把约 `20--80 ms` 的视觉处理延迟当成当前时刻的位置误差。

## 相对运动模型一致性

SMC 与当前 MPC 使用同一相对运动定义：`p_rel = p_target - p_vehicle`、
`v_rel = p_rel_dot`。因此模型为

```text
v_rel_dot = -M^-1 D v_rel - M^-1 (tau - tau_h)
tau = tau_h - D v_rel - M a_rel
```

阻尼项必须是 `-D v_rel`；使用 `+D v_rel` 会把相对速度反馈成正反馈。
每次控制更新的 slew 起点也采用下位机回传的上一拍实际执行力，而不是上一拍主机
请求力，以便把推进器延迟、混控限幅和执行反馈纳入同一模型输入。

## 预检与执行

在 `control/` 目录执行。SMC 使用 `MPC_dual_model` 中与 MPC 共用的通信、安全和动力学依赖，但控制器、参数和入口均保存在独立的 `SMC_controller/` 目录：

```bash
# 正式候选预检；当前未批准参数会 fail-closed
uv run --project MPC_dual_model python -m SMC_controller.finesub_smc_control

# 实验候选预检；不打开硬件
uv run --project MPC_dual_model python -m SMC_controller.finesub_smc_control --experimental

# 只有明确加 --execute 才会连接实机，建议始终限定运行时间
uv run --project MPC_dual_model python -m SMC_controller.finesub_smc_control \
  --experimental --execute --max-runtime-sec 30

# 如果视觉进程使用了新的输出文件，显式指定当前正在写入的 JSONL
uv run --project MPC_dual_model python -m SMC_controller.finesub_smc_control \
  --experimental --vision-jsonl /absolute/path/to/pipeline_results.jsonl \
  --execute --max-runtime-sec 30
```

`finesub_v4pro1_smc.json` 只保存 SMC 增益和基础配置引用；正式参数从 `auto_runtime` 读取，实验参数从 `experimental_auto` 读取。实验模式仍使用现有实验视觉门禁和通道上限，不会改写基础 MPC 配置。

当前配置默认使用 `lower_local_hold`，与已验证 MPC 实机固件保持一致；SMC 仍计算偏航滑模量，但不向下位机发送 direct-yaw。只有确认已刷入并验证支持 direct-yaw 的固件后，才可显式加 `--direct-yaw`。

正式 AUTO 只有在相机/body 外参、三轴动力学、偏航动力学和相关实机验收门禁完成后才可放行。
