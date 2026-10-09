# FineSUB 纯 PID 三轴平移 + yaw 控制器

本目录用纯 PID 完成与 `MPC_dual_model` 相同类型的任务：根据双目相机给出的鱼目标三维位置，让 ROV 保持目标位于机体系 FRD

```text
[forward, right, down] = [0.80, 0.0, 0.0] m
```

控制器输出综合平移力和 yaw 力矩；实机 PID 在首次有效遥测时锁存当前 yaw
角作为保持目标，持续抑制偏航误差：

```text
tau = [F_forward, F_right, F_down] N
N_yaw                                      N*m
```

它不使用 Fossen 模型、Kalman、双模型融合、QP、预测时域或在线优化。平移和 yaw 控制量都只来自 P、I、D 三项：

```text
e[k]      = p_target_body[k] - p_reference
de_f[k]   = LPF((e[k] - e[k-1]) / dt)
tau_raw   = tau_baseline + Kp*e + Ki*sum(e*dt) + Kd*de_f

psi_goal  = psi_startup
e_yaw     = wrap(psi_goal - psi_imu)
N_yaw     = Kp_yaw*e_yaw + Ki_yaw*sum(e_yaw*dt) - Kd_yaw*omega_imu
```

`tau_baseline` 是 AUTO 开启时锁存的当前实际合力，只是无扰切换和目标丢失回退值，不是动力学模型。D 项使用一阶低通，第一帧强制为零，避免启用瞬间的微分冲击；I 项有积分限幅和反算抗饱和。

yaw 只控制绕 FRD 下轴的航向角，正 yaw 表示艏部向右转；roll 和 pitch 不在这个 PID 中控制。默认保持首次有效遥测的当前航向，也可以给 `reference_yaw_rad` 指定固定绝对航向。角度使用弧度并通过 `wrap()` 选择最短旋转方向。

## 与 MPC 的对应关系

| 能力 | 纯 PID 实现 |
|---|---|
| 三轴相对位置保持 | 三套独立 PID |
| yaw 航向保持 | 单独的角度 PID，锁存启动时 yaw，D 项使用 IMU yaw 角速度 |
| roll / pitch | 不控制，始终给混控器传 0 |
| 测量噪声处理 | D 项一阶低通、误差死区 |
| 综合力上下限 | `force_min/max` |
| 每拍力变化率 | `delta_force_min/max` |
| 斜向运动的共享电机限制 | 8 条逐推进器非对称力约束 |
| 目标丢失 | 限速返回锁存基线力 |
| MCU 高层命令 | `device_command` |
| Python 直驱 8 个 ESC | `thruster_allocation.throttles` |

PID 没有 MPC 的未来视场约束和前瞻能力，因此无法严格保证未来目标始终留在 FOV 内；它通过较高的横向/垂向增益、微分阻尼、饱和保护来达到相近的跟踪目的。

## 目录

- `pid_controller.py`：纯 PID、D 项低通、抗积分饱和和执行器约束。
- `pid_tracker.py`：与视频循环对接的总入口。
- `yaw_pid_controller.py`：仅 yaw 角度的标量 PID、角度环绕和力矩限幅。
- `device_adapter.py`：FineSUB 命令缩放与 8 推进器混控。
- `camera_transform.py`：OpenCV 相机坐标转机体系 FRD；实机/相机窗口使用
  当前池顶相机的旋转与安装偏移，避免把相机坐标误当成机体坐标。
- `live_integration_example.py`：实时代码接入示例及默认参数。
- `example_simulation.py`：不连接实机的闭环演示。
- `finesub_protocol.py`：与 `V4pro1_MPC` 当前 v5 固件一致的 37 字节命令、174 字节遥测和 CRC16/MODBUS。
- `finesub_transport.py`：MPC 同款 TCP/UDP/USART3 串口、session 握手、命令确认和遥测超时保护。
- `hardware_session.py`：相机测量到 PID、IMU yaw、实际执行力反馈和可配置传输命令的实机总入口。
- `vision_gate.py`：相机距离/跳变/启动确认门控；已锁定后遇到单帧视觉跳变会忽略该帧并保持上一帧有效位置，不把异常坐标送入 PID。
- `hardware_diagnostic.py`：只发 disarm、永不解锁电机的连接诊断程序，支持 MPC 风格 JSON。
- `camera_pid_tuner.py`：池顶相机窗口、ROI/手动目标、实时误差曲线和 OpenCV PID 滑块调参。
- `dual_vision_pid_runtime.py`：实机/干跑的正确双路视觉入口。双目
  `pipeline_results.jsonl` 提供 PID 的三维位置；池顶相机和
  `/finsrov/vision/refracted_pose_6d` 只提供位置、速度、跳变辅助信息和窗口显示。
  旧的 `camera_pid_tuner.py` 单目像素/固定距离模式只适合离线观察，不应作为
  双目实机控制输入。
- `tests/`：控制方向、约束、回退、接口和闭环收敛测试。

## 安装与验证

本项目使用 `uv` 管理独立虚拟环境：

```bash
cd /home/fins/Zhouyuheng_workspace/MPC/PID_controller
uv sync --dev
uv run pytest -q
uv run python example_simulation.py
```

相机窗口需要 GUI 依赖（仍然只改本 PID 目录）：

```bash
uv sync --extra gui
uv run python camera_pid_tuner.py --camera-index 3 --fx 600 --fy 600 --range-m 2.0
```

双目实机/干跑窗口使用现有 stereo pipeline 的 JSONL，池顶视频仅作辅助显示；
下面命令默认保持 disarmed：

```bash
uv run python dual_vision_pid_runtime.py \
  --stereo-jsonl /home/fins/Zhouyuheng_workspace/tracking_depth/depth-estimation-dev/output/<session>/pipeline_results.jsonl \
  --camera-device /dev/video2
```

`--stereo-jsonl` 必须是正在更新的双目输出；如果它停止更新，PID 会进入等待/安全
状态。窗口中的 TOP position/speed 不会送入 `PIDTracker`。只有明确增加
`--enable-arm` 并按下 `a` 才会请求实机 armed；不要把旧的池顶单目窗口接到实机。

若要沿用 MPC 当前的网络桥接配置，只替换传输层，仍使用双目入口：

```bash
uv run python dual_vision_pid_runtime.py \
  --stereo-jsonl /home/fins/Zhouyuheng_workspace/tracking_depth/depth-estimation-dev/output/<session>/pipeline_results.jsonl \
  --runtime-config /home/fins/Zhouyuheng_workspace/MPC/MPC_dual_model/finesub_v4pro1_mpc.json \
  --camera-device /dev/video2
```

设备索引以本机 `v4l2-ctl --list-devices` 为准，也可以传
`--camera-device /dev/video0`。本机探测到 OpenCV index 0、3 可读，不能假定
`video2` 一定是池顶相机；先在窗口里确认画面再框选目标。

启动后首帧用鼠标框选池中目标；也可以直接给 `--roi x,y,w,h`，或用
`--manual-target U V` 固定一个像素点。窗口含 `PID camera`、`PID error` 和
`PID tuning` 三个视图。调参滑块会即时修改已有的 forward/right/down 和
yaw PID 增益、参考位置以及单目固定距离，不会清除积分和微分历史。按
`r` 重置 PID，`d`/空格 disarm，`q`/Esc 退出。

默认始终 dry-run；不会打开任何传输。串口可以用 `--port`，而 MPC 同款
TCP/UDP 连接用 `--runtime-config /path/to/runtime.json`。无论哪种方式，
启动先发送 disarm，且只有同时加 `--enable-arm` 后按 `a` 才会请求 armed。
池顶相机若没有标定内参，程序会在画面上警告；近似内参只适合观察曲线，
不能直接用于实机闭环。实机前仍须拆桨/固定机体并先运行：

```bash
uv run python hardware_diagnostic.py --runtime-config \
  /home/fins/Zhouyuheng_workspace/MPC/MPC_dual_model/finesub_v4pro1_mpc.json \
  --seconds 10
```

## V4pro1_MPC 实机接口

当前 PID 上位机匹配下位机源码中的 v5 原生协议：

- USART3，115200 baud，8N1；板端引脚为 `PD8/TX`、`PD9/RX`。
- 命令顺序为归一化 `[forward,right,down,yaw]`，范围分别为 `±0.35/±0.35/±0.50/±0.20`。
- 实机默认使用下位机 `LOWER_LOCAL_HOLD` yaw 模式：armed 首帧锁存当前 yaw，避免未经验证的 direct-yaw 通道和共享推进器混控引起旋转；上位机仍计算 yaw PID 误差用于窗口诊断。下位机 roll/pitch PID 仍然工作，PID 上位机不控制 roll/pitch。
- 新上位机进程先发送 disarm 帧并等待 session/sequence/CRC 回执，确认后才允许发送 armed 帧。
- 遥测超过 0.20 秒、命令被拒绝、failsafe 或缺少实际执行反馈时，上位机只发送零值 disarm。
- armed 期间遥测失联、failsafe、执行反馈异常或持续的无效/范围错误视觉会锁存 disarm；单帧视觉跳变只保持上一帧有效位置并继续 armed，不会直接停机。
- 下位机自身超过 500 ms 没有命令时也会 failsafe 并要求重新 disarm 握手。

PID 实机实验的默认平移包络与当前 MPC 实验包络一致：
`force_max=[4.730162,4.997534,7.06314] N`、每拍变化率
`[1.2,0.8,1.0] N`、PID 实机归一化平移通道 `±0.20`；yaw 实机不再发送
direct-yaw，而是使用下位机当前角度保持环。这不是实机最终 PID 调参值，
只是保留当前已授权的实验包络并避免未验证 direct-yaw 共享混控引起旋转。
实机入口和相机窗口还使用同一套 PID 目录内的相机外参：OpenCV
`[right,down,forward]` 经过旋转后得到 FRD `[forward,right,down]`，并加上
相机在机体中的安装原点偏移；其 `[0,0,0.60] m` 图像中心点对应当前池顶
相机标定参考位置。`camera_to_body_position()` 仍保留轴对齐的显式基线，
但实机路径不会再调用这个未标定基线。
MPC 风格 JSON 的视觉范围和跳变边界会被只读复用。PID 启动仍要求至少
3 个连续样本；已锁定后单帧跳变会被忽略并保持上一帧有效位置。若新位置
连续满足重捕获确认次数，则在保持 armed 的情况下无缝切换；若只是回到旧
位置则立即恢复跟踪，不会因为一次跳变触发 disarm。

第一次接线后，拆桨或可靠固定机体，只运行永不解锁的诊断：

```bash
uv run python hardware_diagnostic.py --port /dev/ttyUSB0 --seconds 10
```

必须持续看到 `session_ok=True`、`armed=False`、CRC 无错误后，才能接入视觉循环。控制循环必须以 20 Hz 左右调用：

```python
from hardware_session import build_serial_hardware_session

hardware = build_serial_hardware_session("/dev/ttyUSB0")
hardware.connect()  # 这里只会先发零值 disarm

try:
    while running:
        position_camera_xyz = get_latest_stereo_position()
        if position_camera_xyz is None:
            hardware.target_lost()  # 默认立即 disarm
            continue

        result = hardware.step(
            position_camera_xyz,
            arm_requested=operator_arm_switch,
        )
        print(result.status)
finally:
    hardware.close()
```

实机串口路径中不要再发送 `device_command` 或 Python 的 8 路 `throttles`；`hardware_session` 已把 PID 的牛顿/牛米输出转换为下位机归一化四通道，再由当前 `V4pro1_MPC` 完成最终电机混控。

## 最小接入

```python
import numpy as np

from live_integration_example import build_tracker, one_control_update

tracker = build_tracker(calibrated_reference=True)
last_force_body = np.zeros(3)
last_yaw_moment = 0.0

# AUTO 从 OFF 切到 ON，只调用一次：
tracker.latch_baseline(last_force_body, last_yaw_moment, imu_yaw_rad)

# 每次得到相机坐标 [right, down, forward] 后：
output = one_control_update(
    tracker,
    position_camera_xyz,
    last_force_body,
    imu_yaw_rad=imu_yaw_rad,
    imu_yaw_rate_rad_s=imu_yaw_rate_rad_s,
    last_achieved_yaw_moment=last_yaw_moment,
    calibrated_camera=True,
)
tau = output.pid.force
N_yaw = output.yaw_pid.yaw_moment
yaw_channel = output.yaw_channel
cmd = output.device_command
motor = output.thruster_allocation.throttles
last_force_body = tau.copy()  # 无力反馈时使用上一拍饱和命令近似
last_yaw_moment = N_yaw

# 目标丢失：
safe = tracker.target_lost(last_force_body, last_yaw_moment)
last_force_body = safe.force.copy()
last_yaw_moment = safe.yaw_moment
```

只能选择一种下发路径：继续向 MCU 发送 `device_command` 加 `yaw_channel`，或由 Python 将已经包含 yaw 的 `throttles` 直发 8 个 ESC，不能二次混控。旧的 `device_command` 结构只包含平移三通道，因此 MCU 直控时必须把 `yaw_channel` 同时送入支持 yaw-direct 的协议，不能丢弃它。
reloadreloadreload
## 上机调参顺序

默认 PID 增益是基于当前仿真参数给出的安全起点，不是实机最终值。建议先把输出限幅降至额定值的 10%–20%，固定 ROV 并逐轴核对正负号，然后按下面顺序调参：

1. 先令 `Ki=0, Kd=0`，从小到大增加 `Kp`，直到响应足够快但尚未持续振荡。
2. 增加 `Kd` 抑制超调；相机噪声导致推力抖动时增大 `derivative_filter_time_constant`。
3. 最后缓慢增加 `Ki` 消除恒定水流造成的静差。
4. forward、right、down 分开调好后，再检查斜向运动是否触发共享推进器约束。
5. 标定 `force_min/max`、`delta_force_min/max`、电机正反向力和命令符号后才能放开输出。

若相机帧率不是固定 20 Hz，必须把 `PIDConfig.dt` 改成实际控制周期；抖动很大时应进一步把 `dt` 改成每拍实测值接口后再上机。
