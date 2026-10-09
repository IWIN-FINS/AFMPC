# 固定线性阻尼双模型融合 MPC

本目录实现鱼目标三维相对位置跟踪。坐标为机体系 FRD：`x` 向前、`y` 向右、`z` 向下；MPC 输出综合平移力

```text
tau = [F_forward, F_right, F_down]  (N)
```

本目录中的基础 `MPCTracker` 只负责平移；正式/实验实机入口已固定改用
`MPC_dual_model_yaw.RotationAwareMPCTracker`。实机上位机直接控制 yaw，下位机继续稳定
roll/pitch，平移和 yaw 最后进入同一套 8 推进器混控。

## 1. 双模型

按 `MPC-model.pdf`，固定 Fossen 低速模型应显式区分恢复/环境平衡力
`tau_h=h(0)`、上一拍实际总力 `tau_previous` 和优化增量
`u=Delta tau`：

```text
M_t vehicle_velocity_dot + D_L vehicle_velocity + tau_h = tau
tau_h = [0,0,0.807290] N  # 当前实机候选，body FRD
```

相对速度模型精确零阶保持离散化得到 `F,G`。两个候选模型应为：

```text
模型1: v1[j+1] = F v[j] + G (tau[j] - tau_base,k)
模型2: v2[j+1] = F v[j] + G (tau[j] - tau_h)
```

模型1假设目标延续求解起点的匹配运动；平移 tracker 按轴用 gated EMA 从实际执行力
更新 `tau_base,k`，并在该次完整 horizon 内固定使用 `tau[j]-tau_base,k`。模型2假设目标静止，使用扣除固定恢复力后的
潜器净作用力。相邻预测力差 `Delta_tau[j]=tau[j]-tau[j-1]` 单独用于变化率代价和约束，
不会替换本轮固定的 model-1 base。

**实现状态：** 活动平移 `MPC_dual_model` 已恢复 gated EMA 工作点，同时继续把 `tau_h`
作为独立 Fossen 项保存。EMA 只改变模型1的 `tau_base`，不覆盖 `tau_h`。`tau_achieved` 来自下位机
最终混控后油门回显；逐桨 RPM 推力曲线只并行记录，不再覆盖控制反馈。实际视觉间隔按遥测 yaw 增量旋转，MPC horizon 按当前 yaw rate
旋转；FOV/距离约束使用相机外参和相机原点。完整公式、不变量及仍未解决的时基/执行器
偏差见 `MODEL_PDF_IMPLEMENTATION_AUDIT.md`。

各方向的融合权重由历史多步位置预测误差计算。每个历史时刻保存
模型一和模型二的状态预测起点；随后使用实际执行过的综合力逐步推进。未来
第 `h` 步位置到达时，再用滤波后的位置计算误差。于是：

```text
e_m(i,h) = p_filtered[i+h] - p_prediction_m[i+h | i]
M1 = sum(gamma_i * beta_h * e1(i,h)^2)
M2 = sum(gamma_i * beta_h * e2(i,h)^2)
C  = sum(gamma_i * beta_h * e1(i,h) * e2(i,h))
a1 = clip((M2-C+epsilon)/(M1+M2-2C+2epsilon), amin, 1-amin)
a2 = 1-a1
```

当两模型的候选差异 `Delta=M1+M2-2C` 小于每个方向的噪声门槛时，不用噪声级差异更新该方向，而是保持该方向上一权重。当前门槛为 `(1e-7,1e-7,1e-7) m2`。`Delta` 明显大于门槛时仍使用上式：模型一残差更小则 `a1` 接近 `0.99`，快速变速阶段模型二残差更小时 `a1` 向 `0.01` 降低。

默认初始 `a1=0.8`，先偏向模型1；有足够观测后由在线误差自动修正。主跟踪器和实机入口统一采用 6 个起点、最大 3 步预测的阶梯历史：

```text
起点年龄 r:       1  2  3  4  5  6
允许最大步长 H_r: 3  3  2  2  1  1
预测步权重 omega: 0.5, 0.3, 0.2
```

令当前时刻为 `k`、预测起点为 `t0`、被预测的目标时刻为 `s`：

```text
r = k-t0
h = s-t0
只评价 1 <= h <= min(r,H_r)
```

例如起点为 `k-3` 时，保留 `k-2|k-3` 和 `k-1|k-3`，不评价三步的 `k|k-3`。`k-3|k-3` 是两个模型共同的滤波初值，`h=0`，只作为回放起点而不参与评分。每个当前时刻仅在实际启用的阶梯格子上重新归一化时间/步长权重。

`build_default_staircase_fusion()` 是默认阶梯参数的唯一构造入口，`MPCTracker` 和 `live_integration_example.py` 均调用它，避免一个入口使用阶梯而另一个入口又回到 20×8。`FusionConfig(staircase_horizon_caps=None)` 仍保留原来的全多步历史方式，仅供显式对照实验和现有调用兼容。预测起点的模型权重和 `tau_base` 均冻结；历史记录也保持各自起点的 base。

活动 QP 直接优化绝对总力序列。令 `R_j` 把第 `j` 步 body FRD 坐标转到下一预测
body FRD（horizon 内用当前 yaw rate 恒定外推），融合式为：

```text
v[j+1] = R_j(F v[j] + G tau[j])
         - A1 R_j G tau_base,k - A2 R_j G tau_h
Delta tau[0] = tau[0] - tau_achieved,k-1
Delta tau[j] = tau[j] - tau[j-1]
```

其中 `A2=I-A1`。`a1=1` 退化为本轮固定工作点模型1，`a1=0` 退化为扣除固定恢复力的目标静止模型2。上一实际力是第一项变化率的参考；`tau_base,k` 则由 gated EMA 独立更新，后续相邻计划力只参与变化率。

## 2. FineSUB 推进器分配

电机顺序与 `TaskSUB.cpp` 一致：

```text
0 LFLower, 1 LFUpper, 2 LBUpper, 3 LBLower,
4 RBLower, 5 RBUpper, 6 RFUpper, 7 RFLower
```

上层输入 `[roll,pitch,down]`，下层输入 `[yaw,forward,right]`。`FineSUBThrusterAllocator` 原样采用 `V5_SUB.hpp` 的两个 4x3 矩阵及写电机时的正负号，并在最终对每台电机独立限幅到 `[-1,1]`。

检查到的源码把已经算出的深度 PID 输出替换成了常数 `0.0`。本实现默认 `enable_depth=True`，因为要完成三轴 MPC；若只想逐字复现那一行，可设为 `False`。

在 UnderwaterVision 当前直驱链路中，MPC 不把力分配成 8 路命令，而是直接发送合力；
库默认/仿真 QP 使用 V4 Pro1 的 8 路物理包络约束。QP 使用的 canonical 顺序为

```text
[V_LF, V_LB, V_RB, V_RF, H_LF, H_LB, H_RB, H_RF]
```

逐台正向限额为 `[8.4749,7.3809,7.3809,8.4749,7.3809,8.4749,7.3809,8.4749] N`，反向限额为 `[7.9750,5.7618,5.7618,7.9750,5.7618,7.9750,5.7618,7.9750] N`。四台垂直推进器均分 `F_down`；四台 45 度水平推进器按 `[F+R,F-R,-F-R,-F+R]/(2 sqrt(2))` 分配。由此得到纯轴保守边界 `force_min=(-16.2968,-16.2968,-23.0472) N`、`force_max=(16.2968,16.2968,29.5236) N`，斜向请求还会被 8 条逐推进器非对称不等式进一步收紧。

所以前向和右向不能同时各达到单轴最大值，且正反向不再假设对称。只有在确实向 MCU/ESC 下发 8 路命令时，才使用下面的分配器输出。注意只能选择一种发送方式：

- 仍向现有 MCU 发送 forward/right/depth 高层命令：使用 `output.device_command`，由 MCU 完成混控。
- Python 直接控制 8 个 ESC：使用 `output.thruster_allocation.throttles`。

不能先在 Python 分配成 8 路后，再让 MCU 做第二次混控。

`active_mpc_parameters.controller` 中的 `I3` 仍供基础平移 tracker 使用；安全实机入口会在
严格 yaw 构造器中用实机 M1–M8 顺序的 `[Fx,Fright,Fdown,N_yaw]` 矩阵替换这一约束，
让平移和直接 yaw 共享推进器余量。由于 CG/CB 和 roll/pitch 实际 PID 输出尚不可用，QP
不会伪造完整六轴矩阵，也无法预留未知的 roll/pitch 瞬时用量。

## 3. 文件

- `fossen_fixed_dl_model.py`：Fossen 模型和精确离散化。
- `relative_kalman.py`：仅用相对位置估计相对速度，无需 DVL。
- `model_fusion.py`：历史多步位置预测误差评分和在线融合权重。
- `mpc_controller.py`：代价函数、约束、双模型增广预测和 QP。
- `device_adapter.py`：现有高层命令映射及 FineSUB 8 推进器分配。
- `mpc_tracker.py`：测量、滤波、权重更新、MPC 和分配的总入口。
- `auto_readiness.py`：正式 AUTO 的只读预检和全部硬门禁。
- `auto_tracker.py`：只从已批准实机参数构造控制器，不允许仿真默认值回退。
- `auto_only_runtime.py`：无手柄 AUTO 状态机，只读新视觉 JSONL。
- `live_integration_example.py`：接入现有视频控制循环的示例。
- `example_simulation.py`：不连接实机的闭环演示。
- `offline_tuning_audit.py`：只读复算实机 trace 的时基、混控、RPM 力候选、每轮固定 base 和 horizon 性能。
- `REAL_DEVICE_TUNING_GUIDE.md`：实机 MPC/PID 全参数清单、症状判据和推荐调参顺序。
- `MODEL_PDF_IMPLEMENTATION_AUDIT.md`：逐项对照 `MPC-model.pdf` 的每轮固定 base、固定 Fossen 恢复力与当前实现/剩余偏差。

## 4. 安装与测试

```powershell
uv sync
uv run python -m unittest discover -s tests -v
uv run python example_simulation.py
```

QP 默认选择 OSQP，并将单次求解时间限制为 40 ms；如果运行环境尚未安装
OSQP，会自动使用 NumPy ADMM 后备求解器。后备求解器已经把每轮迭代的
重复通用线性求解改成一次预计算后的矩阵向量乘法，但实机仍建议安装
`requirements.txt` 中的 OSQP。上述 fallback 只适用于仿真/离线库调用；正式
AUTO 门禁强制 `backend=osqp` 且 `time_limit_seconds<=0.035`，OSQP 缺失时不会连接实机。

活动控制器仍以绝对总力 `tau[j]` 为 QP 决策变量，持续力代价直接惩罚总推进力：

```text
J_force  = sum_j tau[j]^T R tau[j]
```

变化量代价保持为 `Delta_tau[j]^T S Delta_tau[j]`。`tau_base,k` 和 `tau_h` 仍只进入
双模型状态预测，不再作为持续力代价的中心。QP 超时、失败或实验模式视觉缺口时，力受
变化率约束地回到固定 `tau_h`。trace 继续记录 `tau_h`、本轮固定 `tau_base,k`、计划 `tau`
和计划 `Delta_tau`；兼容字段 `mpc_force_reference_frd_n` 当前记录零向量。

## 5. 正式实机 AUTO 入口

正式使用不经过手柄，也不在 MANUAL/AUTO 之间切换。唯一入口是仓库根目录的
`finesub_auto_control.py`，模型固定为 `dual-yaw`，视觉输入固定为外部红鱼系统产生的
JSONL。MPC 只读该文件，不启动、停止或修改视觉系统。

默认命令只做预检，绝不打开 UDP/串口：

```powershell
uv run --project MPC_dual_model python finesub_auto_control.py
```

只有输出 `AUTO READY` 后，显式加入 `--execute` 才允许建立通信并执行 AUTO：

```powershell
uv run --project MPC_dual_model python finesub_auto_control.py --execute
```

预检同时强制检查：正式入口模式、`dual-yaw` 模型、视觉文件存在、视觉质量门、冻结证据
SHA-256、相机到机体刚性外参、三轴实机 `M_t/D_L`、Kalman/MPC/fusion 完整参数、
v5 协议、`<=0.10` 通道限幅、同艇 RPM/推力先验以及所有未决门清零。参数缺失不会
回退到 `live_integration_example.py` 的 UnderwaterVision 数值。

执行时先确认 disarmed-zero 新会话，再等待新鲜遥测、执行反馈和连续视觉，随后只发送
armed-zero；只有 armed-zero 得到同会话/序号/CRC 的肯定回显后，第一条新视觉记录才会
触发 MPC。锁定后任何视觉拒绝或超时、遥测/failsafe、命令拒绝/确认超时、通信发送失败或
QP fallback 都会发送 disarmed zero、退出并要求重启进程。不存在自动重捕获后自行再次上锁。

当前 `finesub_v4pro1_mpc.json` 的 `auto_runtime.enabled=false`，相机外参和三轴实机水动力
参数也未获批准，所以预检会返回 `AUTO BLOCKED`；这正是当前预期状态。
`rov_track_control3.py` 只保留旧手动/CSRT 查看用途，其中手柄按钮 3 已永久拒绝 AUTO。

## 6. 最小库调用方式

```python
from live_integration_example import build_tracker, one_control_update
import numpy as np

tracker = build_tracker()
last_force_body = np.zeros(3)
tracker.latch_baseline(last_force_body)

output = one_control_update(
    tracker,
    position_camera_xyz,
    last_force_body,
)

tau = output.mpc.force
a1 = output.mpc.model1_weight
a2 = output.mpc.model2_weight
cmd = output.device_command
motor = output.thruster_allocation.throttles
last_force_body = tau.copy()  # 没有力反馈时的近似
```

`position_camera_xyz` 是 OpenCV 相机坐标 `[right,down,forward]`（米）。默认转换为机体系 `[forward,right,down]`。

外部红鱼视觉系统保持独立运行时，MPC 只读其 JSONL，不直接修改视觉代码。使用
`PipelineJsonlTail` 和 `VisionMeasurementGate` 读取新记录，并且只有
`decision.control_ready` 为真时才允许把 `decision.measurement.position_camera_xyz_m`
送入相机外参变换；其余情况必须走 `target_lost`/禁止锁存 AUTO。门限覆盖结果时效、
pipeline 延迟、置信度、depth NIS、工作距离、非物理跳变和重捕获连续帧数。
当前控制任务前向工作范围为 `0.15--1.50 m`；新刚性目标冻结回放曾暴露约 `2.0 m` 的稳定错误簇，
旧 `2.50 m` 上限在重捕获后可能放行 2 帧，因此 MPC 输入门已收紧到 `1.50 m`。

```python
from MPC_dual_model import PipelineJsonlTail, VisionMeasurementGate

source = PipelineJsonlTail("output/session/pipeline_results.jsonl")
gate = VisionMeasurementGate()
for record in source.poll():
    decision = gate.evaluate(record)
    if decision.control_ready:
        position_camera_xyz = decision.measurement.position_camera_xyz_m
```

当前三轮动态标定回放中，887 条记录有 793 条通过控制门；实测的 4 帧
`1.84--1.85 m` 假深度全部被拒绝。相机旋转仍是候选，配置中的
`enabled_for_control=false` 尚未解除，因此上述接口本身不构成进入 AUTO 的许可。

2026-08-13 又使用 Tag17 中心到红鱼视觉中心约 `0.23 m` 的刚性竖直偏移，对中心、左、右、上、
下五个位置做了独立 raw-tag 交叉检查。固定竖直解释下 `437` 对数据保留 `405` 个内点，
RMSE/P95 为 `0.0311/0.0542 m`；新旋转候选与三轮动态候选相差 `7.04 deg`，五位置等权结果相对
全样本又变化 `3.63 deg`；采用 tag 中点位于机体中心正上方约 `0.15 m` 的现场估计后，换算原点与
CAD 候选仍相差约 `0.0724 m`。因此新候选只写入分析与配置证据，
不替换控制外参。可重复命令及完整留一位置结果见 `rigid_target_extrinsic_analysis.py` 和
`calibration_logs/onboard_rigid_target_extrinsic_20260813.analysis.json`。

## 7. 上机前必须标定

示例值只是软件占位值，必须标定：

1. `M_t`：三个方向的等效质量和附加质量。
2. `D_L`：固定线性阻尼。
3. `force_min/max` 与 `delta_force_min/max`。
4. `positive_force_at_limit`：各方向满通道对应的实际综合力。
5. 相机外参与测量噪声。
6. 电机正反号；先拆桨或固定机体逐方向核对。

建议先把力限幅降到额定值的 10%--20%，依次检查 forward、right、down 和 roll、pitch、yaw，再进行三轴水池测试。

完整的实机标定、调参和上机顺序见 `CALIBRATION_CHECKLIST.md`。
