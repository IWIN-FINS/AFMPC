# 加入 yaw 旋转的双模型融合 MPC

本目录是在 `MPC_dual_model` 平移融合代码上增加 yaw 后的实验版本。坐标约定为机体系 FRD：`x` 向前、`y` 向右、`z` 向下；正 yaw、正角速度 `omega` 和正力矩 `N` 均表示艏部向右转。

## 1. 当前控制结构

本版与 PDF 的分层结构一致：

```text
相机位置 + IMU yaw/omega
        -> 旋转补偿卡尔曼滤波
        -> 二维历史阶梯双模型融合
        -> yaw HOLD/TURN/SETTLE 状态机
        -> yaw 角度外环 + 角速度内环，得到 N
        -> 冻结未来 psi/omega/N 轨迹
        -> 联合 QP 优化 Delta tau 和每步八台推进器力
        -> 实机 M1..M8 四轴等式同时满足平移和冻结 yaw 力矩
```

`N` 不再是 QP 的第 4 个决策量，但它以冻结参数进入每个预测步的推进器等式。一次 QP 求解期间，yaw 目标及预测旋转矩阵固定；到下一实际控制周期，控制器根据新相机和 IMU 测量重新生成。

## 2. 旋转相对运动模型

目标位置 `p` 和平移相对速度 `v` 都在当前机体系表示。若一步内实际或预测 yaw 增量为 `Delta psi`：

```text
R = Rz(-Delta psi)
v[k+1] = R (F v[k] + G f_eff[k])
p[k+1] = R p[k] + Ts v[k+1]
```

这与 PDF 的“先预测下一步速度，再做多步位置递推”一致。纯转头且没有平移时：

```text
p[k] = Rz(-Delta psi) p[k-1]
v_rel = (p[k] - Rz(-Delta psi)p[k-1]) / Ts = 0
```

因此转头产生的视觉运动不会被误认为目标的平移速度。

yaw 动力学采用：

```text
m_omega * omega_dot
  + d_omega * omega
  + d_omega2 * |omega| * omega = N
psi_dot = omega
```

线性部分精确离散，二次阻尼在单个采样周期内冻结。角度增量使用梯形积分：

```text
Delta psi[j] = Ts/2 * (omega[j] + omega[j+1])
```

而不是旧版的 `Ts*omega[j]`。

## 3. yaw 状态机和双环 PID

状态机定义于 `yaw_controller.py`：

- `HOLD`：保持当前长期艏向 `psi1`，主要依靠平移 MPC 跟踪。
- `TURN`：当 `|alpha| >= alpha_on` 连续若干帧时，令 `psi2=wrap(psi+alpha)` 并转向。
- `SETTLE`：目标回到 `alpha_off` 内后继续减速；实际角度、角速度和视线角连续满足完成条件后，执行 `psi1 <- psi2`。

偏航角使用相机光心射线，而不是机体原点到目标的方位角。令
`R_vis_body` 把机体系向量转换到按 `[相机前、相机右、相机下]` 排列的可视坐标系，
`r_bc` 为相机光心在机体系的位置，则：

```text
p_vis = R_vis_body (p_body - r_bc)
alpha = atan2(p_vis,right, p_vis,forward)
```

MPC 的水平、垂直视场约束也作用于同一个 `p_vis`，平移动力学状态仍为 `p_body`。

外环根据 `wrap(psi_goal-psi)` 生成受最大角速度和最大角加速度限制的 `omega_command`；内环根据 `omega_command-omega` 生成受绝对值和变化率限制的 `N`。积分器带限幅和条件抗饱和。目标继续向视场外运动并超过紧急阈值时，只更新终点 `psi2`，不会重置当前 PID 状态。

## 4. 平移双模型和 QP

设模型一权重为逐轴对角矩阵 `A1`，决策变量为
`u[j]=Delta tau[j]`。`tau_base,k` 在每次求解开始时直接取上一拍实际达到的力，
并在该次完整 horizon 内保持不变；它不是 EMA 参数。`tau_h=h(0)` 是固定写入
Fossen 方程的恢复/平衡力：

```text
model1: f_eff[j] = tau[j] - tau_base
model2: f_eff[j] = tau[j] - tau_h
v[j+1] = R[j]Fv[j] + R[j]Gtau[j]
         - A1 R[j]Gtau_base - (I-A1) R[j]Gtau_h
```

令 `e=p-p_d`，增广状态为：

```text
x[j] = [e[j], v[j], tau[j-1]]
tau[j] = tau[j-1] + u[j]
```

位置递推为：

```text
e[j+1] = R[j]e[j] + Ts v[j+1] + (R[j]-I)p_d
```

这形成仿射线性时变模型 `x[j+1]=A[j]x[j]+B[j]u[j]+d[j]`，因此在冻结 yaw 轨迹后仍是凸 QP。

代价包括多步位置误差、速度、绝对总力和相邻力增量：

```text
J_force = sum tau[j]' R_tau tau[j]
J_delta = sum (tau[j]-tau[j-1])' S_tau (tau[j]-tau[j-1])
```

第一项力增量为 `tau[0]-tau_achieved[k-1]`。旧的 `effective-force` 模式及
`force_cost_mode` 开关已经删除，避免基线力被错误当成“免费控制量”。

约束包括：

- 三轴绝对力限制；
- 三轴力增量限制；
- V4 Pro1 八台推进器逐台、正反向非对称的平移+yaw 联合可达域；
- 带松弛量的水平、垂直视场和前向距离限制。

每个预测步额外包含八个推进器力变量 `u_thr`，并执行：

```text
W_4x8 u_thr[j] = [Fx,Fright,Fdown,N_yaw][j]
-F_reverse[i] <= u_thr[i,j] <= F_forward[i]
```

因此平移和 yaw 会占用同一组推进器余量，且保留八推进器相对四轴任务的零空间，
不再用固定伪逆分配缩小可达域。矩阵顺序与固件/遥测一致，直接采用实机 M1..M8
正油门力方向和 CAD 原点 yaw 力臂；当前实验构造器将同艇全油门逐桨曲线缩放到
`0.20` 包络。未知 CG 和本地下位机 roll/pitch PID 用量不进入 QP，所以这仍不是完整
六轴实测可达域。

## 5. 二维历史阶梯评价

旋转版继续使用：

```text
1 <= h <= min(r,H,H_cap(r))
H_cap = [5,5,5,5,5,4,3,1]
omega_h = [1,1,1,1,1]
```

历史回放每一步都使用实际执行力、预测起点保存的 `tau_base` 和实际 IMU yaw 增量，
把两个模型预测转到同一目标时刻的机体系后再评分。`h=0` 只作为共同滤波起点，
不参与评分；启用格子的权重会重新归一化。

## 6. 运行

在当前 MPC 工作区下：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --project MPC_dual_model --group dev \
  python -m pytest -q MPC_dual_model_yaw/tests
uv run --project MPC_dual_model python -m MPC_dual_model_yaw.example_yaw_simulation
```

实时入口：

```python
from MPC_dual_model_yaw.live_integration_example import (
    build_tracker,
    one_control_update,
    to_finesub_command,
)

tracker = build_tracker()
tracker.latch_baseline(last_force, last_yaw_moment, imu_yaw_rad)

output = one_control_update(
    tracker=tracker,
    position_camera_xyz=position_camera_xyz,
    imu_yaw_rad=imu_yaw_rad,
    imu_yaw_rate_rad_s=imu_yaw_rate_rad_s,
    last_achieved_force_body=last_force,
    last_achieved_yaw_moment=last_yaw_moment,
    rotation_body_from_camera=R_bc,
    camera_origin_in_body=r_bc_body,
)

tau = output.mpc.force
N = output.yaw_control.yaw_moment
mode = output.yaw_control.mode
motor_force_diagnostic = output.mpc.thruster_force
command = to_finesub_command(output, armed=True)
```

相机位置、IMU yaw 和 IMU `omega` 必须对应同一拍摄时刻。`one_control_update()`
会同时保留两套几何量：转换到机体系的位置供平移动力学和卡尔曼滤波使用；从相机光心
出发的 `[前、右、下]` 视线射线供 yaw 状态机和水平/垂直视场约束使用。因此非零
`r_bc_body` 或非正装相机不会凭空产生偏航误差。实机安全入口发送高层
`[forward,right,down,yaw]`，由 MCU 混控；`output.mpc.thruster_force` 是 QP 可达域
对应的诊断分配，不能再作为第二套命令重复下发。

调用 `target_lost()` 后，平移力按变化率退回固定 `tau_h`，yaw 力矩退回其锁存基准；同时
旧目标的卡尔曼状态、融合历史、yaw 目标/PID 和时间基准会被清空。下一帧有效视觉测量
会以当前 IMU yaw 建立新的 `HOLD` 方向，并从该测量重新初始化目标位置和速度，避免
完成丢失前遗留的转向目标或把多帧缺测压缩成一次 `dt` 状态更新。

## 7. 当前实机候选参数与边界

- `build_tracker()` 和安全实机入口共用严格构造器，读取
  `experimental_auto.active_mpc_parameters` 与 `active_yaw_parameters`：`dt=0.10 s`、`N=15`、实机 `M/D`、
  `tau_h=[0,0,0.80729] N`、0.6 m 相机参考对应的机体系参考点、当前 Q/Qv/R/S、
  卡尔曼、融合、相机外参与 `0.20` 三轴限额，不再复制仿真参数。
- yaw 动力学读取实机候选 `effective_inertia=0.33453415 kg*m^2`、
  `linear_damping=0.32251723 N*m/(rad/s)`；它在参数文件中仍为
  `enabled_for_control=false`，上层双环 PID 也尚未水池闭环验证。
- `build_hardware_adapter()` 与正式平移入口共用最终电机油门回显反算实际力；RPM 推力
  估计只记录诊断，不会覆盖 `tau_achieved/tau_base`。
- 仅补偿 yaw，不包含 roll/pitch 对视觉相对坐标的旋转影响。
- 尚未实现相机延迟的 IMU 历史回放。
- QP 使用实机 M1..M8 的四轴平移+yaw 几何和正反推力限额；roll/pitch 余量、死区、
  电池电压、推进器互扰及几何/质心误差仍未建模。
- 默认要求 OSQP；未安装时会在控制器构造阶段明确报错，不会在 10 Hz 循环中静默切换到较慢求解器。

完整实机参数见 `CALIBRATION_CHECKLIST.md`；与 PDF 和参考推导的逐项差异见
`MODEL_DIFFERENCES.md`。
