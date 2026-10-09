# yaw 旋转版实机候选、标定和参数确认清单

本清单已按当前代码逐个配置项核对。平移参数读取当前 FineSUB 实机实验块；yaw 动力学、
六轴几何和上层双环 PID 仍是未获准正式控制的候选，不能据此跳过低限幅验证。

- **测量/辨识**：应由尺寸、传感器、系泊或水池试验得到；
- **实验整定**：需在仿真和低限幅水池试验中逐步调节；
- **设计选择**：不是传感器能直接测出的量，但必须明确选择并记录依据。

## A. 坐标、采样与传感器（测量/辨识）

- [ ] 标定相机到机体系旋转矩阵 `R_bc`，不要直接沿用
  `ALIGNED_OPENCV_TO_BODY` 示例矩阵。
- [ ] 测量机体控制原点到相机光心的杠杆臂 `r_bc_body`，单位 m。
- [ ] 标定相机三维位置的比例、零偏、畸变和随距离变化的误差。
- [ ] 确认机体系为 `x` 前、`y` 右、`z` 下；确认相机输入三个轴的顺序和符号。
- [ ] 确认正 `psi`、正 `omega`、正 `N` 都使艏部向右转。
- [ ] 测量实际控制周期/相机周期 `dt`，统计平均值、抖动和丢帧；当前代码要求平移、
  yaw 和滤波使用同一个固定 `dt`。
- [ ] 对齐相机曝光时间、IMU yaw、IMU `omega` 和上周期执行力的时间戳。
- [ ] 测量相机链路、IMU、通信和执行器延迟；实现相机曝光时刻对应的 IMU 历史姿态
  回放后再进入自动控制。
- [ ] 测量并补偿 IMU yaw 和角速度零偏、温漂与噪声；验证跨越 `+pi/-pi` 时角度增量
  连续。
- [ ] 确认实机能否获得 `last_achieved_force_body` 和
  `last_achieved_yaw_moment`；若只能得到命令值，必须另建执行器/推力估计器，不能把命令
  直接标成实际力。

## B. 平移和 yaw 动力学（测量/辨识）

- [ ] 辨识平移惯性矩阵 `M_t`，包括刚体质量和三轴附加质量/耦合项，单位 kg。
- [ ] 辨识低速线性平移阻尼矩阵 `D_L`，单位 N/(m/s)。
- [ ] 用开环或系泊试验验证由 `M_t,D_L,dt` 得到的离散 `F,G` 的一步和多步预测误差。
- [ ] 判断实际工作速度下二次平移阻尼是否可忽略。当前代码没有 `D_Q`；若不可忽略，
  应先扩展模型，不能仅靠调大 `D_L` 掩盖。
- [x] `tau_base,k=tau_achieved,k-1`：每次求解直接锁存上一实际力并在本次 horizon 内
  固定；旧 gated EMA 已删除。验证执行回显的时间索引和单位正确即可。
- [ ] 辨识有效 yaw 惯量 `effective_inertia=m_omega=I_z-N_rdot`，单位 kg·m²。
- [ ] 辨识线性 yaw 阻尼 `linear_damping=d_omega`，单位 N·m/(rad/s)。
- [x] 当前实机候选：`effective_inertia=0.33453415 kg*m²`、
  `linear_damping=0.32251723 N*m/(rad/s)`；置信度中低，仍为
  `enabled_for_control=false`。
- [ ] 辨识二次 yaw 阻尼 `quadratic_damping=d_omega2`；若数据不足暂设 0，必须同时限制
  工作角速度并记录适用范围。
- [ ] 测量持续 yaw 偏置力矩/扰动力矩；当前模型没有 `d_N` 估计器，若偏置明显应补充
  扰动状态或明确设置安全基线 `yaw_moment_base`。
- [ ] 分别验证一步 `omega[k+1]`、一步 `Delta psi[k]` 和多步 `psi[k+j]` 的开环预测误差。

## C. 卡尔曼滤波（测量/实验整定）

- [ ] 由静态目标数据统计三轴位置测量标准差 `position_std`。
- [ ] 由目标运动、水流和模型残差整定三轴等效加速度标准差 `acceleration_std`。
- [ ] 根据首次检测误差设置 `initial_position_std`。
- [ ] 根据首次速度未知程度设置 `initial_velocity_std`；当前初始速度均值固定为 0。
- [ ] 检查旋转期间创新是否有 yaw 相关偏差；当前滤波器没有传播 IMU yaw 不确定性。
- [ ] 用遮挡、跳点和误检数据确定创新门限/离群点策略；当前代码尚未实现该功能。

## D. 双模型历史评价（实验整定/设计选择）

- [x] 当前实验 `staircase_horizon_caps=(5,5,5,5,5,4,3,1)`：与平移融合参数同步。
- [x] 当前实验 `prediction_horizon_weights=(1,1,1,1,1)`：与平移融合参数同步。
- [ ] `forgetting_factor=0.8`：用不同机动阶段的数据检查旧样本衰减速度。
- [ ] `position_error_clip=(0.5,0.5,0.5)`：按视觉离群误差和正常模型误差分布确定。
- [ ] `epsilon=1e-10`：检查误差很小时权重是否受数值项主导。
- [ ] `indistinguishable_score_threshold=(1e-7,1e-7,1e-7)`：确认低噪声仿真中
  双模型不可区分时保持模型一偏好的阈值合适。
- [ ] `weight_update_rate=0.10`：检查模型权重平滑速度能否兼顾转向瞬态与稳态辨识。
- [ ] `minimum_weight=0.01`：决定是否保留双模型最低占比；它使代码不能得到 0/1
  纯模型权重，与 PDF 公式不同。
- [ ] 当前实验 `initial_model1_weight=(0,0,0)`：归一化后受 `minimum_weight=0.01` 保护，
  与当前平移融合配置一致。
- [ ] 验证所有启用格子的权重归一化、当前拍更新次序和三轴权重变化符合实验预期。

## E. yaw 状态机与双环 PID（实验整定/设计选择）

当前保守候选：`alpha_on/off/emergency=6.0/1.2/8.0 deg`，
`outer_kp/kd=1.5/0.2`，`inner_kp/ki=0.60/0.02`，最大角速度 `30 deg/s`，
最大角加速度 `60 deg/s²`；力矩绝对限额来自实机通道映射
`-2.041126..+1.807854 N*m`，每个 `0.10 s` 周期变化 `+-0.50 N*m`。这些 PID 数值尚未
水池闭环验证。

- [ ] `alpha_on`：正常平移跟踪仍可靠、但需要开始转头的视线角阈值。
- [ ] `alpha_off`：必须小于 `alpha_on`，形成回差，避免反复切换。
- [ ] `alpha_emergency`：必须大于 `alpha_on` 且小于扣除裕量后的真实视场边界。
- [ ] `trigger_frames`、`settle_frames`：结合视觉帧率、误检持续时间和减速时间确定。
- [ ] `require_outward_motion_to_trigger`：决定是否必须满足视线继续向外运动才进入 TURN。
- [ ] `yaw_tolerance`、`omega_tolerance`：确定转向完成的角度与角速度阈值。
- [ ] 先整定内环 `inner_kp/inner_ki/inner_kd` 和 `inner_integral_limit`，再启用外环。
- [ ] 整定外环 `outer_kp/outer_ki/outer_kd` 和 `outer_integral_limit`；建议先令积分为 0。
- [ ] `omega_command_max`：由视觉清晰度、模型适用范围和可用力矩共同确定。
- [ ] `omega_command_acceleration_max`：限制期望角速度变化，避免转向命令跳变。
- [ ] `yaw_moment_min/max`：由实机可用 yaw 力矩确定，正负方向应分别测量。
- [ ] `delta_yaw_moment_min/max`：由执行器响应、供电和姿态闭环稳定性确定。
- [ ] `use_dynamics_feedforward`：在 yaw 参数可信且反馈环稳定后再决定是否启用。
- [ ] 验证 TURN 中紧急重定向、跨角度边界和目标短时跳动均不会使
  `omega_command` 或 `N` 不连续。

## F. MPC 参考、权重与约束（实验整定/设计选择）

- [ ] `horizon` 和 `dt`：确认总预测时间 `horizon*dt` 覆盖主要平移响应时间。
- [ ] `reference_position=[d*,0,z*]`：根据相机成像、目标尺寸和跟踪任务确定。
- [ ] `position_weights`、`velocity_weights`：逐轴确定位置和相对速度优先级。
- [ ] `terminal_weight_scale`：当前同时缩放位置与速度终端权重，不能独立设置。
- [ ] `force_weights`、`delta_force_weights`：在跟踪、能耗和控制平滑性之间整定。
- [x] 力代价固定为 PDF 的绝对总力；旧 `effective-force` 模式和开关已删除。
- [x] 三轴数值限额读取当前实验块：最小 `[-5.050680,-4.783308,-6.867140] N`，最大
  `[4.730162,4.997534,7.063140] N`；QP 同时保留实机 M1..M8 四轴平移+yaw 等式。逐桨限额目前是旧
  全油门曲线按 `0.20` 通道缩放的先验，不是实测联合可达域。
- [ ] `delta_force_min/max`：三轴每周期力变化限值。
- [ ] `forward_distance_min/max`：可测距离、碰撞安全距离和任务距离范围。
- [ ] `horizontal_half_fov_deg`、`vertical_half_fov_deg`：使用实际有效视场半角，不只照抄
  镜头标称值。
- [ ] `fov_margin_deg`：覆盖延迟、估计误差和一个控制周期内可能发生的目标运动。
- [ ] `forward_axis/horizontal_axis/vertical_axis`：与实机机体系索引核对；默认 0/1/2。
- [ ] `slack_quadratic_weight`、`slack_linear_weight`、`slack_max`：用可恢复越界和严重
  越界场景整定。
- [ ] 记录 OSQP `rho`、`sigma`、`max_iterations`、绝对/相对容差、终止检查周期和时间
  限制；用目标硬件验证平均、95% 分位和最坏求解时间均满足 10 Hz 预算。
- [ ] 人为制造不可行、超时和数值失败，验证输出按限速返回上次可行力或固定 `tau_h`，
  而不是突变为零力。

## G. 力/力矩通道和控制分配（测量/辨识）

- [ ] `ForceCommandAdapter.positive_force_at_limit`：分别测量三轴正向通道限值对应的力。
- [ ] `ForceCommandAdapter.signs`：逐轴验证机体系力与下位机通道符号。
- [ ] `ForceCommandAdapter.command_limits`：与下位机协议的整数通道范围一致。
- [ ] `YawMomentChannelAdapter.positive_yaw_moment_at_limit`：测量正 yaw 通道限值对应的
  实际力矩。
- [ ] `YawMomentChannelAdapter.negative_yaw_moment_at_limit`：单独测量负 yaw 通道限值，
  不假设正反对称。
- [ ] `YawMomentChannelAdapter.channel_limit` 和 `sign`：与下位机 yaw 通道范围、符号一致。
- [ ] `FineSUBThrusterAllocator.positive_force_at_limit`：与平移通道标定保持一致。
- [ ] `translation_channel_limits`、`attitude_channel_limits`：按供电、温升和同时动作时的
  饱和余量确定。
- [ ] `deadband`：由电机实际启动/停止试验测量；验证正反方向是否需要不同死区。
- [ ] `enable_depth`：根据本次实验是否允许深度推力明确设置。
- [ ] 逐个验证八路电机编号、旋向、符号和代码混控矩阵与固件当前版本一致。
- [ ] 标定真实推力曲线、正反推力不对称、电压影响、流速影响和推进器互扰；当前转换
  只采用线性对称比例。
- [x] 当前 QP 已将平移和冻结 yaw 力矩共同放入实机 M1..M8 四轴推进器约束，并保留
  八推进器零空间；没有用未知质心伪造 roll/pitch 力矩行。
- [ ] 增加下位机 roll/pitch PID 实际输出/需求遥测，或建立保守余量，之后才能把姿态
  占用纳入联合可达域。
- [ ] 只能选择“发送高层力/姿态通道”或“Python 直接发送电机输出”一种执行链路，
  避免与 MCU 重复混控。

## H. 安全逻辑与上水顺序

- [ ] 验证启动时上一实际力能正确成为首个 `tau_base`；目标丢失时平移应限速退回固定
  `tau_h`，yaw 力矩退回 `yaw_moment_base`。
- [ ] 明确视觉丢失、时间戳异常、IMU 无效、连续 QP 失败和通信中断的退出阈值。
- [ ] 完成全部单元测试和无硬件闭环仿真。
- [ ] 固定机体，低限幅逐轴验证 `Fx/Fy/Fz/N` 符号和实际执行反馈符号。
- [ ] 仅启用 yaw 内环，初始限幅取预计可用值的 10%--20%。
- [ ] 加入 yaw 外环和状态机，验证右侧目标产生正 `N`。
- [ ] 只做低速纯 yaw，确认旋转补偿后的估计平移速度接近 0。
- [ ] 再启用平移 MPC 与 yaw 联合控制，记录测量、滤波状态、模型权重、slack、QP
  状态、求解耗时、实际力/力矩和各电机饱和。
- [ ] 所有参数应保留“辨识日期、试验工况、单位、置信区间/重复性和适用速度范围”。

## I. 当前仍需改代码后才能标定的项目

- [ ] 相机延迟对应的 IMU 历史姿态回放。
- [ ] roll/pitch 旋转补偿或完整三维姿态补偿。
- [ ] 二次平移阻尼 `D_Q`（若实验表明不可忽略）。
- [ ] yaw 扰动力矩/偏置估计（若实验表明不可忽略）。
- [ ] 视觉离群点门控、缺测滤波和测量置信度自适应噪声。
- [ ] 把后级执行饱和与实际达到的力/力矩可靠反馈给滤波和历史评价。
