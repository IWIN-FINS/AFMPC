# 实机标定与调参清单

以下量必须在水池、拖曳台、固定架或回放数据实验中确定。代码中的默认值仅用于软件仿真，不能视为 FineSUB 的实机参数。

## FineSUB V4 Pro1 实机进度（2026-08-13）

| 阶段 | 当前结论 | 是否可用于控制 |
|---|---|---|
| 通信与零输出 | v5 命令/遥测、CRC、回显和烧录后零输出已通过；失联停机按用户要求跳过 | 仅允许受保护的低限幅标定，不允许 AUTO |
| IMU | roll/pitch/yaw 轴和符号已实测；静态自然姿态约 `roll=-4.1 deg, pitch=+0.35 deg` | 坐标符号可用；零偏仍需按每次上电复核 |
| 深度 | 压力单位链已修复；既有 100× 数据修正后，20 cm 点误差约 `-0.59 mm`，烧录后水下约 20 cm 读数为 `0.2152 m` | 相对尺度与 FRD 正号可用；不作为绝对池面世界深度 |
| 电机映射/符号 | M1--M8、逻辑到 DSHOT、正反向、水平/垂直分组和四通道混控均已在水中 `<=10%` 验证 | 低限幅标定模式可用 |
| 指令到 RPM | M1--M8 的 `+/-5/7.5/10%` 共48点已记录；RPM 起转中位/P95约 `0.104/0.154 s`，全部电机在正反 `5%` 均持续转动 | 可作执行状态和低限幅时序参考，不等于牛顿推力 |
| RPM 到推力 | 操作者确认 FinsSim M005--M008 水中测力扫频来自同一台潜器；当前 M1--M8 的 48 个 RPM 点已代入，`10%` 单台约正向 `0.85--1.02 N`、反向绝对值 `0.65--0.95 N` | 水平低限幅曲线按同艇实测采用；垂直四台按同推进单元旋向组迁移，置信度中低 |
| 质量/几何/配平 | GitHub 已溯源 `12.11 kg` 校准质量先验（中等置信，缺原始称重记录）；V4 STEP 推进器轴线与位置已提取；用户接受 CG/CB 水平近似重合、竖向低优先级 | `12.11 kg` 只作刚体质量初值；水平平移/平衡 yaw 可采用该简化，完整姿态力矩分配仍禁用 |
| 水平水动力 | surge 两档与短滑行已有候选。sway 又用池顶折射校正位姿完成 `+/-7.5%` 双向 2 s 脉冲和 10 s 滑行；正/负净位移 `+0.529/-0.495 m`，但拟合窗口从 2 s 增至 12 s 时 `M_eff` 从 `15.2` 漂到 `38.1 kg`、`D_L` 从 `16.15` 降到 `6.17 N/(m/s)` | sway 确认不是 raw PnP 单独导致；系绳/水流/yaw 耦合或线性模型失配仍在，不选择单值 |
| 升沉水动力 | 正浮力实机在 `+5%/+7.5%` 两档各完成两次 2 s 向下脉冲与自然上浮；联合候选 `M_eff=26.26 kg`、`D_L=11.29 N/(m/s)`、净浮力约 `0.81 N` 向上，RMSE `4.9 mm`；跨幅值候选门通过 | 已解决“完全未识别”，但单向推力、竖直推力曲线迁移和系绳影响尚未解除，暂不启用控制 |
| yaw 动力学 | `I_eff≈0.3345 kg*m²`、线性阻尼 `≈0.3225 N*m/(rad/s)` 为低置信候选 | 否 |
| 相机 | 旧三轮动态候选偏离名义 `6.11 deg`；新刚性 Tag17/红鱼五位置拟合为 `10.12 deg`，`405` 内点的 RMSE/P95 为 `3.11/5.42 cm`，两候选相差 `7.04 deg`；采用操作者约 `15 cm` tag 高度后，换算原点与 CAD 相差 `7.24 cm` | 输入安全门可用；外参接受门全部未通过，不启用 MPC 状态修正 |
| MPC 权重/闭环 | 正式无手柄 AUTO-only 入口及 fail-closed 门禁已完成；尚无获批三轴实机模型、yaw 闭环和实机调权 | 当前预检必须阻断；以后只允许固定 `dual-yaw` 的 `finesub_auto_control.py --execute` |
| 风险接受实验 AUTO | 操作者接受当前外参与 sway/heave 候选误差；独立 `finesub_experimental_auto.py` 使用 `[0.8,0,0] m` 参考和三轴 `+/-0.10` 实测上限 | 仅用于首次红鱼追踪实验；逐帧留痕，不改变正式 AUTO 阻断状态 |


原始数据、SHA-256、方法、单位、拟合值和置信度统一见
`PARAMETER_PROVENANCE.md` 与 `calibration_logs/*.analysis.json`。

## 下次开机的最少测量（按顺序）

以下项目做完前不启动 AUTO。正式入口已经固定使用 `dual-yaw`，不再接受手柄切换或
`--model` 选择；旧 `rov_track_control3.py` 的按钮 3 会明确拒绝 AUTO。

1. **舱载相机剩余核对**：三轮动态记录和五位置刚性 Tag17/红鱼记录均已完成；不要再重复同一种
   手持布置。五位置固定竖直拟合为 `RMSE=0.0311 m, P95=0.0542 m`，与旧动态旋转相差 `7.04 deg`，
   留一位置最大旋转变化 `3.53 deg`；采用操作者估计的 tag 中点正上方约 `0.15 m` 后，换算的相机
   原点与 CAD 仍相差约 `0.0724 m`。
   若要解除门槛，应把已知三维标靶直接刚性固定到 ROV body，或直接测量相机安装姿态与 tag-body
   高度；再用当前相机做棋盘重投影复核。红鱼继续换位置不会消除上述基准歧义。
2. **相机延迟**：连续动态数据给出约 `0.10 s`；五个短片段的 1 mm-RMSE 近优区间为
   `0.06--0.25 s`，只能作为一致性检查。它们都不是曝光到执行器的端到端延迟，不能写固定补偿。
3. **可选复核**：若以后使用超过 `10%` 的输出，再记录满量程 RPM 和电池电压；若要最终解除
   sway 门槛，应取消系绳横向载荷并约束 yaw，或采用能区分线性/二次阻尼的更大空间拖曳/多幅值试验；
   不再重复同一种池面松系绳单幅值滑行。

本轮按用户要求不再重复水深尺规测量；MS5837 单位、相对尺度和 FRD 正号已用既有 100× 数据复核。
继续暂缓的是升沉候选的最终控制批准、绝对池面水深基准、CG/CB 垂向间距、roll/pitch 力矩、二次阻尼和 MPC 权重。
这些项目不会用仿真值自动补齐。

## 正式 AUTO 的软件门禁（2026-08-13）

只读预检命令：

```powershell
uv run --project MPC_dual_model python finesub_auto_control.py
```

当前必须输出 `AUTO BLOCKED`，且在返回前不会创建网络/串口 transport。待本清单的外参和三轴
动力学门全部通过、配置的 `enabled_for_control` 显式开启且 `unresolved_gates=[]` 后，才使用：

```powershell
uv run --project MPC_dual_model python finesub_auto_control.py --execute
```

启动不是直接给非零力：disarmed-zero 会话确认 → 新鲜遥测/执行反馈/连续视觉 →
armed-zero 确认 → 下一条新视觉触发第一拍 MPC。运行后任何视觉、遥测、命令确认、transport
或 QP 故障都锁死停机，必须重启进程；程序没有手柄、MANUAL 或运行中重新 AUTO 的路径。

## 当前风险接受实验入口（2026-08-13）

操作者要求停止重复测量，并明确接受当前标定误差先进行实物实验。只读预检：

```powershell
uv run --project MPC_dual_model python finesub_experimental_auto.py
```

预检必须输出 `EXPERIMENTAL AUTO READY` 后才可执行：

```powershell
uv run --project MPC_dual_model python finesub_experimental_auto.py --execute
```

它使用现有三轮动态外参候选、CAD 相机原点、surge/sway/heave 实机候选和 `[0.80,0,0] m` 参考；
三轴均可达到已验证的绝对通道 `0.10`，默认运行 60 s。已知的 `7.04 deg` 外参分歧和 sway
窗口敏感性保留为实验风险，不再阻塞该入口，但仍阻塞正式 AUTO。实验入口继续执行所有在线信号、
命令确认和求解器锁死停机，并为每次运行自动保存逐帧 JSONL。

## Unity UnderwaterVision 当前填写状态（2026-08-08）

结论：仿真接入所需参数已经填写，但 `CALIBRATION_CHECKLIST` 尚未完成实机标定。下表区分已验证值、仿真假设和当前直驱模式不使用的参数；“待实测”项目不能仅靠 Unity 补齐。

| 类别 | 当前值 | 状态与依据 |
|---|---|---|
| 控制坐标与可见量 | MPC 只接收机体系前/右/下的目标相对位置 `[x,z,-y]` | Unity 内部用目标与 ROV 变换生成无噪声理想相对量，但 ROS/MPC 不再接收两者的世界位姿。话题为 `/sim/finsrov/mpc/target_relative_body`，frame 为 `FinsROV/mpc_body_frd`。场景关闭 pose、DVL、depth 发布，只保留 IMU。 |
| `rotation_body_from_camera`、`camera_origin_in_body`、双目尺度 | 动态/刚性目标旋转候选分别偏离名义 `6.11/10.12 deg`，彼此相差 `7.04 deg`；CAD 原点候选 `[0.260993,0.007788,-0.123651] m`；仍禁用 | Unity 理想相对量直接输出 FRD。五位置独立交叉检查没有通过残差、留一稳定性、候选一致性和原点一致性门槛，不能据此解除 AUTO。 |
| `position_std` | `(0.015, 0.015, 0.025) m` | 仿真假设，未由静止视觉数据统计。 |
| 时间与延迟 | `dt=0.05 s`；新视觉对池顶参考的有效延迟约 `0.10 s`；MPC JSONL 结果时效门限 `0.25 s` | `0.10 s` 是两个算法输出间的动态延迟，不是曝光到执行器总延迟，暂不做固定补偿。过期结果直接拒绝。 |
| `M_t` | `diag(26.07276, 26.79684, 26.07276) kg` | UnderwaterVision 仿真辨识值，顺序为前/右/下（Unity `X/Z/-Y`）；速度拟合 `R²=0.9986/0.9956/0.9928`。它包含当前 Unity 环境表现出的等效质量，不是实机参数。 |
| `D_L` | `diag(93.88006, 143.69195, 280.86849) N/(m/s)` | 用 `+/-6 N`、`+/-12 N` 三轴阶跃的完整速度响应辨识；拟合得到的二次阻尼接近零，当前采用线性项。原始结果见 `dynamics_20260806_212129.json`。 |
| `acceleration_std` | `(0.35, 0.35, 0.45) m/s^2` | 滤波器默认假设，未由模型残差统计。 |
| `force_min/max` | `min=(-16.2968,-16.2968,-23.0472) N`，`max=(16.2968,16.2968,29.5236) N` | 由 V4 Pro1 canonical 8 台推进器的非对称 `force_n` 限额和 45 度水平安装几何计算；QP 同时逐台约束，斜向力不是独立轴盒限。 |
| `delta_force_min/max` | `+/- (4.0, 4.0, 5.6) N/周期` | 工程限幅，取各轴包络约 20%；执行器响应尚未辨识。 |
| 直驱映射 | `positive_force_at_limit=(16.2968,16.2968,29.5236)`，`signs=(1,1,1)` | 当前 `DirectWorldWrench` 路径已验证；Unity 不执行推进器动力分配，但 MPC QP 仍执行逐台可行域约束。 |
| 设备命令与分配器 | `command_limits=(99,99,45)`；分配器参数未标定 | 当前桥接节点把 `thruster_allocator` 置空，因此这些值不参与 Unity 直驱控制。 |
| MPC 任务 | 参考 `(0.8,0,0) m`，预测步数 `10`，预测时域 `0.5 s` | 已接入并用于仿真调参。 |
| MPC 状态权重 | 位置 `(10000,14000,25000)`，速度 `(2,20,12)`，终端倍数 `4` | 2026-08-08 UnderwaterVision `0.0975 m/s` 跑道轨迹最终值。 |
| MPC 输入权重 | 绝对总力 `(0.003,0.002,0.04)`，力变化 `(0.01,0.01,0.5)` | 横向权重按半圆动态段调整；逐台非对称硬限幅保持不变。 |
| 距离与视场 | 前向距离 `[0.25,1.50] m`；半视场 `(42,30) deg`；边界 `5 deg` | 约束已启用；实际镜头有效视场和双目工作距离待标定。 |
| 松弛变量 | 二次权重 `5e4`，一次权重 `100`，`slack_max=5.0` | 仿真调参值。 |
| 双模型融合 | 初始权重 `(0.8,0.8,0.8)`，最小权重 `0.01`，窗口 `6`，预测 `3`，遗忘 `0.8`，权重更新率 `0.35` | 当前启用。模型一在一次求解/历史起点内固定使用 `tau-tau_base,k`，模型二使用 `tau-tau_h`；阶梯上限 `(3,3,2,2,1,1)`，预测权重 `(0.5,0.3,0.2)`；每个方向独立选择模型。权重更新率只平滑模型选择参数，不平滑控制力。候选差异小于噪声门槛时保持上一权重；差异可辨时才按残差更新。 |
| 融合保护 | 位置误差截断 `(0.5,0.5,0.5) m`，`epsilon=1e-10`，候选差异门槛 `(1e-7,1e-7,1e-7) m2` | `epsilon` 只防止除零；候选差异门槛用于拒绝无法区分的噪声级模型差异，避免匀速轴权重在 `0` 和 `1` 之间抖动。 |
| QP 求解器 | OSQP，`rho=10`，`sigma=1e-8`，最多 `1500` 次，容差 `2e-5/3e-4`，时限 `0.035 s` | 时限已修正为小于 `0.05 s` 控制周期；仍需持续记录最坏求解时间。 |
| `tau_h` 与 `tau_base,k` | 实机 `tau_h=(0,0,0.807290) N`；`tau_base,k=tau_achieved,k-1` | `tau_h` 是固定 Fossen 零速平衡力，只进入 model2/fallback；`tau_base,k` 每次求解直接复制上一实际力并在该 horizon 内固定，只进入 model1。活动链路没有 EMA 学习率；当前 `tau_achieved` 仍是混控后油门反算近似。 |

当前有效场景是水平 XZ 跑道闭环：`0.75 m` 直线、`0.20 m` 半圆半径、恒定深度，依次执行 `+X` 直线、半圆、`-X` 直线、半圆回到起点，速率恒为 `0.0975 m/s`，起跑延迟 `2.0 s` 与目标相对位置发布同步。最终 65 s 试验的匹配段误差均值/P95 为 `0.01237/0.02665 m`，低加速度段 P95 `0.01655 m`，稳态 P95 `0.01032 m`，QP 成功率 `100%`。水平逐推进器包络触顶率为 `23.1%`；`0.10 m/s` 即使用最终权重，匹配段 P95 仍为 `0.03205 m`，因此 `0.0975 m/s` 是当前限额下最高确认速度。

## 1. 坐标系与视觉

| 参数 | 代码位置 | 单位 | 如何确定 |
|---|---|---:|---|
| `rotation_body_from_camera` | `camera_transform.py` | 无 | 采集多个已知相机点和机体系点，拟合刚体旋转；验证相机前/右/下分别映射到机体前/右/下。 |
| `camera_origin_in_body` | `camera_transform.py` | m | 测量左相机光心相对机体控制原点的前/右/下位移。 |
| 双目尺度、左右相机外参 | 视觉系统上游 | m | 用已知距离标定板或标尺验证三维输出尺度。 |
| `position_std` | `KalmanConfig` | m | ROV 和目标固定时采集位置序列，取各轴标准差。 |
| 相机时间戳与总延迟 | 集成层 | s | 记录曝光、传输、推理到控制命令的时间；用于实际 `dt` 和后续延时滤波。 |

## 2. ROV 平移动力学

| 参数 | 代码位置 | 单位 | 如何确定 |
|---|---|---:|---|
| `M_t` | `FixedLinearDampingRelativeModel` | kg（含附加质量） | 三轴阶跃推力/自由衰减实验拟合；允许为 3x3 非对角矩阵。 |
| `D_L` | `FixedLinearDampingRelativeModel` | N/(m/s) | 三轴匀速或自由衰减数据拟合固定线性阻尼。 |
| `dt` | `FixedLinearDampingRelativeModel` | s | 使用相邻有效三维量测的真实时间间隔；不要只假设 0.05。 |
| `acceleration_std` | `KalmanConfig` | m/s² | 从鱼突转、水流和模型残差统计得到；它描述模型无法解释的加速度。 |

## 3. 推进器和低层控制链

| 参数 | 代码位置 | 单位 | 如何确定 |
|---|---|---:|---|
| `force_min/max` | `MPCConfig` | N | 以连续可用推力，不是瞬时额定峰值，作为上限。 |
| `delta_force_min/max` | `MPCConfig` | N/控制周期 | 阶跃试验中由安全推力变化率和执行器响应确定。 |
| `positive_force_at_limit` | `ForceCommandAdapter`、分配器 | N | 分别标定 forward/right/down 的满命令实际综合力。 |
| `signs` | `ForceCommandAdapter` | ±1 | 固定 ROV、低推力逐轴测试正负方向。 |
| `command_limits` | `ForceCommandAdapter` | 设备命令 | 从 MCU 协议和安全限幅确认。 |
| `translation_channel_limits`、`attitude_channel_limits` | `FineSUBThrusterAllocator` | 归一化通道 | 在不发生单电机持续饱和的条件下调定。 |
| `deadband` | `FineSUBThrusterAllocator` | 归一化通道 | 测量推进器开始稳定产生推力的最小命令。 |
| 每台推进器油门-推力曲线、响应延迟 | 当前代码未实现 | N、s | 用推力计、RPM/电流或水池实验建立；用于估计 `tau_achieved`。 |

## 4. MPC 任务与安全参数

| 参数 | 代码位置 | 单位 | 如何确定 |
|---|---|---:|---|
| `reference_position` | `MPCConfig` | m | 任务要求的鱼相对位置，例如 `[0.6,0,0]`。 |
| `horizon` | `MPCConfig` | 控制步 | 先用 8-12；与 `dt` 一起决定预见时间。 |
| `position_weights` | `MPCConfig` | 代价权重 | 由允许位置误差归一化后，在仿真中调节。 |
| `velocity_weights` | `MPCConfig` | 代价权重 | 由允许相对速度和超调容忍度调节。 |
| `force_weights` | `MPCConfig` | 代价权重 | 直接惩罚绝对总推进力；不再减去融合加权基准推进力。 |
| `delta_force_weights` | `MPCConfig` | 代价权重 | 由允许的力变化和平滑性确定。 |
| `forward_distance_min/max` | `MPCConfig` | m | 由安全距离、双目有效工作距离确定。 |
| `horizontal_half_fov_deg`、`vertical_half_fov_deg` | `MPCConfig` | deg | 用实际镜头有效半视场，而不是宣传全视场。 |
| `fov_margin_deg` | `MPCConfig` | deg | 留给检测框误差、相机畸变和控制延迟的安全余量。 |
| 松弛权重与 `slack_max` | `MPCConfig` | 代价、m | 先在仿真中确认突转时可行，再限制可接受的违规程度。 |

## 5. 双模型融合参数

| 参数 | 代码位置 | 单位 | 如何确定 |
|---|---|---:|---|
| `initial_model1_weight` | `FusionConfig` | 0-1 | 根据先验，可先偏向模型一；应由实测残差修正。 |
| `minimum_weight` | `FusionConfig` | 0-0.5 | 防止任一模型完全失去恢复能力。 |
| `window` | `FusionConfig` | 历史起点数 | 大则稳定、慢；小则快、噪声敏感。建议先从 20 开始。 |
| `prediction_horizon` | `FusionConfig` | 控制步 | 多步位置评分长度；建议先从 6-10 开始。 |
| `forgetting_factor` | `FusionConfig` | 0-1 | 越接近 1 越平稳，越小越迅速适应目标机动。 |
| `horizon_weight_decay` | `FusionConfig` | 0-1 | 多步预测越远是否降权；先从 0.9-1.0 开始。 |
| `position_error_clip` | `FusionConfig` | m | 基于正常视觉误差和异常值上界设置。 |
| `epsilon` | `FusionConfig` | 误差平方单位 | 防止两个模型难以区分时除零。 |

## 6. 求解器与运行时安全

| 参数 | 代码位置 | 单位 | 如何确定 |
|---|---|---:|---|
| `backend` | `QPSolverSettings` | 名称 | 实机优先 `osqp`；仿真可 `auto`。 |
| `time_limit_seconds` | `QPSolverSettings` | s | 小于控制周期并留给视觉、通信和安全逻辑时间。 |
| `max_iterations`、`absolute_tolerance`、`relative_tolerance` | `QPSolverSettings` | 无 | 通过最坏工况仿真平衡精度与时间。 |

## 上机顺序

1. 固定 ROV、低推力确认相机坐标和三轴正负号。
2. 标定相机外参和双目尺度。
3. 标定推进器命令到实际综合力，并记录响应延迟。
4. 用固定目标和低推力辨识 `M_t`、`D_L`。
5. 在仿真中调 MPC 和融合权重。
6. 水池低限幅闭环，仅启用前向轴。
7. 依次加入右向、下向、目标机动、在线融合。

## 当前工具边界与记录格式

`calibration_tool.py link` 是第一阶段的停桨通信测试：只发送 `armed=false`
的零帧，并将命令、确认、状态、IMU、压力/深度、四通道回显、8 路实际油门和
RPM（若固件提供）写入 CSV。它不需要视频，也不调用 MPC。

`calibration_tool.py channel-step` 测试 v4 的四个归一化混控通道，默认
5%，硬上限 10%，并要求命令行显式确认水中系留、防护罩和物理急停。四通道经
固件混控后可能同时驱动多台推进器，因此 CSV 中的 M1--M8 回显只能用于验证
混控结果，不能替代单电机编号/符号测试。

`calibration_tool.py motor-step` 使用 MPC v5 的专用单电机标定标志，绕过姿态/深度
混控，主机和固件均将原始 DSHOT 油门硬限为 `+/-0.10`。当前 NX 只接受 37 字节
raw UDP 包，因此当前命令本身就是带 CRC 和 `BB` 帧尾的原生 37 字节帧；MCU
只解析前 32 字节。每次 CSV 仍需另存试验布置、供电电压、推力计读数、原始文件
名、拟合方法、单位、重复性和置信程度。

`calibration_tool.py failsafe-zero` 用于第一阶段失联停机验证。它要求显式确认推进器
电源已断开或全部卸桨；命令幅值始终为零。工具建立 disarmed 会话后短暂确认
armed-zero，再停止发包，要求固件报告 failsafe 且 M1--M8 回显全零，最后恢复
disarmed。任何门禁失败都会以非零退出码结束并继续发送 disarmed 零帧。

## 联合相机/下位机采集与分析

### 当前实机的被动 v1 遥测

2026-08-11 实机只读检查确认，当前 NX 在线地址是 `192.168.0.2`，
`V5StreamerNX.py` 在 `58766/UDP` 接收旧式 37 字节控制帧，并把 MCU 串口数据持续
转发到上位机 `192.168.0.10:54321`。当前 MCU 回传格式是 FinsSim v1 的
`A5 5A`、98 字节 telemetry（约 100 Hz），不是 FineSUB MPC v4 的 `55 54`
telemetry。烧录 v4 前不得运行 v4 的 `link`、`failsafe-zero`、`channel-step` 或
`motor-step`；烧录后先在推进器断电条件下确认 v4 telemetry，再恢复水中低限幅测试。

`legacy_udp_calibration.py` 只绑定本机 UDP 端口并被动记录，不创建命令帧、不发送
UDP 数据，也不订阅相机话题。它保存解码值、原始帧十六进制、源地址、CRC/丢字节
计数和主机/MCU 时间：

```bash
uv run --project MPC_dual_model python -m MPC_dual_model.legacy_udp_calibration \
  --config MPC_dual_model/finesub_v4pro1_mpc.json \
  --csv calibration_logs/static_passive_legacy_v1.csv \
  --duration 30 --phase static_passive_legacy_v1
```

该模式只适合通信质量、静态 IMU、深度和 RPM 零值检查。因为 v1 telemetry 没有
命令确认、实际电机油门或固件 failsafe 状态，不得用它确认失联停机或推进器映射。

`ros_sensor_calibration.py` 将俯视相机 `/finsrov/vision/refracted_pose_6d` 与
FineSUB v4 原始 telemetry 写到同一个主机单调时间轴。工具没有 armed 或非零命令
选项，只会维持 `armed=false` 的全零会话；记录完整 IMU 四元数、三轴角速度、三轴
加速度、深度、压力、M1--M8 油门/RPM、相机位姿、协方差和视觉质量状态。

在 MPC 根目录运行（必须通过 FinsSim 的 ROS 环境，但不修改 FinsSim）：

```bash
/home/fins/UnderwaterSim/Code/FinsSim/ros2_ws/scripts/run_ros2_uv.sh \
  python3 -m MPC_dual_model.ros_sensor_calibration \
  --config MPC_dual_model/finesub_v4pro1_mpc.json \
  --csv calibration_logs/static_disarmed.csv \
  --duration 30 --rate 20 --phase static_disarmed
```

离线分析：

```bash
uv run --project MPC_dual_model python -m MPC_dual_model.calibration_analysis \
  calibration_logs/static_disarmed.csv \
  --output calibration_logs/static_disarmed.analysis.json
```

分析器先检查 CSV 是否出现 armed/非零命令，再统计 IMU 静止零偏候选、噪声、重力
模长、深度漂移、视觉位置噪声。手动依次做 roll/pitch/yaw 激励后，它会枚举三轴
排列和符号，并用视觉四元数导出的机体系角速度与 MCU gyro 做相关，给出轴映射和
相机延迟候选；上下移动达到至少 0.05 m 后，拟合
`vision_world_z = slope * pressure_depth + intercept`，pool-world z-up 的预期斜率为
`-1`。只有对应 `gate` 为 true 的结果才允许进入人工复核，工具不会自动写实机参数。
