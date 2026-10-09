# `MPC-model.pdf` 与活动双模型实现对照

日期：2026-08-14；2026-08-15 更新平移基线策略。本文记录 `MPC_dual_model` 的数学语义、已经完成的修正和仍存在的偏差。

## 1. 已确认的 `tau_base` 语义

原始 PDF 在时刻 `k` 直接使用上一实际力。2026-08-15 操作者根据旧版模型的实机表现，
要求平移融合模型恢复按轴 gated EMA；当前活动平移实现因此明确采用：

```text
eligible_i = |position_error_i| <= 0.20 m
             and |relative_velocity_i| <= 0.20 m/s
alpha_i = 0.02  # |position_error_i| <= 0.03 m and |velocity_i| <= 0.02 m/s
alpha_i = 0.08  # otherwise, while still eligible
tau_base,k,i = (1-alpha_i) tau_base,k-1,i
               + alpha_i tau_achieved,k-1,i
```

三个 FRD 轴均启用；不满足门限的轴保持上一 base。这份 `tau_base,k` 在本次完整预测 horizon 内保持不变。模型1假设鱼继续以求解起点的
潜器速度匀速运动，所以同一轮预测的第 `j` 步为：

此前试拟的 forward 匹配运动例外（速度、加速度和连续确认门）当前配置关闭；活动实现
只使用上面的原始位置/速度资格门。

```text
model 1: v1[j+1] = F v1[j] + G (tau[j] - tau_base,k)
```

到了下一次 MPC 求解，才按门控 EMA 更新 base。因此，先前
“horizon 第2步以后应把 model-1 baseline 换成上一预测力”的解释是错误的。

这项 EMA 是操作者选择的实验策略，和 PDF 的直接
`tau_base,k=tau_achieved,k-1` 并不完全相同；独立 `tau_h`、模型2和力变化率定义不受影响。

预测序列中的相邻力仍然会滚动，但用途是力变化率代价和约束：

```text
Delta tau[k] = tau[k] - tau_achieved,k-1
Delta tau[j] = tau[j] - tau[j-1], j>k
```

它不替换本轮 model-1 的固定 `tau_base,k`。

## 2. 独立的 Fossen 平衡力

低速平移动力学为：

```text
M_t u_dot + h(u) = tau
h(u) = D_L u + D_Q(|u| o u) + g(eta) - tau_env
```

当前零速候选为：

```text
tau_h = h(0) = [0, 0, 0.807290] N  # body FRD，正 z 向下
```

`tau_h` 是固定物理项，不是在线 baseline，也不能被水平力更新覆盖。目标静止模型为：

```text
model 2: v2[j+1] = F v2[j] + G (tau[j] - tau_h)
```

令 `W=diag(a1)`、`A2=I-W`，忽略偏航旋转时的融合式为：

```text
v[j+1] = F v[j] + G tau[j]
         - W G tau_base,k - A2 G tau_h
```

纯 model1 在 `tau[j]=tau_base,k` 时相对速度输入为零；纯 model2 在
`tau[j]=tau_h` 时相对速度输入为零。

## 3. 本次已经完成的修正

| 部分 | 修正后的活动实现 |
|---|---|
| 物理模型 | `restoring_force_frd_n` 独立保存 `tau_h`，不再复用 `model.tau_base` |
| MPC 每轮工作点 | tracker 在位置/速度门内按轴用 gated EMA 学习上一实际执行力；`solve()` 接收该独立 base，并在整个 horizon 的仿射项中保持不变 |
| 力变化率 | 第一项为 `tau[0]-tau_achieved,k-1`，以后为 `tau[j]-tau[j-1]`，只用于代价和硬约束 |
| model2 | 所有预测统一使用 `tau-tau_h` |
| Kalman 均值 | 一步历史区间使用该区间起点的 EMA base 构造 model1，并与 `tau-tau_h` 的 model2 融合 |
| 历史模型评分 | 每个预测起点锁存自己的 `tau_base`；该记录的所有后续步都保持同一基准 |
| 位置离散式 | 按 PDF 使用 `v+=Fv+Gf`、`p+=p+Ts*v+` |
| QP 代价 | 绝对总力仍是决策变量；持续力代价为 `tau^T R tau`；相邻力变化仍为 `Delta tau^T S Delta tau` |
| 不可区分融合 | 保持上一权重，不再无证据地强制拉向 model1 |
| fallback/视觉缺口 | 受变化率约束地回到固定 `tau_h`，不再撤成 armed-zero 或回到 EMA baseline |
| 配置和诊断 | 平移模型使用 `0.02/0.08` gated EMA；匹配运动例外当前关闭；分别记录本轮固定 base、base 与实际力之差、`tau_h` 和计划 `Delta tau` |
| 实际执行力 | `tau_achieved` 由最终混控后电机油门回显反算；8 路 RPM 二次推力曲线继续并行计算并写日志，但不覆盖控制反馈 |
| yaw 坐标 | 状态估计按相邻视觉更新间的实际 yaw 增量旋转；MPC horizon 按当前 FRD yaw 角速度的匀速假设逐步旋转 body 坐标 |
| 相机约束 | body 预测位置先减相机在 body 中的原点，再用外参转成 `[camera-forward,camera-right,camera-down]`，FOV 与距离均约束该相机射线 |

活动控制器仍以绝对力序列 `tau[j]` 作为 QP 决策变量。相邻差分矩阵显式构造
`Delta tau`，变化率代价/约束仍作用于实际相邻总力。为避免融合权重和
`tau_base/tau_h` 在启动时制造维持大推力的线性奖励，当前持续力代价统一改为：

```text
J_force = sum_j tau[j]^T R tau[j]
```

该修改不改变双模型状态方程、horizon 内固定 base、Fossen 平衡力、实际力反馈、绝对/变化率硬约束
或 fallback。代价不再把 `tau_base` 或 `tau_h` 当作省力中心；需要维持非零推进力时，
优化器会明确权衡位置/速度误差、绝对总力和力变化率。`MPCResult.force_reference` 为兼容旧 trace
保留，但当前恒为零，不参与 QP。

## 4. 已加入的关键不变量

- 同一 horizon 连续两步都使用同一个 `tau_base,k`；
- 下一次求解的 base 每轴最多向新的上一实际执行力移动配置限值；
- 首项 `Delta tau` 仍使用未经限速的真实上一实际执行力；
- `tau=tau_h` 时纯 model2 保持零相对速度输入；
- `tau=tau_base,k` 时纯 model1 保持零相对速度输入；
- PDF 的半隐式位置递推成立；
- Kalman 的 model1 一步输入等于两个连续实际力之差；
- 输出同时满足绝对力限制和相邻力变化限制；
- 求解失败与视觉缺口的目标力是固定 `tau_h`；
- yaw 变化时历史预测、Kalman 均值/协方差和 MPC horizon 使用相同的坐标旋转；
- 相机平移和旋转都会改变 FOV/距离约束；
- 无论 RPM 有效位如何，控制 `tau_achieved` 都保持为最终油门回显反算；RPM 候选仅作诊断。

## 5. 仍存在的偏差或模型缺项

以下项目没有在本次公式修正中假装解决：

1. **固定 `dt=0.10 s`。** 视觉实际存在约 `0.10/0.20 s` 混合间隔，但 Kalman、融合和
   MPC 每次只推进一个固定步长。应改成严格 10 Hz 状态 tick，或按采集时间动态离散化。
2. **没有推进器延迟/惯性状态。** MPC 假设指令力立即实现，实测存在命令、起转和停转
   延迟；这会直接导致制动偏晚。
3. **RPM 推力仍是静态先验而非测力传感器。** 水平曲线来自同艇测力，垂直曲线按同型
   推进单元旋向迁移；当前没有推进器动态、RPM 去尖峰或实际水中负载修正。2026-08-14
   水试已证明不能直接作为 `tau_base`/变化率参考，因此它只记录，控制始终使用最终油门回显。
4. **相机/IMU 仍未按曝光时刻对齐。** horizon 已改由上位机 yaw 状态机预测未来
   角度/角速度/力矩，但实机入口仍用处理该视觉结果时的最新 IMU，而不是曝光时刻的
   IMU 历史插值；快速转向时会形成相位误差。
5. **已加入实机 M1..M8 四轴可达域，但 roll/pitch 余量未知。** 安全入口不再使用
   `I3` 或 Unity V/H 顺序，而是联合约束 `[F_frd,N_yaw]`；下位机 roll/pitch PID 的
   未来输出没有遥测/预测接口，仍无法在 QP 中预留其瞬时推进器用量。
6. **水动力仅为固定对角 `M_t,D_L`。** 未包含二次阻尼、轴间耦合、姿态相关恢复项、
   拖缆/水流变化；sway 标定已经显示窗口敏感。
7. **视觉外参仍是风险接受候选。** 两套旋转相差约 `7.0 deg`；现在 FOV/距离约束已正确
   使用所选相机外参，但外参本身的系统误差仍会进入状态和约束。
8. **目标匀速/静止双候选仍是简化。** 手持鱼突然加速、停止或绕行时，模型没有显式目标
   加速度状态，只能依靠 Kalman 过程噪声和在线融合追赶。

这些问题中，1--3 对“刹不住”和实际预测误差的影响最大；4--5 对转向与高输出下的
耦合影响更大；6--8 需要新的标定或模型扩展，不能仅靠扫 `Q/R/S` 修复。
