# 固定线性阻尼模型一 MPC

本目录实现 FineSUB 鱼目标三维相对位置跟踪，并且可以独立运行。坐标系、控制器输入输出、QP 约束、设备命令和八推进器分配格式与
`MPC_dual_model` 对齐，但预测、估计和公开接口只包含模型一，不导入融合模型目录，也不计算模型二或融合权重。

## 1. 模型一

状态和控制量为：

```text
x = [p_rel, v_rel]
p_rel = p_target - p_vehicle
tau = [F_forward, F_right, F_down]  (N)
```

固定线性阻尼低速模型为：

```text
M_t v_rel_dot + D_L v_rel = -(tau - tau_base)
p_rel_dot = v_rel
```

代码对连续系统做精确零阶保持离散化。模型一把上一拍实际综合力视为当前匹配状态的基础力，因此整个预测域滚动使用：

```text
x[k+1] = A_d x[k] + B_d (tau[k] - tau[k-1])
```

为保持 QP 线性，控制器采用增广状态 `z=[x,tau_previous]`：

```text
x+            = A_d x + B_d tau - B_d tau_previous
tau_previous+ = tau
```

本目录不存在 `model_fusion.py`、`OnlineModelFusion`、`model1_weight` 或
`model2_weight`。卡尔曼滤波的预测均值也使用同一个模型一公式。

## 2. MPC 与安全约束

代价包含相对位置、相对速度、有效作用力 `tau[k]-tau[k-1]`、力变化率和终端状态。约束包含：

- 三轴综合力上下限；
- 相邻控制拍的力变化率上下限；
- 前向跟踪距离上下限；
- 水平和垂直相机视场；
- 距离与视场的非负软约束松弛量。

求解器优先使用 OSQP；不可用时回退到 NumPy ADMM。求解超时或失败时先保持上一份可行力；尚无可行解时，按力变化率限制返回 AUTO
启用时锁存的基础力。目标丢失也使用相同的限速回退。

## 3. 坐标和 FineSUB 输出

视觉输入遵循 OpenCV 左相机光学坐标：

```text
[right, down, forward]
```

`camera_to_body_position` 默认转换到 FineSUB 机体系 FRD：

```text
[forward, right, down]
```

`MPCResult.force` 与融合版本的 `force` 格式一致，均为三维平移综合力。跟踪器同时返回：

- `device_command`：现有 MCU 高层前进、右移、深度命令；
- `thruster_allocation.throttles`：FineSUB 固件顺序的 8 路归一化电机量。

两条下发路径只能选择一条，不能先在 Python 中分配 8 路后再让 MCU 二次混控。

## 4. 文件

- `fossen_fixed_dl_model.py`：固定 `D_L` Fossen 平移模型和精确离散化；
- `dense_qp.py`：OSQP 与 NumPy ADMM 求解后端；
- `relative_kalman.py`：仅由相对位置估计相对速度；
- `mpc_controller.py`：模型一增广预测、代价、约束和安全回退；
- `camera_transform.py`：左相机光学系到机体系的坐标转换；
- `device_adapter.py`：高层命令映射与 FineSUB 八推进器分配；
- `mpc_tracker.py`：测量、滤波、MPC 和分配的总入口；
- `live_integration_example.py`：接入现有视频控制循环的示例；
- `example_simulation.py`：不连接实机的闭环示例；
- `MATH_COMPARISON.md`：PDF、融合代码和纯模型一代码的数学差异。

## 5. 安装与验证

在 `D:\FINSMCAT\Machine\MPC` 下运行：

```powershell
python -m pip install -r MPC_model1/requirements.txt
python -m unittest discover -s MPC_model1/tests -v
python -m MPC_model1.example_simulation
```

也可以进入本目录后直接运行 `python example_simulation.py`。

## 6. 最小接入

```python
import numpy as np

from MPC_model1.live_integration_example import (
    build_tracker,
    one_control_update,
)

tracker = build_tracker()
last_force_body = np.zeros(3)

# AUTO 从关闭切换为启用时锁存实际保持力。
tracker.latch_baseline(last_force_body)

output = one_control_update(
    tracker,
    position_camera_xyz,
    last_force_body,
)

tau = output.mpc.force
cmd = output.device_command
motor = output.thruster_allocation.throttles
last_force_body = tau.copy()  # 无力反馈时的近似
```

公开输出没有任何融合权重字段。

## 7. 上机前必须标定

示例参数只是占位值，至少需要实测：

1. 三轴等效质量及附加质量 `M_t`；
2. 固定线性阻尼 `D_L`；
3. 综合力及力变化率限幅；
4. 各方向满命令对应的实际综合力和符号；
5. 相机外参与位置测量噪声；
6. 电机顺序、正反号、死区和饱和。

首次水池测试建议把力限幅降到额定值的 10%--20%，逐轴核对
forward、right、down、roll、pitch、yaw 后再进行三轴闭环测试。
