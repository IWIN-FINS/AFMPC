# FinsROV 实机控制代码

## 目录

- `MPC_dual_model/`：双模型融合平移 MPC、滤波、通信和实机运行代码。
- `SMC_controller/`：SMC 控制器、参数、实机入口、实时误差图和测试。
- `MPC_dual_model_yaw/`：加入 yaw 控制的旋转 MPC。
- `MPC_model1/`：固定模型 1 MPC。
- `MPC_model2/`：固定模型 2 MPC。
- `PID_controller/`：三轴 PID 和 yaw PID。
- `finesub_experimental_auto.py`：融合/旋转 MPC 实机入口。
- `SMC_controller/finesub_smc_control.py`：SMC 实机入口。
- `finesub_auto_control.py`：正式 AUTO 预检入口。

主要运行配置：`MPC_dual_model/finesub_v4pro1_mpc.json`。

## 安装与测试

```bash
cd MPC_dual_model
uv sync --dev
uv run pytest -q
cd ..
uv run --project MPC_dual_model pytest -q SMC_controller/tests MPC_dual_model_yaw/tests MPC_model1/tests MPC_model2/tests
```

只检查配置、不连接硬件：

```bash
uv run --project MPC_dual_model python finesub_auto_control.py
```

其他入口：

```bash
uv run --project MPC_dual_model python finesub_experimental_auto.py --help
uv run --project MPC_dual_model python -m SMC_controller.finesub_smc_control --help
uv run --project PID_controller python -m PID_controller.hardware_diagnostic --help
```

没有明确添加 `--execute` 时，不会开始实机控制。
