# FinsROV 控制代码

## 目录

- `MPC_dual_model/`：双模型融合平移 MPC、SMC、滤波、通信和实机运行代码。
- `MPC_dual_model_yaw/`：加入 yaw 控制的旋转 MPC。
- `MPC_model1/`：固定模型 1 MPC。
- `MPC_model2/`：固定模型 2 MPC。
- `PID_controller/`：三轴 PID 和 yaw PID。
- `apps/`：实机命令行入口。
- `legacy/`：保留的旧版 SMC 代码，不作为当前实机入口。

主要运行配置：`MPC_dual_model/finesub_v4pro1_mpc.json`。

## 安装与测试

```bash
uv sync --dev
uv run pytest -q
```

只检查配置、不连接硬件：

```bash
uv run finsrov-auto-preflight
```

其他入口：

```bash
uv run finsrov-experimental-auto --help
uv run finsrov-smc --help
uv run finsrov-hardware-diagnostic --help
```

没有明确添加 `--execute` 时，不会开始实机控制。

