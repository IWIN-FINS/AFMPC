# FinsROV 实机代码整理

本目录整理了 FinsROV 实机实验使用的代码和参数。

## 目录

- `control/`：融合 MPC、旋转 MPC、固定模型 MPC、PID、SMC、通信协议和实机入口。
- `vision/`：双目相机采集、鱼检测、立体深度、跟踪、滤波和标定工具。
- `firmware/`：V4Pro1 下位机固件。
- `parameters/`：从实际运行配置导出的参数快照。
- `docs/`：参数、结构、数据接口和实机安全说明。

不包含仿真代码和池顶相机代码。模型权重、录像和实验数据不放在本目录中。

## 控制代码

```bash
cd control
uv sync --dev
uv run pytest -q
uv run finsrov-auto-preflight
```

## 双目视觉

```bash
cd vision
uv sync --dev
uv run pytest -q
uv run depth-demo-video --help
```

模型路径配置在 `vision/src/depth_estimation/config.yaml`，需要单独放置模型权重。

## 参数

参数总览见 [`docs/PARAMETERS.md`](docs/PARAMETERS.md)。修改实际运行参数后，可重新导出参数快照：

```bash
cd control
uv run python ../tools/export_parameters.py
```

实机运行必须明确使用 `--execute`，运行前检查推进器、急停、通信、视觉和标定状态。

