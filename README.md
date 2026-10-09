# FinsROV 实机代码

本仓库归类保存实际实验使用的原始代码。控制、视觉和下位机文件均从实验机现有工作目录原样复制，没有改写代码、参数或路径。

## 目录

- `control/`：融合 MPC、旋转 MPC、固定模型 MPC、PID、SMC、通信协议和实机入口。
- `vision/`：双目相机采集、鱼检测、立体深度、跟踪、滤波和标定工具。
- `firmware/`：V4Pro1 下位机固件。
- `parameters/`：实机参数索引；实际运行参数以 `control/` 中的原始 JSON 为准。
- `docs/`：参数、结构、数据接口和实机安全说明。

不包含 Unity/仿真代码和池顶相机代码。模型权重、录像、运行日志和实验数据不提交到仓库。

## 控制代码

融合 MPC、旋转 MPC、固定模型、SMC 和实机通信代码位于 `control/`。原始实机入口为：

```bash
cd control
uv sync --project MPC_dual_model
uv run --project MPC_dual_model python finesub_experimental_auto.py --help
uv run --project MPC_dual_model python finesub_smc_control.py --help
```

PID 使用自己的原始环境：

```bash
cd control
uv sync --project PID_controller
```

## 双目视觉

```bash
cd vision
uv sync --dev
uv run pytest -q
uv run depth-demo-video --help
```

模型和标定路径保持实验时的原始配置，见 `vision/src/depth_estimation/config.yaml`。模型权重仍放在实验机原位置。

参数总览见 `docs/PARAMETERS.md`，原始一键运行记录见 `control/START_VISION_TRACKING.md`。实机运行必须明确使用 `--execute`。
