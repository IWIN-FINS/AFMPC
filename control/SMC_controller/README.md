# FineSUB 滑模控制器（SMC）

本目录保存实机 SMC 专属代码，与双模型融合平移 MPC 分开管理。SMC 仍复用
`MPC_dual_model/` 中经过验证的通信协议、安全门控、相机坐标变换、动力学模型、
Kalman 状态估计和推进器适配代码，避免维护两套不一致的实机基础链路。

## 目录内容

- `finesub_smc_control.py`：SMC 实机预检与运行入口；
- `smc_controller.py`：全潜器平移与 yaw 滑模控制器；
- `smc_config.py`：SMC 配置加载与运行参数检查；
- `finesub_v4pro1_smc.json`：SMC 增益、安全距离及基础 MPC 配置引用；
- `realtime_smc_position_error_plot.py`：SMC 实时位置误差图；
- `tests/`：SMC 配置、控制器和误差图测试；
- `SMC_REAL_DEVICE_GUIDE.md`：实机链路、模型约定和安全机制的详细说明。

## 安装

SMC 与 MPC 共用同一套 Python 依赖环境。在仓库的 `control/` 目录执行：

```bash
uv sync --project MPC_dual_model --dev
```

## 测试

```bash
uv run --project MPC_dual_model pytest -q SMC_controller/tests
```

## 预检

以下命令只检查配置，不连接实机：

```bash
# 查看全部参数
uv run --project MPC_dual_model python -m SMC_controller.finesub_smc_control --help

# 正式参数预检
uv run --project MPC_dual_model python -m SMC_controller.finesub_smc_control

# 实验参数预检
uv run --project MPC_dual_model python -m SMC_controller.finesub_smc_control --experimental
```

## 实机运行

只有显式添加 `--execute` 才会连接实机并发送控制命令：

```bash
uv run --project MPC_dual_model python -m SMC_controller.finesub_smc_control \
  --experimental \
  --vision-jsonl /absolute/path/to/pipeline_results.jsonl \
  --execute
```

如需限制单次运行时间，可添加 `--max-runtime-sec 30`。完整的实机安全要求和控制
链路说明见 [SMC_REAL_DEVICE_GUIDE.md](SMC_REAL_DEVICE_GUIDE.md)。

## 实时误差图

```bash
uv run --project MPC_dual_model python -m SMC_controller.realtime_smc_position_error_plot
```

## 参数来源

SMC 专属参数以 `finesub_v4pro1_smc.json` 为准。该文件通过 `base_config` 引用
`../MPC_dual_model/finesub_v4pro1_mpc.json`，用于继承双方共用的实机动力学、
硬件限幅、视觉门控和通信配置。`parameters/control/smc.json` 只是自动生成的参数
快照，不作为直接修改入口。
