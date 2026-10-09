# 参数快照

这里保存便于查看和对比的参数副本，不直接修改这些文件。

实际参数来源：

- MPC、yaw、硬件和通信协议：`control/MPC_dual_model/*.json`
- SMC：`control/MPC_dual_model/finesub_v4pro1_smc.json`
- PID：`control/PID_controller/live_integration_example.py`
- 双目视觉：`vision/src/depth_estimation/config.yaml`

修改实际参数后重新生成：

```bash
cd control
uv run python ../tools/export_parameters.py
```

