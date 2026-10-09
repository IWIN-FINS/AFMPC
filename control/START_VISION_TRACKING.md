# FineSUB 一键启动视觉追踪

更新时间：2026-08-16

下面是一条可整段复制到一个 Bash 终端执行的命令。它会依次：

1. 启动双目红鱼视觉程序并打开实时相机窗口（`--pipeline-mode yolo-only`，
   模型与标定取自 `src/depth_estimation/config.yaml` 现有配置）；
2. 固定使用 `--process-every 3`，把视觉结果写入本次独立目录；
3. 打开 FineSUB MPC 实时位置误差图；
4. 以前台方式启动 `dual` 融合平移模型实机实验 AUTO（yaw 由下位机保持），并把它绑定到本次视觉 JSONL 和误差图 trace；
5. 不设置实验时长，直到操作者按 `Ctrl+C` 或控制器因故障退出；退出时自动清理本命令启动的视觉和误差图进程。

该命令只运行现有视觉程序，不修改 `tracking_depth` 的代码或参数文件。参数按
2026-08-14 实机会话的 shell 历史与 `vision.log` 恢复。

```bash
set -u
MPC_DIR=/home/fins/Zhouyuheng_workspace/MPC
VISION_DIR=/home/fins/Zhouyuheng_workspace/tracking_depth/depth-estimation-dev
RUN_DIR="$VISION_DIR/output/finesub_mpc_translation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

# 1. 双目红鱼视觉：UDP 5600 并排流 → YOLO+立体深度 → 本次目录 JSONL
cd "$VISION_DIR"
DISPLAY=:0 CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m depth_estimation.demo_video \
  --config src/depth_estimation/config.yaml \
  --pipeline-mode yolo-only \
  --udp-sbs-port 5600 --udp-latency-ms 20 \
  --display-scale 0.8 --display-every 1 --process-every 3 --print-every 10 \
  --async-display-mode live \
  --record-monitor "$RUN_DIR/monitor.mp4" --record-every 3 --record-fps 10 \
  --record-raw "$RUN_DIR/raw_stereo.mp4" \
  --result-jsonl "$RUN_DIR/pipeline_results.jsonl" >"$RUN_DIR/vision.log" 2>&1 &
VISION_PID=$!

# 2. FineSUB MPC 实时位置误差图（默认跟随最新 experimental_auto trace）
cd "$MPC_DIR"
uv run --project MPC_dual_model python -m MPC_dual_model.realtime_position_error_plot \
  --forward-only \
  >"$RUN_DIR/error_plot.log" 2>&1 &
PLOT_PID=$!

cleanup() {
  kill "$VISION_PID" "$PLOT_PID" 2>/dev/null
  wait "$VISION_PID" "$PLOT_PID" 2>/dev/null
}
trap cleanup EXIT

# 3. 最多等 30 秒视觉 JSONL；超时通常表示 UDP 5600 视频流没有到达
jsonl_waited=0
until [ -s "$RUN_DIR/pipeline_results.jsonl" ]; do
  sleep 1
  jsonl_waited=$((jsonl_waited + 1))
  if [ "$jsonl_waited" -ge 30 ]; then
    echo "30 秒内没有 JSONL，自动停止；请查看 $RUN_DIR/vision.log" >&2
    exit 1
  fi
done
echo "视觉结果目录：$RUN_DIR"

# 4. 前台运行 dual 平移实机实验 AUTO，绑定本次视觉 JSONL；Ctrl+C 结束
uv run --project MPC_dual_model python finesub_experimental_auto.py \
  --execute --vision-jsonl "$RUN_DIR/pipeline_results.jsonl"
```

## 运行结果位置

- 相机窗口：视觉程序的实时 GUI；
- 实时误差窗口：`FineSUB MPC Position Error — Live`；
- 视觉录像：`tracking_depth/depth-estimation-dev/output/finesub_mpc_translation_<时间>/monitor.mp4`；
- 原始双目录像：同目录下的 `raw_stereo.mp4`（未标注、全帧率相机原始并排画面，供离线回放/数据集）；
- 视觉日志：同目录下的 `vision.log`；
- 误差图日志：同目录下的 `error_plot.log`；
- MPC trace：`MPC/calibration_logs/experimental_auto_<时间>.jsonl`。

如果相机窗口没有出现，先查看 `vision.log`；如果 30 秒内没有 JSONL，命令会自动停止，通常表示 UDP 5600 视频流没有到达。
