# FinsROV 双目视觉代码

本目录包含实机双目视觉流程：鱼目标检测、双目深度估计、目标跟踪、时序滤波、相机标定和带时间戳的数据输出。

运行核心来自 `/data/Zhouyuheng_workspace/tracking_depth/depth-estimation-dev/`，复制时未修改代码和配置内容。

不包含仿真和 Unity 相机服务代码。

## 坐标系

视觉输出使用 OpenCV 相机坐标：`[X 向右, Y 向下, Z 向前]`，单位为米。控制程序会使用标定外参转换到潜器 FRD 机体系。

## 安装与测试

```bash
uv sync --dev
uv run --with pytest pytest -q
```

## 模型文件

模型路径以原始 `src/depth_estimation/config.yaml` 为准。Fast-FoundationStereo 运行源码已归类到 `third_party/Fast-FoundationStereo/`，模型权重、训练数据和实验录像不在本仓库重复保存。

## 实机运行

接收 5600 端口的左右并排双目画面，并保存三维结果、每帧时间戳和录像：

```bash
uv run depth-demo-video \
  --udp-sbs-port 5600 \
  --result-jsonl runtime/pipeline_results.jsonl \
  --frame-jsonl runtime/frame_timestamps.jsonl \
  --record-raw runtime/stereo_raw.mp4 \
  --record-monitor runtime/stereo_monitor.mp4
```

使用两个相机设备：

```bash
uv run depth-demo-video --left 0 --right 1 \
  --result-jsonl runtime/pipeline_results.jsonl
```

## 双目标定

```bash
uv run stereo-capture-calibration --help
uv run stereo-calibrate --help
uv run stereo-preview-udp --help
```

更换相机、镜头、分辨率、焦距、水下壳体或双目基线后，需要重新标定。
