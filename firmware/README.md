# V4Pro1 下位机固件

本目录保留当前固件、板级支持包和编译所需的 CMSIS-DSP、ETL 文件。

## 编译

```bash
cmake --list-presets
cmake --preset linux-gcc-robomaster-a
cmake --build --preset linux-gcc-robomaster-a
```

烧录前需要确认所选开发板、推进器顺序与方向、传感器方向、通信协议和限幅参数。

