# 四路环视 BEV 与 RK3566 迁移说明

这次改造把四路视频输入、环视合成和现有 PC 车载大屏接在了一起。它采用了开源项目中已经验证过的工程分层：相机帧先经过一次性映射准备，运行时只做 resize、缓存 remap 和融合。当前实现是 OpenCV CPU 版本，目的是先把输入、同步、画面和性能边界跑通，再替换成实测标定和 RKNN/硬件加速实现。

## 已接入的运行链路

```text
front/left/right/rear.mp4
        │
        ▼
  SurroundVideoSource
        │ 四路同步读取
        ▼
  SurroundBEVComposer
        │ 缓存 planar remap + feather 融合
        ▼
  WebSocket: video_jpeg + surround_bev_jpeg
        │
        ▼
  PC ADAS HMI（3D ADAS 驾驶主画面 + 前视/导航地图/BEV 辅助区）
```

PC 上可以这样启动：

```powershell
python tools/adas_hmi_sim/run.py `
  --mode surround `
  --surround-dir C:\path\to\four_cameras `
  --bev-config tools/adas_hmi_sim/configs/surround_bev.json `
  --model C:\path\to\yolo11n_320x192.rknn `
  --surround-detector-input-size 320x192 `
  --port 8767 `
  --no-browser
```

有真实鱼眼内参、外参后，先在 PC 生成上游 LUT，再把整个 LUT 目录拷到板上：

```powershell
python tools/build_calibrated_surround_lut.py `
  --calibration-yaml C:\path\to\calibration.yaml `
  --output outputs\surround_lut_320 `
  --source-size 1920x1536 `
  --output-size 320x320 `
  --projection bowl

python tools/adas_hmi_sim/run.py `
  --mode surround `
  --surround-dir C:\path\to\four_cameras `
  --bev-lut-dir outputs\surround_lut_320 `
  --port 8767 `
  --no-browser
```

`--bev-lut-dir` 会优先于近似 JSON 配置；它要求输入视频尺寸和生成 LUT 时的相机标定尺寸一致，不能拿普通 16:9 道路视频套用鱼眼标定。

目录需要包含 `front.mp4`、`left.mp4`、`right.mp4` 和 `rear.mp4`；后视也接受文件名 `back.mp4`。不传 `--model` 时，`surround` 模式只负责真实四路画面和 BEV，检测列表为空，因而不会把“没有接入 YOLO/RKNN”伪装成检测结果。传入 `.rknn` 后只在 front stream 上按目标频率做检测，并把检测结果送入同一个 Scene Builder；矩形模型必须显式传 `--surround-detector-input-size`，例如 `320x192`。`pc-system` 模式仍可用 mock 检测和四路模拟传输验证完整 HMI 协议。

## RK3566 实测结果

测试输入是板上已有的四个视频文件：

```text
/home/cat/adas-edge-rect/videos/ch0_front.mp4
/home/cat/adas-edge-rect/videos/ch1_left.mp4
/home/cat/adas-edge-rect/videos/ch2_right.mp4
/home/cat/adas-edge-rect/videos/ch3_back.mp4
```

测试程序运行在 LubanCat-1 RK3566（aarch64、4 核、4 GiB、OpenCV Python）上，四路解码和 BEV 合成均在 CPU 完成。结果是：

这四个文件是 1280×720 的普通道路视频，不是同一套经过鱼眼标定、时间同步的车载四摄像头素材；抽帧可见 front/left 是夕阳道路，right/rear 是夜间道路。因此它们可以证明板端解码、合成和负载，但不能证明实车拼接几何质量。要得到小米/特斯拉式全景，必须换成同一车辆的四路鱼眼流并提供对应标定参数。

| 输入 resize | BEV 输出 | 合成速率 | 单帧解码 | 单帧合成 |
| ---: | ---: | ---: | ---: | ---: |
| 640×360 | 640×640 | 约 2.27 FPS | — | — |
| 320×180 | 320×320 | 约 7.37 FPS | 39.9 ms | 95.3 ms |
| 256×144 | 256×256 | 约 8.67 FPS | 39.1 ms | 75.7 ms |
| 224×126 | 224×224 | 约 9.68 FPS | 38.8 ms | 64.0 ms |

把 front 的 `yolo11n_rk3566_int8_split_320x192.rknn` 同时接入后，板端 224×224 BEV + front RKNN 的 30 帧串行测试为总计约 **7.68 FPS**，BEV 合成约 61.7 ms/帧，NPU 推理约 29.3 ms/帧，实际得到 140 个检测结果。这个数字才是“画面和检测一起跑”的参考；它仍然不包含 FPGA PCIe DMA 和最终 HDMI 合成。

因此，**RK3566 能跑这条四路环视画面链路，但当前纯 CPU 版本不能承诺 30 FPS**。实际产品应把 BEV 更新目标定在约 8–10 Hz，浏览器/原生 HMI 可以用最近一帧保持 30 Hz 显示。板上推荐先用 `surround_bev_rk3566_224.json` 验证链路；如果画质和带宽允许，改用 256 配置。

## 边界和下一步

当前默认配置明确标记为 `APPROXIMATE_PLANAR`。它是用于验证工程链路的近似平面映射，不是四个鱼眼相机的实测内参、外参、畸变校正或完整 AVM 标定，所以不能把画面当成小米/特斯拉量产级全景效果。开源参考为 [Open-3D-Surround-View](https://github.com/nick8592/Open-3D-Surround-View)；本项目已把它的鱼眼投影、2D/Bowl LUT 和权重融合移植为 `calibrated_surround.py`，MIT 许可全文随源码保存。更重的 [Tesla-BevCam-Clone](https://github.com/Ga0512/Tesla-BevCam-Clone) 还包含 YOLO、深度和车道网络，原始 Python 依赖在当前 RK3566 环境没有完整安装，不能直接移植成实时板端链路。

后续真正接入 FPGA→PCIe→RK3566 时，只需要让 PCIe/DMA 接收层输出 `Mapping[str, np.ndarray]`，复用 `SurroundBEVComposer.compose()`；前端协议和 BEV 卡片不需要重写。检测器也应作为单独 stage 接入，不能从四路画面是否显示推断出 YOLO 已经在板端运行。
