# models/

模型权重**不随仓库分发**（见根目录 `.gitignore`），需要时用脚本下载：

```powershell
python scripts/download_models.py
```

脚本会从多个镜像源自动回退，写入本目录。

## 当前需要的权重

| 文件 | 用途 | 大小 | 缺失时的行为 |
|---|---|---|---|
| `face_detection_yunet_2023mar.onnx` | OpenCV DNN 人脸检测（5 关键点） | ~230 KB | 自动降级到 Haar，再降级到 synthetic |
| `face_landmarker.task` | MediaPipe FaceLandmarker（468 关键点，可选） | ~3 MB | 自动降级到 YuNet |

缺失任何一个都不会让系统崩溃——感知层的降级链会接管，并在
`ModalityObservation.backend` 字段里如实标注本帧实际生效的实现。

## 为什么手动放

`storage.googleapis.com`（MediaPipe 官方模型源）在国内网络直连经常超时，
因此不建议在 `pip install` 阶段自动拉取，而是提供一个可重试、多源回退的显式脚本。
