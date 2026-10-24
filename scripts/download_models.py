"""下载感知层所需的真实开源模型权重（多镜像源自动回退）。

需要的模型：

  * ``face_detection_yunet_2023mar.onnx`` —— OpenCV DNN 人脸检测（YuNet，~230KB）
    这是 OpenCV 5.x 环境下**唯一可用**的真实人脸检测路径
    （cv2 5.0 已移除 Haar 的 CascadeClassifier）。
  * ``face_landmarker.task`` —— MediaPipe FaceLandmarker（~3.7MB，468 关键点）
    特征最丰富，但源在 Google Storage，部分网络环境下无法直连。

用法::

    python scripts/download_models.py

脚本会依次尝试多个镜像源，并在数据明显过小（例如 Git LFS 指针、错误页）时
自动换源。全部失败时会打印手动下载指引，**不会**抛异常。

注意：模型缺失不会让系统崩溃 —— 面部子模态会自动继续降级
（见 docs/02-architecture.md §7），只是精度与置信度下降。
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: 每个模型的可接受最小字节数；小于它视为拿到的是 LFS 指针/错误页
MIN_BYTES: dict[str, int] = {
    "face_detection_yunet_2023mar.onnx": 50_000,
    "face_landmarker.task": 1_000_000,
}

MODELS: dict[str, list[str]] = {
    "face_detection_yunet_2023mar.onnx": [
        # GitHub LFS 的真实内容由 media.githubusercontent.com 提供
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://raw.githubusercontent.com/opencv/opencv_zoo/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://ghproxy.net/https://raw.githubusercontent.com/opencv/opencv_zoo/"
        "main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://gh-proxy.com/https://raw.githubusercontent.com/opencv/opencv_zoo/"
        "main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ],
    "face_landmarker.task": [
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task",
        # `[F]` 本机实测（2026，国内网络）：storage.googleapis.com 直连超时
        # （WinError 10060），且未找到可靠的国内镜像。
        # 若你有可用网络，请手动下载后把文件放进 models/ 目录。
    ],
}


def download(name: str, urls: list[str], target_dir: Path) -> bool:
    target = target_dir / name
    minimum = MIN_BYTES.get(name, 1)
    if target.exists() and target.stat().st_size >= minimum:
        print(f"[跳过] {name} 已存在（{target.stat().st_size} 字节）")
        return True

    target_dir.mkdir(parents=True, exist_ok=True)
    for url in urls:
        print(f"[尝试] {url}")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                data = response.read()
        except Exception as exc:  # noqa: BLE001
            print(f"       失败：{exc!r}")
            continue

        if len(data) < minimum:
            print(
                f"       数据过小（{len(data)} 字节 < {minimum}），"
                "很可能是 Git LFS 指针或错误页，换下一个源"
            )
            continue

        target.write_bytes(data)
        print(f"[完成] {name}（{len(data)} 字节）-> {target}")
        return True

    print(f"[失败] {name}：所有镜像源都不可用。")
    return False


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

    target_dir = Path(os.environ.get("EC_MODEL_DIR", str(ROOT / "models")))
    print(f"模型目录：{target_dir}\n")

    results = {name: download(name, urls, target_dir) for name, urls in MODELS.items()}

    print("\n" + "-" * 70)
    for name, ok in results.items():
        print(f"{'[可用]' if ok else '[缺失]'} {name}")

    missing = [n for n, ok in results.items() if not ok]
    if missing:
        print(
            "\n手动下载指引：\n"
            "  1) 用浏览器/其他网络环境下载上面列出的 URL；\n"
            "  2) 把文件放进 models/ 目录（文件名保持不变）；\n"
            "  3) 或用环境变量 EC_MODEL_DIR 指向你的模型目录后重跑。\n"
            "  缺失的模型只会让对应的面部后端降级，不影响系统整体运行。"
        )
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
