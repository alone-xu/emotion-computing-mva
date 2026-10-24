"""探测本机可用的真实感知后端，输出一张「这台机器到底能跑什么」的表。

用法::

    python scripts/check_environment.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CHECKS: list[tuple[str, str, str]] = [
    ("numpy", "数值基础（必需）", "numpy"),
    ("opencv", "面部：YuNet(DNN) / Haar 降级", "cv2"),
    ("mediapipe", "面部：FaceLandmarker 首选（Tasks API）", "mediapipe"),
    ("librosa", "语音：声学特征（真实可用）", "librosa"),
    ("sounddevice", "语音：麦克风采集（可选）", "sounddevice"),
    ("transformers", "语义：情感模型（需设 EC_ENABLE_TRANSFORMERS=1）", "transformers"),
    ("scipy", "科学计算（librosa 的间接依赖）", "scipy"),
]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")
    print("-" * 78)
    missing: list[str] = []
    for label, desc, module in CHECKS:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "?")
            print(f"[ 可用 ] {label:<15} {version:<12} {desc}")
        except Exception as exc:  # noqa: BLE001
            missing.append(label)
            print(f"[ 缺失 ] {label:<15} {'-':<12} {desc}   ({exc.__class__.__name__})")
    print("-" * 78)
    if missing:
        print("缺失的依赖会触发自动降级（见 docs/02-architecture.md §7）。")
        print("安装：python -m pip install " + " ".join(missing))
    else:
        print("全部后端可用。")

    # 实际装配一次，看看真实生效的后端组合
    print("\n实际装配结果：")
    from emotion_computing.config import Config
    from emotion_computing.perception.base import build_sensor

    cfg = Config()
    for name in ("face", "speech", "text"):
        sensor = build_sensor(name, cfg)
        print(f"  {name:<8} -> {sensor.backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
