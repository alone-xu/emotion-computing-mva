"""① 感知层：把原始信号变成统一的模态观测。

对外只暴露 ModalityObservation（见 contracts.py）。
所有真实后端都遵循"首选 → 降级 → 合成兜底"的装配链，永不抛异常给上层。
"""

from __future__ import annotations

from .base import ModalitySensor, build_sensor, warn_once
from .fusion import FusionEngine

__all__ = ["ModalitySensor", "build_sensor", "warn_once", "FusionEngine"]
