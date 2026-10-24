"""② 快系统：低延迟的即时紧张度估计与事件检测。

类比 Kahneman 的 System 1 —— 快、直觉、低延迟、不做深度推理。
只依赖当前与极短历史（滑动窗口），绝不阻塞在慢系统上。
"""

from __future__ import annotations

from .estimator import FastEstimator
from .events import EventDetector

__all__ = ["FastEstimator", "EventDetector"]
