"""② 快系统 · 即时紧张度估计（EWMA 平滑）。

实现需求 F-1 ~ F-3、F-6。

平滑系数按真实 Δt 用 ``α = 1 - exp(-Δt/τ)`` 计算，
因此在不同帧率（10fps 摄像头 vs 1Hz 离线采样）下行为一致 —— 这是可复现性的前提。
"""

from __future__ import annotations

import math
from typing import Any

from ..contracts import PerceptionFrame, StressSample


class FastEstimator:
    """把 PerceptionFrame 变成平滑后的实时紧张度样本。"""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.tau: float = float(config.tau_fast)
        self.value: float | None = None
        self._prev_t: float | None = None
        self._confidence: float = 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def alpha_for(dt: float, tau: float) -> float:
        """时间常数形式的时间感知平滑系数。"""
        if tau <= 0:
            return 1.0
        return float(1.0 - math.exp(-max(dt, 0.0) / tau))

    def reset(self) -> None:
        self.value = None
        self._prev_t = None
        self._confidence = 0.0

    # ------------------------------------------------------------------
    def update(self, frame: PerceptionFrame) -> StressSample:
        dt = 1.0
        if self._prev_t is not None:
            dt = max(frame.timestamp - self._prev_t, 1e-6)

        alpha = self.alpha_for(dt, self.tau)
        raw = float(frame.normalized)

        if self.value is None:
            # 首个样本不做平滑，避免从 0 缓慢爬升
            self.value = raw
        else:
            self.value = alpha * raw + (1.0 - alpha) * self.value

        self._prev_t = frame.timestamp
        self._confidence = self._frame_confidence(frame)

        return StressSample(
            timestamp=frame.timestamp,
            raw=raw,
            smoothed=float(min(1.0, max(0.0, self.value))),
            confidence=self._confidence,
            contributions=dict(frame.contributions),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _frame_confidence(frame: PerceptionFrame) -> float:
        """按模态贡献占比加权平均置信度；无可用模态时为 0。"""
        available = [o for o in frame.observations if o.available]
        if not available:
            return 0.0
        weights = frame.contributions or {}
        total_w = sum(weights.get(o.modality, 0.0) for o in available)
        if total_w <= 0:
            return float(sum(o.confidence for o in available) / len(available))
        return float(
            sum(weights.get(o.modality, 0.0) * o.confidence for o in available) / total_w
        )
