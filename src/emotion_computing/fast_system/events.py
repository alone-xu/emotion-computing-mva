"""② 快系统 · 事件检测。

实现需求 F-3 ~ F-5：只依赖滑动窗口内的极短历史。

事件类型：
  * ``spike``           —— 短时间内的显著跳变（幅度 + 显著性双重判据）
  * ``sustained_high``  —— 持续处于高位（同一次高位只报一次，回落后重置）
  * ``rise`` / ``fall`` —— 趋势方向显著但幅度未达 spike
"""

from __future__ import annotations

from collections import deque
from typing import Any

from ..contracts import StressEvent, StressSample


class EventDetector:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.window: float = 30.0
        self._history: deque[tuple[float, float]] = deque()
        self._high_since: float | None = None
        self._sustained_fired: bool = False

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._history.clear()
        self._high_since = None
        self._sustained_fired = False

    # ------------------------------------------------------------------
    def _value_at(self, timestamp: float, lookback: float) -> float | None:
        """取"约 lookback 秒之前"的样本值（滑动窗口内最近的点）。"""
        target = timestamp - lookback
        best: tuple[float, float] | None = None
        for ts, val in self._history:
            if ts <= target:
                best = (ts, val)
            else:
                break
        return best[1] if best else None

    @staticmethod
    def _recent_std(values: list[float]) -> float:
        if len(values) < 3:
            return 0.0
        mean = sum(values) / len(values)
        return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5

    def _dominant(self, sample: StressSample) -> str:
        if not sample.contributions:
            return "unknown"
        return max(sample.contributions.items(), key=lambda kv: kv[1])[0]

    # ------------------------------------------------------------------
    def update(self, sample: StressSample) -> list[StressEvent]:
        cfg = self.config
        events: list[StressEvent] = []
        t = sample.timestamp
        s = sample.smoothed

        self._history.append((t, s))
        while self._history and t - self._history[0][0] > self.window:
            self._history.popleft()

        # ---- spike / rise / fall ----
        baseline = self._value_at(t, 3.0)
        if baseline is not None:
            delta = s - baseline
            recent = [v for _, v in self._history]
            std = self._recent_std(recent)
            significant = std > 1e-6 and abs(delta) > cfg.spike_sigma_k * std

            if abs(delta) >= cfg.theta_spike and significant:
                events.append(
                    StressEvent(
                        timestamp=t,
                        kind="spike" if delta > 0 else "fall",
                        magnitude=float(delta),
                        dominant_modality=self._dominant(sample),
                        detail=(
                            f"3s 内变化 {delta:+.3f}（阈值 {cfg.theta_spike:.2f}，"
                            f"显著性 {abs(delta) / max(std, 1e-9):.1f}σ）"
                        ),
                    )
                )
            elif abs(delta) >= cfg.theta_spike * 0.5:
                events.append(
                    StressEvent(
                        timestamp=t,
                        kind="rise" if delta > 0 else "fall",
                        magnitude=float(delta),
                        dominant_modality=self._dominant(sample),
                        detail=f"3s 内温和变化 {delta:+.3f}",
                    )
                )

        # ---- sustained_high ----
        if s >= cfg.theta_hi:
            if self._high_since is None:
                self._high_since = t
            elif not self._sustained_fired and (t - self._high_since) >= cfg.t_hi:
                events.append(
                    StressEvent(
                        timestamp=t,
                        kind="sustained_high",
                        magnitude=s,
                        dominant_modality=self._dominant(sample),
                        detail=f"已连续 {t - self._high_since:.0f}s 高于阈值 {cfg.theta_hi:.2f}",
                    )
                )
                self._sustained_fired = True
        else:
            self._high_since = None
            self._sustained_fired = False

        return events
