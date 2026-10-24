"""合成信号源：演示与测试用的**确定性**紧张度生成器。

为什么需要它（docs/01-requirements.md §5.1.5）：
  * 它是降级矩阵的兜底，保证任何机器上端到端都能跑通；
  * 它是单元测试的基础设施 —— 我们**知道真值曲线**，
    因此可以断言"紧张度上升时预测也应该上升"这类性质。

它生成的不是真实情绪数据，**不可用于任何效果评估**。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..contracts import ModalityObservation
from .base import ModalitySensor

#: 默认紧张度真值曲线的控制点 (秒, 紧张度)。形状：静息 → 上升 → 峰值 → 部分恢复 → 二次上升。
DEFAULT_CONTROL_POINTS: tuple[tuple[float, float], ...] = (
    (0, 0.18),
    (30, 0.20),
    (60, 0.34),
    (120, 0.60),
    (180, 0.86),
    (215, 0.80),
    (255, 0.46),
    (300, 0.38),
    (360, 0.56),
    (420, 0.72),
    (470, 0.44),
    (540, 0.30),
)

#: 各模态的响应参数 (gain, offset)，模拟"不同模态敏感度不同"。
_MODALITY_RESPONSE: dict[str, tuple[float, float]] = {
    "face": (0.90, 0.04),
    "speech": (1.00, 0.02),
    "text": (0.85, 0.06),
}


class SyntheticStressDriver:
    """一条可控的紧张度真值曲线 + 模态化的观测噪声。"""

    def __init__(
        self,
        control_points: Sequence[tuple[float, float]] | None = None,
        seed: int = 7,
        noise: float = 0.02,
        dropout: float = 0.0,
    ) -> None:
        points = tuple(control_points) if control_points else DEFAULT_CONTROL_POINTS
        self._xs = np.array([p[0] for p in points], dtype=float)
        self._ys = np.array([p[1] for p in points], dtype=float)
        self.rng = np.random.default_rng(seed)
        self.noise = float(noise)
        #: 每个模态被"随机丢失"的概率，用于验证融合层的缺失重分配
        self.dropout = float(dropout)
        self._counter: dict[str, int] = {}

    # -- 真值 -------------------------------------------------------------
    def truth(self, t: float) -> float:
        """该时刻的紧张度真值（无噪声），[0, 1]。"""
        return float(np.clip(np.interp(t, self._xs, self._ys), 0.0, 1.0))

    def duration(self) -> float:
        return float(self._xs[-1])

    # -- 观测 -------------------------------------------------------------
    def modality_score(self, modality: str, t: float) -> float:
        """某一模态在该时刻"观测到"的紧张度（真值 + 模态偏置 + 噪声）。"""
        gain, offset = _MODALITY_RESPONSE.get(modality, (0.9, 0.05))
        noisy = gain * self.truth(t) + offset + float(self.rng.normal(0.0, self.noise))
        return float(np.clip(noisy, 0.0, 1.0))

    def is_missing(self, modality: str, t: float) -> bool:
        if self.dropout <= 0.0:
            return False
        return bool(self.rng.random() < self.dropout)


class SyntheticSensor(ModalitySensor):
    """把 SyntheticStressDriver 包装成标准模态传感器（兜底后端）。"""

    def __init__(self, name: str, driver: Any = None) -> None:
        super().__init__()
        self.name = name
        self.backend = "synthetic"
        self.driver: SyntheticStressDriver = (
            driver if isinstance(driver, SyntheticStressDriver)
            else SyntheticStressDriver()
        )
        if not isinstance(driver, SyntheticStressDriver):
            self._fallback_reason = "未提供合成驱动，使用默认曲线"

    def sense(self, payload: dict[str, Any]) -> ModalityObservation:
        t = float(payload.get("t", 0.0))
        if self.driver.is_missing(self.name, t):
            return self._missing(t, "synthetic_dropout")
        return ModalityObservation(
            modality=self.name,
            timestamp=t,
            score=self.driver.modality_score(self.name, t),
            confidence=0.5,  # 合成数据置信度人为压到中位
            available=True,
            backend="synthetic",
            features={"truth": self.driver.truth(t)},
            note="合成信号，非真实测量",
        )
