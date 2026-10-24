"""③ 轻量慢系统：趋势、短期预测、可继续性判断。

类比 Kahneman 的 System 2 —— 慢、审慎 —— 但受"轻量"约束：
**不训练神经网络、不引入深度模型**，全部使用可解释的在线统计方法。
"""

from __future__ import annotations

from .capacity import CapacityAssessor
from .predictor import StressPredictor
from .trend import cusum, robust_slope

__all__ = ["CapacityAssessor", "StressPredictor", "cusum", "robust_slope"]
