"""③ 轻量慢系统 · 短期紧张度预测。

实现需求 L-2 ~ L-3、L-6。

模型（架构文档 §5.3.2）：**均值回复 + 趋势外推**

    Ŝ(t+H) = S_t + slope·H + (μ_base − S_t)·(1 − exp(−H/τ_rev))

物理解释：人总会平复（均值回复项），但当前趋势也会延续（趋势项）。
两者线性叠加，参数可在线自适应，全过程可解释、无需训练。

═══════════════════════════════════════════════════════════════════════
诚实性声明 [V]（不得删除）
    本模型**未经真实数据回测**。``Forecast.validated`` 恒为 False。
    不要在任何对外表述中给出本模块的"预测准确率"。
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from ..contracts import Forecast
from .trend import robust_slope


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


class StressPredictor:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.tau_rev: float = float(config.tau_rev)
        #: 预测残差的在线估计（稳健标准差代理）
        self.sigma_resid: float = 0.05
        self._decay: float = 0.1  # 残差 EWMA 系数

    # ------------------------------------------------------------------
    def forecast(
        self,
        times: Sequence[float],
        values: Sequence[float],
        baseline: float,
        horizon: float | None = None,
    ) -> Forecast:
        cfg = self.config
        H = float(horizon if horizon is not None else cfg.forecast_horizon)

        if not values:
            return Forecast(
                horizon_s=H,
                point=_clip01(baseline),
                lower=_clip01(baseline),
                upper=_clip01(baseline),
                slope=0.0,
                baseline=baseline,
                method="ou_trend_v1",
                validated=False,
            )

        s_t = float(values[-1])

        # 只取趋势窗口内的数据
        window = cfg.trend_window
        t_end = times[-1]
        win_t = [t for t in times if t >= t_end - window]
        win_v = list(values)[-len(win_t) :] if win_t else list(values)
        slope_per_min = robust_slope(win_t, win_v, per_minute=True)
        slope_per_s = slope_per_min / 60.0

        tau = max(self.tau_rev, 1e-6)
        mean_reversion = (baseline - s_t) * (1.0 - math.exp(-H / tau))
        trend_term = slope_per_s * H

        point = _clip01(s_t + trend_term + mean_reversion)

        # 不确定性区间：随预测视界按 sqrt 扩张
        spread = 1.96 * self.sigma_resid * math.sqrt(max(H, 1.0) / max(cfg.forecast_horizon, 1.0))
        return Forecast(
            horizon_s=H,
            point=point,
            lower=_clip01(point - spread),
            upper=_clip01(point + spread),
            slope=slope_per_min,
            baseline=float(baseline),
            method="ou_trend_v1",
            validated=False,
        )

    # ------------------------------------------------------------------
    def update_residual(self, predicted: float, actual: float) -> None:
        """用真实观测更新残差估计 —— 在线自适应，无需重训练。"""
        residual = abs(float(actual) - float(predicted))
        self.sigma_resid = (1.0 - self._decay) * self.sigma_resid + self._decay * residual
