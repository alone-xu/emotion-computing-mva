"""③ 轻量慢系统 · 可继续性判断（capacity）。

实现需求 L-4 ~ L-5：综合**预测值**、**高位持续时间**、**恢复力**，
输出 OK / WATCH / PAUSE，并给出人类可读的理由。

设计立场（需求文档 §3）：
    受试者本人也是使用者，系统必须"不令人不适"。
    因此默认偏保守 —— 宁可漏报，不要骚扰。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..contracts import CapacityAssessment, Forecast

#: 斜率归一化参考值（紧张度/分钟）—— 超过它即视为"快速恶化"
SLOPE_REFERENCE: float = 0.30


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def high_duration(times: Sequence[float], values: Sequence[float], theta: float) -> float:
    """当前连续处于阈值之上的时长（秒）；若当前不在高位则返回 0。"""
    if not values or values[-1] < theta:
        return 0.0
    i = len(values) - 1
    while i > 0 and values[i - 1] >= theta:
        i -= 1
    return float(times[-1] - times[i])


class CapacityAssessor:
    def __init__(self, config: Any) -> None:
        self.config = config

    # ------------------------------------------------------------------
    def assess(
        self,
        times: Sequence[float],
        values: Sequence[float],
        forecast: Forecast,
        resilience: float = 0.5,
        contributions: Mapping[str, float] | None = None,
    ) -> CapacityAssessment:
        cfg = self.config
        s_t = float(values[-1]) if values else 0.0
        resilience = _clip01(resilience)

        dur_high = high_duration(times, values, cfg.theta_hi)
        dur_mid = high_duration(times, values, cfg.theta_mid)

        # 可继续性得分：预测越低、趋势越平、恢复力越强 → 分越高
        forward_slope = max(forecast.slope, 0.0)
        score = _clip01(
            1.0
            - 0.60 * forecast.point
            - 0.20 * min(forward_slope / SLOPE_REFERENCE, 1.0)
            - 0.20 * (1.0 - resilience)
        )

        # ---- 状态机（含迟滞：上下阈值分离）----
        theta_hi = cfg.theta_hi
        theta_mid = cfg.theta_mid

        if forecast.point >= theta_hi or (s_t >= theta_hi and dur_high >= cfg.t_hi):
            state = "PAUSE"
        elif forecast.point >= theta_mid or (s_t >= theta_mid and forecast.slope > 0):
            state = "WATCH"
        else:
            state = "OK"

        # ---- 可解释理由 ----
        reasons: list[str] = []
        reasons.append(
            f"预测 {forecast.horizon_s:.0f}s 后紧张度 {forecast.point:.2f}"
            f"（阈值 WATCH {theta_mid:.2f} / PAUSE {theta_hi:.2f}）"
        )
        if dur_high > 0:
            reasons.append(f"当前已连续 {dur_high:.0f}s 高于 {theta_hi:.2f}")
        elif dur_mid > 0:
            reasons.append(f"当前已连续 {dur_mid:.0f}s 高于 {theta_mid:.2f}")
        reasons.append(
            f"趋势 {forecast.slope:+.2f}/分钟"
            + ("（上升）" if forecast.slope > 0 else "（下降或平稳）")
        )
        # 需求 L-5：说明当前趋势主要由哪个模态驱动。
        # 用"最近一个样本的模态贡献占比"作代理 —— 慢系统的输入是标量序列，
        # 拿不到窗口内逐帧的贡献分解，这里如实标注"代理"，不假装是严格归因。
        if contributions:
            dominant, share = max(contributions.items(), key=lambda kv: float(kv[1]))
            if float(share) > 0.0:
                reasons.append(
                    f"最近样本的紧张度主要由 {dominant} 驱动"
                    f"（贡献占比 {float(share) * 100:.0f}%，以当前样本为代理）"
                )
        reasons.append(f"恢复力估计 {resilience:.2f}（越高越能自我平复）")
        if not forecast.validated:
            reasons.append("⚠ 预测模型未经真实数据校准（validated=False）")

        return CapacityAssessment(
            state=state,
            score=score,
            forecast=forecast,
            resilience=resilience,
            reasons=tuple(reasons),
        )
