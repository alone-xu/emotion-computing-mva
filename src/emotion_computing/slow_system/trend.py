"""③ 轻量慢系统 · 趋势建模与变点检测。

实现需求 L-1。

为什么用 Theil–Sen 而不是最小二乘：紧张度序列常含离群尖峰，
最小二乘会被单点带偏；成对斜率中位数对离群稳健。
点数超过阈值时先做均匀抽稀，保证 O(n²) 可控。
"""

from __future__ import annotations

import math
from typing import Sequence


def _thin(
    times: Sequence[float], values: Sequence[float], max_points: int = 60
) -> tuple[list[float], list[float]]:
    n = len(times)
    if n <= max_points:
        return list(times), list(values)
    step = (n - 1) / (max_points - 1)
    idx = [int(round(i * step)) for i in range(max_points)]
    return [times[i] for i in idx], [values[i] for i in idx]


def robust_slope(
    times: Sequence[float], values: Sequence[float], per_minute: bool = True
) -> float:
    """稳健趋势斜率（默认单位：紧张度/分钟）。"""
    if len(times) < 2 or len(values) < 2:
        return 0.0
    ts, vs = _thin(times, values)
    slopes: list[float] = []
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            dt = ts[j] - ts[i]
            if dt > 1e-6:
                slopes.append((vs[j] - vs[i]) / dt)
    if not slopes:
        return 0.0
    slopes.sort()
    m = len(slopes)
    median = slopes[m // 2] if m % 2 == 1 else 0.5 * (slopes[m // 2 - 1] + slopes[m // 2])
    return median * 60.0 if per_minute else median


def _std(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def cusum(
    times: Sequence[float],
    values: Sequence[float],
    k: float = 0.5,
    h: float = 5.0,
    target: float | None = None,
    sd_floor: float = 0.02,
) -> dict:
    """CUSUM 变点检测：区分"正常波动"与"持续性偏移"。

    ``k`` 为松弛量（以 σ 为单位），``h`` 为报警阈值（以 σ 为单位）。
    返回首个变点的时间、方向（+1 上升 / -1 下降）与当前统计量。
    """
    n = len(values)
    if n < 4:
        return {"change_point": None, "direction": 0, "stat": 0.0, "target": target}

    # target 取"会话初期基线"而不是全局均值：
    # 若用全局均值，当序列是「前半段低、后半段高」时，逐点累积会先在
    # 前半段攒出一个向下的偏移，把"一开始的低位"误报成一次下降变点。
    # 本系统关心的是「相对基线的持续性偏移」，因此基线取前 25% 样本。
    if target is None:
        n_base = max(4, int(0.25 * n))
        target = sum(values[:n_base]) / n_base
    sd = max(_std(values), sd_floor)

    s_hi = 0.0
    s_lo = 0.0
    change_point: float | None = None
    direction = 0

    for i, v in enumerate(values):
        s_hi = max(0.0, s_hi + (v - target) - k * sd)
        s_lo = max(0.0, s_lo - (v - target) - k * sd)
        if change_point is None:
            if s_hi > h * sd:
                change_point, direction = times[i], 1
            elif s_lo > h * sd:
                change_point, direction = times[i], -1

    return {
        "change_point": change_point,
        "direction": direction,
        "stat": max(s_hi, s_lo) / sd,
        "target": float(target),
    }
