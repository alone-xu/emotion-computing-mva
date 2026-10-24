"""③ 轻量慢系统 · 趋势、预测、可继续性测试。"""

from __future__ import annotations

import pytest

from emotion_computing.config import Config
from emotion_computing.contracts import Forecast
from emotion_computing.slow_system import CapacityAssessor, StressPredictor
from emotion_computing.slow_system.capacity import high_duration
from emotion_computing.slow_system.trend import cusum, robust_slope


def _flat_forecast(point: float = 0.6, slope: float = 0.2) -> Forecast:
    return Forecast(
        horizon_s=60.0, point=point, lower=point - 0.05, upper=point + 0.05,
        slope=slope, baseline=0.3, method="test", validated=False,
    )


def test_assess_attributes_dominant_modality() -> None:
    """需求 L-5：有贡献分解时，理由里要说明是谁把紧张度推上去的。"""
    assessor = CapacityAssessor(Config())
    times = [0.0, 1.0, 2.0]
    values = [0.6, 0.6, 0.6]

    with_contrib = assessor.assess(
        times, values, _flat_forecast(), 0.5,
        contributions={"face": 0.7, "speech": 0.2, "text": 0.1},
    )
    assert any("face" in r and "驱动" in r for r in with_contrib.reasons), with_contrib.reasons

    without = assessor.assess(times, values, _flat_forecast(), 0.5)
    assert not any("驱动" in r for r in without.reasons), "没给贡献就不该编造归因"


# --------------------------------------------------------------------------
# 趋势
# --------------------------------------------------------------------------


def test_robust_slope_positive_on_rising() -> None:
    times = [float(i) for i in range(30)]
    values = [0.20 + 0.010 * i for i in range(30)]
    assert robust_slope(times, values) > 0


def test_robust_slope_negative_on_falling() -> None:
    times = [float(i) for i in range(30)]
    values = [0.80 - 0.010 * i for i in range(30)]
    assert robust_slope(times, values) < 0


def test_robust_slope_ignores_single_outlier() -> None:
    """稳健性：一个离群尖峰不应反转整体趋势判断。"""
    times = [float(i) for i in range(30)]
    values = [0.20 + 0.010 * i for i in range(30)]
    clean = robust_slope(times, values)
    values[15] = 0.95  # 注入离群点
    dirty = robust_slope(times, values)
    assert dirty > 0, "注入离群点后趋势方向不应反转"
    assert abs(dirty - clean) < abs(clean), "中位数斜率不应被单点带偏太多"


def test_cusum_finds_upward_change_point() -> None:
    times = [float(i) for i in range(80)]
    values = [0.20] * 40 + [0.70] * 40
    result = cusum(times, values)
    assert result["change_point"] is not None
    assert result["direction"] == 1
    assert 35 <= result["change_point"] <= 50


def test_cusum_no_change_on_flat() -> None:
    times = [float(i) for i in range(60)]
    values = [0.4] * 60
    assert cusum(times, values)["change_point"] is None


# --------------------------------------------------------------------------
# 预测
# --------------------------------------------------------------------------


def test_forecast_is_marked_unvalidated() -> None:
    """需求 L-6：未经真实数据校准必须显式标记。"""
    predictor = StressPredictor(Config())
    forecast = predictor.forecast([0.0, 1.0], [0.3, 0.4], baseline=0.3)
    assert forecast.validated is False


def test_forecast_follows_upward_trend_short_horizon() -> None:
    cfg = Config(trend_window=60.0, forecast_horizon=30.0, tau_rev=600.0)
    predictor = StressPredictor(cfg)
    times = [float(i) for i in range(30)]
    values = [0.30 + 0.010 * i for i in range(30)]
    forecast = predictor.forecast(times, values, baseline=0.30, horizon=30.0)
    assert forecast.point > values[-1], "上升趋势下短期预测应高于当前值"
    assert forecast.slope > 0
    assert forecast.lower <= forecast.point <= forecast.upper


def test_forecast_mean_reverts_over_long_horizon() -> None:
    """均值回复：人总会平复，长期预测应被拉回基线。"""
    predictor = StressPredictor(Config())
    times = [float(i) for i in range(10)]
    values = [0.80] * 10
    forecast = predictor.forecast(times, values, baseline=0.20, horizon=600.0)
    assert forecast.point < 0.5, f"长期预测应回归基线，实际 {forecast.point}"


def test_forecast_stays_in_unit_interval() -> None:
    predictor = StressPredictor(Config())
    times = [float(i) for i in range(20)]
    values = [0.95 + 0.005 * i for i in range(20)]
    forecast = predictor.forecast(times, values, baseline=0.1, horizon=300.0)
    assert 0.0 <= forecast.lower <= forecast.point <= forecast.upper <= 1.0


def test_residual_update_moves_toward_error() -> None:
    predictor = StressPredictor(Config())
    before = predictor.sigma_resid
    for _ in range(50):
        predictor.update_residual(predicted=0.2, actual=0.9)
    assert predictor.sigma_resid > before


# --------------------------------------------------------------------------
# 可继续性
# --------------------------------------------------------------------------


def test_high_duration_counts_contiguous_run() -> None:
    times = [float(i) for i in range(10)]
    values = [0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    assert high_duration(times, values, 0.75) == pytest.approx(7.0)


def test_capacity_pause_when_forecast_is_high() -> None:
    assessor = CapacityAssessor(Config())
    forecast = Forecast(
        horizon_s=60.0, point=0.90, lower=0.85, upper=0.95,
        slope=0.5, baseline=0.3, method="test", validated=False,
    )
    result = assessor.assess([float(i) for i in range(10)], [0.85] * 10, forecast, 0.4)
    assert result.state == "PAUSE"
    assert result.reasons
    assert any("未" in r and "校准" in r for r in result.reasons), "必须提示预测未校准"


def test_capacity_ok_when_everything_is_low() -> None:
    assessor = CapacityAssessor(Config())
    forecast = Forecast(
        horizon_s=60.0, point=0.20, lower=0.15, upper=0.25,
        slope=-0.1, baseline=0.2, method="test", validated=False,
    )
    result = assessor.assess([float(i) for i in range(10)], [0.2] * 10, forecast, 0.9)
    assert result.state == "OK"
    assert 0.0 <= result.score <= 1.0


def test_capacity_watch_in_the_middle() -> None:
    assessor = CapacityAssessor(Config())
    forecast = Forecast(
        horizon_s=60.0, point=0.62, lower=0.55, upper=0.7,
        slope=0.1, baseline=0.3, method="test", validated=False,
    )
    result = assessor.assess([float(i) for i in range(10)], [0.58] * 10, forecast, 0.5)
    assert result.state == "WATCH"


def test_capacity_score_decreases_with_higher_forecast() -> None:
    assessor = CapacityAssessor(Config())
    times = [float(i) for i in range(10)]
    values = [0.5] * 10
    low = assessor.assess(
        times, values,
        Forecast(60.0, 0.3, 0.25, 0.35, 0.0, 0.3, "test", False), 0.5,
    )
    high = assessor.assess(
        times, values,
        Forecast(60.0, 0.8, 0.75, 0.85, 0.0, 0.3, "test", False), 0.5,
    )
    assert high.score < low.score
