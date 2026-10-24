"""② 快系统 · EWMA 平滑与事件检测测试。"""

from __future__ import annotations

from emotion_computing.config import Config
from emotion_computing.contracts import PerceptionFrame, StressSample
from emotion_computing.fast_system import EventDetector, FastEstimator


def frame(t: float, value: float) -> PerceptionFrame:
    return PerceptionFrame(
        timestamp=t,
        observations=(),
        fused_raw=value,
        normalized=value,
        contributions={"face": 1.0},
    )


def sample(t: float, value: float) -> StressSample:
    return StressSample(
        timestamp=t, raw=value, smoothed=value, confidence=0.8,
        contributions={"face": 1.0},
    )


# --------------------------------------------------------------------------
# EWMA
# --------------------------------------------------------------------------


def test_first_sample_is_not_smoothed() -> None:
    estimator = FastEstimator(Config())
    result = estimator.update(frame(0.0, 0.8))
    assert abs(result.smoothed - 0.8) < 1e-9


def test_ewma_converges_to_constant_input() -> None:
    estimator = FastEstimator(Config())
    result = estimator.update(frame(0.0, 0.9))
    for i in range(1, 200):
        result = estimator.update(frame(float(i), 0.9))
    assert abs(result.smoothed - 0.9) < 0.01


def test_alpha_is_time_aware_and_monotonic() -> None:
    """不同 Δt 下行为可预期：Δt 越大，α 越接近 1（更信任新样本）。"""
    a_small = FastEstimator.alpha_for(0.5, 1.5)
    a_large = FastEstimator.alpha_for(2.0, 1.5)
    assert 0.0 < a_small < a_large < 1.0
    # 两个 1 秒步长与一个 2 秒步长不应相差一个数量级（时间常数一致性）
    two_steps = 1.0 - (1.0 - FastEstimator.alpha_for(1.0, 1.5)) ** 2
    assert abs(two_steps - a_large) < 0.05


def test_smoothing_suppresses_single_frame_noise() -> None:
    estimator = FastEstimator(Config())
    for i in range(50):
        estimator.update(frame(float(i), 0.3))
    spiked = estimator.update(frame(50.0, 1.0))
    assert spiked.smoothed < 0.75, "单帧尖峰不应立刻把平滑值拉到顶"


# --------------------------------------------------------------------------
# 事件检测
# --------------------------------------------------------------------------


def test_spike_detected_on_step_change() -> None:
    detector = EventDetector(Config())
    events = []
    for i in range(20):
        events += detector.update(sample(float(i), 0.2))
    for i in range(20, 30):
        events += detector.update(sample(float(i), 0.8))
    kinds = [e.kind for e in events]
    assert "spike" in kinds, f"阶跃上升应触发 spike，实际事件：{kinds}"
    spike = next(e for e in events if e.kind == "spike")
    assert spike.magnitude > 0
    assert spike.dominant_modality == "face"


def test_no_spike_on_flat_signal() -> None:
    detector = EventDetector(Config())
    events = []
    for i in range(60):
        events += detector.update(sample(float(i), 0.4))
    assert not [e for e in events if e.kind == "spike"], "平稳序列不应触发 spike"


def test_sustained_high_fires_only_once() -> None:
    """同一次高位只应报一次，避免刷屏（需求 F-3）。"""
    detector = EventDetector(Config())
    events = []
    for i in range(90):
        events += detector.update(sample(float(i), 0.9))
    sustained = [e for e in events if e.kind == "sustained_high"]
    assert len(sustained) == 1, f"应只触发一次，实际 {len(sustained)} 次"


def test_sustained_high_resets_after_recovery() -> None:
    detector = EventDetector(Config())
    events = []
    for i in range(40):
        events += detector.update(sample(float(i), 0.9))
    for i in range(40, 60):
        events += detector.update(sample(float(i), 0.2))
    for i in range(60, 100):
        events += detector.update(sample(float(i), 0.9))
    sustained = [e for e in events if e.kind == "sustained_high"]
    assert len(sustained) == 2, "回落后再次升高应重新触发"
