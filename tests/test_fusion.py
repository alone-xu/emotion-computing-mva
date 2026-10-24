"""① 感知层 · 融合与基线归一化测试。"""

from __future__ import annotations

from emotion_computing.config import Config
from emotion_computing.contracts import ModalityObservation
from emotion_computing.perception.fusion import FusionEngine, safe_sigmoid


def obs(
    modality: str, score: float, confidence: float = 0.8, available: bool = True, t: float = 0.0
) -> ModalityObservation:
    return ModalityObservation(
        modality=modality,
        timestamp=t,
        score=score,
        confidence=confidence,
        available=available,
        backend="test",
    )


def test_contributions_sum_to_one() -> None:
    engine = FusionEngine(Config())
    frame = engine.fuse([obs("face", 0.5, 0.8), obs("speech", 0.5, 0.6), obs("text", 0.5, 0.6)])
    assert abs(sum(frame.contributions.values()) - 1.0) < 1e-9
    assert set(frame.contributions) == {"face", "speech", "text"}


def test_missing_modality_weights_are_reallocated() -> None:
    """需求 P-7：缺失模态的权重必须自动从分母消失。"""
    engine = FusionEngine(Config())
    frame = engine.fuse(
        [
            obs("face", 0.7, 0.9),
            obs("speech", 0.0, 0.0, available=False),
            obs("text", 0.0, 0.0, available=False),
        ]
    )
    assert list(frame.contributions) == ["face"]
    assert abs(frame.contributions["face"] - 1.0) < 1e-9
    # 只剩下 face 时，融合值应完全由 face 决定
    assert abs(frame.fused_raw - 0.7) < 1e-9


def test_all_modalities_missing_yields_zero() -> None:
    engine = FusionEngine(Config())
    frame = engine.fuse([obs("face", 0.0, 0.0, available=False)])
    assert frame.fused_raw == 0.0
    assert frame.normalized == 0.0
    assert frame.contributions == {}


def test_baseline_calibration_separates_rest_and_stress() -> None:
    """需求 P-8：标定后，静息应低、紧张应高，且不饱和。"""
    cfg = Config()
    engine = FusionEngine(cfg)

    # 基线期：三个模态都稳定在 0.20 附近（含小噪声）
    for i in range(40):
        jitter = 0.01 * ((i % 3) - 1)
        observations = [
            obs("face", 0.20 + jitter, 0.8),
            obs("speech", 0.20 - jitter, 0.8),
            obs("text", 0.20, 0.8),
        ]
        engine.observe_baseline(observations)
        engine.fuse(observations, float(i))

    assert engine.finalize_baseline()
    assert engine.calibrated
    assert engine.fused_stats is not None

    rest = engine.fuse(
        [obs("face", 0.20, 0.8), obs("speech", 0.20, 0.8), obs("text", 0.20, 0.8)], 41.0
    )
    high = engine.fuse(
        [obs("face", 0.80, 0.8), obs("speech", 0.80, 0.8), obs("text", 0.80, 0.8)], 42.0
    )

    assert rest.normalized < 0.35, f"静息应偏低，实际 {rest.normalized}"
    assert high.normalized > 0.70, f"高位应偏高，实际 {high.normalized}"
    # 关键：不能因为 σ 过小就饱和到 1.0（这是 σ 收缩要解决的问题）
    assert high.normalized < 0.999, "输出饱和说明 σ 收缩未生效"


def test_baseline_needs_enough_samples() -> None:
    engine = FusionEngine(Config())
    engine.fuse([obs("face", 0.2)], 0.0)
    assert engine.finalize_baseline(min_samples=5) is False
    assert engine.calibrated is False


def test_safe_sigmoid_no_overflow() -> None:
    assert safe_sigmoid(1000.0) == 1.0
    assert safe_sigmoid(-1000.0) == 0.0
    assert abs(safe_sigmoid(0.0) - 0.5) < 1e-12


def test_observations_are_clipped_to_unit_interval() -> None:
    assert obs("face", 1.7).score == 1.0
    assert obs("face", -0.4).score == 0.0
    assert obs("face", 0.5, confidence=3.0).confidence == 1.0
