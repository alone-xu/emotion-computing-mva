"""① 感知层 · 真实后端与降级链测试（不依赖摄像头 / 麦克风）。

这些测试的价值在于：**在没有人脸、没有音频、没有文本输入的机器上**，
验证感知层依然不会崩溃，并且降级行为可预期（需求 P-9、验收 E-3）。
"""

from __future__ import annotations

import numpy as np

from emotion_computing.config import Config
from emotion_computing.perception.base import build_sensor, reset_warnings
from emotion_computing.perception.speech import speech_score
from emotion_computing.perception.synthetic import SyntheticStressDriver
from emotion_computing.perception.text import lexicon_anxiety_score


def test_amplitude_perturbation_tracks_shimmer_proxy() -> None:
    """需求 P-3 的 shimmer：用纯 numpy 的帧级振幅微扰代理覆盖。

    刻意不依赖 librosa —— 这条测试必须在「只装了 numpy」的机器上也能跑，
    否则 CI 里的感知层就没有任何真实特征可测了。
    """
    from emotion_computing.perception.speech import amplitude_perturbation

    sr = 16000
    t = np.arange(sr) / sr

    steady = 0.3 * np.sin(2 * np.pi * 180.0 * t)  # 幅度恒定 → 微扰应接近 0
    wobbly = steady * (1.0 + 0.35 * np.sin(2 * np.pi * 7.0 * t))  # 幅度抖动 → 应明显更大

    assert amplitude_perturbation(steady) < 0.05
    assert amplitude_perturbation(wobbly) > amplitude_perturbation(steady) * 3
    assert amplitude_perturbation(np.zeros(0)) == 0.0, "空输入不能抛异常"


def test_synthetic_driver_is_deterministic() -> None:
    first = SyntheticStressDriver(seed=42)
    second = SyntheticStressDriver(seed=42)
    assert first.truth(100.0) == second.truth(100.0)
    assert first.modality_score("face", 55.0) == second.modality_score("face", 55.0)


def test_synthetic_truth_follows_control_points() -> None:
    driver = SyntheticStressDriver()
    assert driver.truth(180.0) > driver.truth(0.0), "峰值应高于静息"
    assert 0.0 <= driver.truth(10_000.0) <= 1.0


def test_build_sensor_never_raises_for_any_modality() -> None:
    reset_warnings()
    cfg = Config()
    for name in ("face", "speech", "text"):
        sensor = build_sensor(name, cfg)
        assert sensor is not None
        assert sensor.backend, f"{name} 必须有明确的 backend 标记"


def test_sensors_handle_payload_without_any_media() -> None:
    """没有任何真实输入时也必须返回合法的观测（需求 P-9）。"""
    reset_warnings()
    cfg = Config()
    driver = SyntheticStressDriver(seed=1)
    for name in ("face", "speech", "text"):
        sensor = build_sensor(name, cfg, driver)
        observation = sensor.sense({"t": 10.0})
        assert observation.modality == name
        assert 0.0 <= observation.score <= 1.0
        assert 0.0 <= observation.confidence <= 1.0
        assert isinstance(observation.backend, str) and observation.backend


def test_face_backend_survives_blank_frame() -> None:
    """真实面部后端遇到"没有人脸的图像"时必须优雅退化，而不是崩溃。"""
    reset_warnings()
    cfg = Config()
    sensor = build_sensor("face", cfg)
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    observation = sensor.sense({"t": 0.0, "frame": blank})
    assert 0.0 <= observation.score <= 1.0
    assert observation.available is False or "synthetic" in observation.backend


def test_speech_backend_survives_tiny_audio() -> None:
    reset_warnings()
    cfg = Config()
    sensor = build_sensor("speech", cfg)
    observation = sensor.sense({"t": 0.0, "audio": np.zeros(8), "sr": 16000})
    assert 0.0 <= observation.score <= 1.0


def test_lexicon_distinguishes_anxious_from_neutral_text() -> None:
    anxious, hits = lexicon_anxiety_score("我好紧张，来不及了，怎么办")
    neutral, _ = lexicon_anxiety_score("今天天气不错")
    assert anxious > neutral
    assert hits, "命中词表不应为空"


def test_lexicon_negation_reduces_score() -> None:
    plain, _ = lexicon_anxiety_score("我紧张")
    negated, _ = lexicon_anxiety_score("我不紧张")
    assert negated < plain


def test_lexicon_intensifier_increases_score() -> None:
    plain, _ = lexicon_anxiety_score("我紧张")
    strong, _ = lexicon_anxiety_score("我非常紧张")
    assert strong > plain


def test_lexicon_handles_empty_text() -> None:
    score, hits = lexicon_anxiety_score("")
    assert score == 0.0
    assert hits == {}


def test_speech_score_orders_tense_above_calm() -> None:
    calm = speech_score(
        {
            "f0_cv": 0.02, "energy": 0.01, "speech_rate": 0.2,
            "pause_ratio": 0.05, "jitter": 0.001, "hnr": 0.90,
        }
    )
    tense = speech_score(
        {
            "f0_cv": 0.40, "energy": 0.25, "speech_rate": 0.90,
            "pause_ratio": 0.60, "jitter": 0.090, "hnr": 0.30,
        }
    )
    assert 0.0 <= calm < tense <= 1.0


def test_synthetic_dropout_yields_unavailable_observation() -> None:
    reset_warnings()
    cfg = Config()
    driver = SyntheticStressDriver(seed=3, dropout=1.0)
    sensor = build_sensor("text", cfg, driver)
    observation = sensor.sense({"t": 5.0})
    assert observation.available is False
