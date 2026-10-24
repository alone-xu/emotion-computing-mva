"""① 感知层 · 语音声学。

首选（也是目前唯一真实可用的）后端：librosa 声学特征。
可选升级：ASR 转写 → 交给语义模块（见 perception/text.py）。

诚实性说明 [V]：
    这里用到的声学指标（F0 统计、jitter、HNR、停顿比等）是语音情感研究中的
    常用量，但**其阈值与权重未在本项目中校准**。输出是「紧张相关语音活动」的
    代理量，不是经过验证的语音情感识别模型结果。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import ModalityObservation
from .base import ModalitySensor, ramp_down, ramp_up

#: 语音特征的紧张度权重（启发式初值 [A]）
SPEECH_FEATURE_WEIGHTS: dict[str, float] = {
    "f0_cv": 0.30,  # 基频变异系数（音调抖动）
    "energy": 0.20,  # 能量抬升
    "speech_rate": 0.15,  # 有声比例（语速代理）
    "pause_ratio": 0.15,  # 停顿异常
    "jitter": 0.10,  # 周期扰动
    "hnr": 0.10,  # 谐噪比（反向）
}


def amplitude_perturbation(y: np.ndarray, hop: int = 512, gate_ratio: float = 0.02) -> float:
    """帧级振幅微扰代理（用于 shimmer）。

    做法：按固定 hop 切帧，取每帧峰值幅度，再用"有声帧"计算相邻帧的相对差均值。
    它**不是**声学定义的 shimmer（那需要逐周期峰值检测），只是"幅度抖动有多厉害"的
    一个可复现代理量；单独抽成函数是为了能在没有 librosa 的机器上被单测覆盖。
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    n_frames = y.size // hop
    if n_frames < 3:
        return 0.0

    frames = y[: n_frames * hop].reshape(n_frames, hop)
    amp = np.max(np.abs(frames), axis=1)
    gate = gate_ratio * (float(np.max(amp)) + 1e-9)
    voiced = amp[amp > gate]
    if voiced.size < 3:
        return 0.0
    return float(np.mean(np.abs(np.diff(voiced))) / (float(np.mean(voiced)) + 1e-9))


def acoustic_features(y: np.ndarray, sr: int, librosa: Any) -> dict[str, float]:
    """从一段单声道波形中提取声学特征。"""
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size < int(0.20 * sr):
        return {}

    # 基频轨迹
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=70.0, fmax=400.0, sr=sr, frame_length=1024
    )
    f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    f0_mean = float(np.mean(f0_valid)) if f0_valid.size else 0.0
    f0_std = float(np.std(f0_valid)) if f0_valid.size else 0.0
    f0_cv = f0_std / (f0_mean + 1e-9)

    # 能量与停顿
    rms = librosa.feature.rms(y=y)[0]
    energy = float(np.mean(rms))
    threshold = 0.10 * (float(np.max(rms)) + 1e-9)
    pause_ratio = float(np.mean(rms < threshold))

    # 语速代理：有声帧比例
    if voiced_flag is not None and len(voiced_flag):
        speech_rate = float(np.mean(voiced_flag))
    else:
        speech_rate = float(np.mean(rms >= threshold))

    # jitter：相邻有声基频周期的平均相对差
    if f0_valid.size > 2:
        jitter = float(np.mean(np.abs(np.diff(f0_valid))) / (f0_mean + 1e-9))
    else:
        jitter = 0.0

    # HNR 的简化代理：谐波能量占比
    hnr = 0.5
    try:
        y_harm, _ = librosa.effects.hpss(y)
        total = float(np.sum(y**2)) + 1e-9
        hnr = float(np.clip(np.sum(y_harm**2) / total, 0.0, 1.0))
    except Exception:  # noqa: BLE001 —— hpss 在极短音频上可能失败
        pass

    # shimmer（振幅微扰的代理，需求 P-3）：固定跳长下有声帧峰值幅度的相邻相对差。
    # 严格的 shimmer 要逐周期找峰值，这里用帧级幅度代替 —— 只求反映振幅抖动，
    # 不假装等同声学定义的 shimmer，报告里也按"代理特征"描述。
    shimmer = amplitude_perturbation(y)

    return {
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_cv": f0_cv,
        "energy": energy,
        "pause_ratio": pause_ratio,
        "speech_rate": speech_rate,
        "jitter": jitter,
        "shimmer": shimmer,
        "hnr": hnr,
    }


def speech_score(features: dict[str, float]) -> float:
    """把声学特征压缩成 [0,1] 的语音紧张度分项（启发式 [V]）。"""
    comps = {
        "f0_cv": ramp_up(features.get("f0_cv", 0.0), lo=0.08, hi=0.35),
        "energy": ramp_up(features.get("energy", 0.0), lo=0.02, hi=0.20),
        "speech_rate": ramp_up(features.get("speech_rate", 0.0), lo=0.30, hi=0.85),
        "pause_ratio": ramp_up(features.get("pause_ratio", 0.0), lo=0.15, hi=0.55),
        "jitter": ramp_up(features.get("jitter", 0.0), lo=0.01, hi=0.08),
        # HNR 越低越紧张 → 反向
        "hnr": ramp_down(features.get("hnr", 0.9), hi=0.75, lo=0.35),
    }
    total_w = sum(SPEECH_FEATURE_WEIGHTS.values())
    score = sum(SPEECH_FEATURE_WEIGHTS[k] * comps[k] for k in SPEECH_FEATURE_WEIGHTS)
    return float(np.clip(score / total_w, 0.0, 1.0))


class LibrosaSpeechSensor(ModalitySensor):
    name = "speech"

    def __init__(self, config: Any, driver: Any = None) -> None:
        super().__init__()
        from .synthetic import SyntheticStressDriver

        self.driver = driver if isinstance(driver, SyntheticStressDriver) else None
        self.config = config

        import librosa

        self._librosa = librosa
        self.sr = int(getattr(config, "audio_sr", 16000))
        self.backend = "librosa"

    def sense(self, payload: dict[str, Any]) -> ModalityObservation:
        t = float(payload.get("t", 0.0))
        audio = payload.get("audio")
        if audio is None or np.size(audio) == 0:
            return self._synthetic_fallback(t, "no_audio")

        try:
            sr = int(payload.get("sr", self.sr))
            feats = acoustic_features(np.asarray(audio, dtype=float), sr, self._librosa)
            if not feats:
                return self._missing(t, "audio_too_short")
            return ModalityObservation(
                modality=self.name,
                timestamp=t,
                score=speech_score(feats),
                confidence=0.7,
                available=True,
                backend="librosa",
                features={k: float(v) for k, v in feats.items()},
                note="声学代理特征，阈值未经校准 [V]",
            )
        except Exception as exc:  # noqa: BLE001
            return self._synthetic_fallback(t, f"librosa_error:{exc!r}")
