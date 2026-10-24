"""① 感知层 · 多模态融合与基线归一化。

实现需求 P-7（缺失模态自动重分配权重）与 P-8（个体基线归一化）。

流水线（架构文档 §5.1.6）：
    Step 1  基线期采集：每个模态的 score 序列、融合 raw 序列
    Step 2  模态级归一化：z-score → sigmoid，得到个体内相对量
    Step 3  加权融合：Σ w·c·s / Σ w·c（缺失模态的权重自然从分母消失）
    Step 4  融合后归一化：对 fused 再做一次基线归一化

关键点：在基线标定完成前，Step 2 与 Step 4 都是恒等变换，
因此系统**冷启动即可用**，标定完成后自动切换到个体化尺度。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from ..contracts import ModalityObservation, PerceptionFrame


def safe_sigmoid(x: float) -> float:
    """数值安全的 sigmoid，避免 math.exp 溢出。"""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class FusionEngine:
    """多模态融合引擎。"""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.weights: dict[str, float] = config.modality_weights()

        # 基线采集缓冲
        self._modality_scores: dict[str, list[float]] = {}
        self._fused_buffer: list[float] = []

        # 标定结果
        self._modality_stats: dict[str, tuple[float, float]] = {}
        self._fused_stats: tuple[float, float] | None = None
        self.calibrated: bool = False

    # ------------------------------------------------------------------
    # 归一化
    # ------------------------------------------------------------------
    def _sigmoid_normalize(self, x: float, mu: float, sigma: float) -> float:
        """把 x 映射成个体内相对量 [0,1]。

        σ 收缩（shrinkage）—— 这是本系统最容易踩的坑：
        静息期只有 30 秒，估出的 σ 天然极小（人在静息时非常稳定），
        若直接做 z-score，任务中任何微小变化都会被放大成好几个 σ，
        输出立刻饱和到 1.0，曲线完全失去分辨力。
        因此把 σ 与一个先验尺度混合：σ_eff = sqrt(σ² + σ_prior²)。
        """
        cfg = self.config
        sigma_eff = math.sqrt(sigma * sigma + cfg.sigma_prior * cfg.sigma_prior)
        sigma_eff = max(sigma_eff, cfg.sigma_floor)
        z = (x - mu) / sigma_eff
        return safe_sigmoid((z - cfg.z_target) / cfg.z_tau)

    def _normalize_modality(self, obs: ModalityObservation) -> float:
        """Step 2：模态级个体归一化。未标定时为恒等。"""
        if not self.calibrated or obs.modality not in self._modality_stats:
            return obs.score
        mu, sigma = self._modality_stats[obs.modality]
        return self._sigmoid_normalize(obs.score, mu, sigma)

    def _normalize_fused(self, fused: float) -> float:
        """Step 4：融合值再归一化。未标定时为恒等。"""
        if not self.calibrated or self._fused_stats is None:
            return fused
        mu, sigma = self._fused_stats
        return self._sigmoid_normalize(fused, mu, sigma)

    # ------------------------------------------------------------------
    # Step 1：基线期
    # ------------------------------------------------------------------
    @property
    def baseline_sample_count(self) -> int:
        return len(self._fused_buffer)

    def observe_baseline(self, observations: Sequence[ModalityObservation]) -> None:
        """在静息基线期内调用，累积统计样本。"""
        if self.calibrated:
            return
        for obs in observations:
            if obs.available:
                self._modality_scores.setdefault(obs.modality, []).append(obs.score)

    def finalize_baseline(self, min_samples: int = 3) -> bool:
        """结束基线期，计算 μ/σ。样本不足则保持未标定状态。"""
        if self.calibrated:
            return True
        if len(self._fused_buffer) < min_samples:
            return False

        for modality, values in self._modality_scores.items():
            if len(values) >= min_samples:
                mu = float(sum(values) / len(values))
                var = sum((v - mu) ** 2 for v in values) / len(values)
                self._modality_stats[modality] = (mu, math.sqrt(var))

        mu_f = float(sum(self._fused_buffer) / len(self._fused_buffer))
        var_f = sum((v - mu_f) ** 2 for v in self._fused_buffer) / len(self._fused_buffer)
        self._fused_stats = (mu_f, math.sqrt(var_f))
        self.calibrated = True
        return True

    @property
    def fused_stats(self) -> tuple[float, float] | None:
        """融合值在基线期的 (μ, σ)；未标定时为 None。"""
        return self._fused_stats

    # ------------------------------------------------------------------
    # Step 2 + 3：融合
    # ------------------------------------------------------------------
    def fuse(
        self, observations: Iterable[ModalityObservation], timestamp: float | None = None
    ) -> PerceptionFrame:
        obs_tuple = tuple(observations)
        t = timestamp if timestamp is not None else (obs_tuple[0].timestamp if obs_tuple else 0.0)
        available = [o for o in obs_tuple if o.available]

        if not available:
            return PerceptionFrame(
                timestamp=t,
                observations=obs_tuple,
                fused_raw=0.0,
                normalized=0.0,
                contributions={},
            )

        numerator = 0.0
        denominator = 0.0
        weight_shares: dict[str, float] = {}
        value_shares: dict[str, float] = {}
        for obs in available:
            w = float(self.weights.get(obs.modality, 0.2))
            wc = w * max(obs.confidence, 1e-6)
            s = self._normalize_modality(obs)
            weighted = wc * s
            numerator += weighted
            denominator += wc
            weight_shares[obs.modality] = weight_shares.get(obs.modality, 0.0) + wc
            value_shares[obs.modality] = value_shares.get(obs.modality, 0.0) + weighted

        fused_raw = numerator / denominator

        # 贡献占比必须按 **w·c·s** 归一化，而不是只按 w·c。
        # 只按权重的话，"主导模态"永远等于权重最大的那个模态
        # （实测中恒为 face，归因完全失效）。按 w·c·s 归一化后，
        # 占比才真正反映「哪个模态把紧张度推了上去」。
        total_value = sum(value_shares.values())
        if total_value > 1e-12:
            contributions = {k: v / total_value for k, v in value_shares.items()}
        else:
            # 所有模态都接近 0：退化为按权重分配，避免除零
            total_w = sum(weight_shares.values()) or 1.0
            contributions = {k: v / total_w for k, v in weight_shares.items()}

        if not self.calibrated:
            # 基线期：把融合值也收进缓冲，供 Step 4 标定
            self._fused_buffer.append(fused_raw)

        normalized = self._normalize_fused(fused_raw)

        return PerceptionFrame(
            timestamp=t,
            observations=obs_tuple,
            fused_raw=fused_raw,
            normalized=normalized,
            contributions=contributions,
        )
