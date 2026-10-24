"""M-3 模态级诊断（03-audit-requirements.md §3.3）。

借鉴 MIT Media Lab "Modality Sabotage" 框架：
  1. 用每个模态**独立**预测 stress（这里直接使用金标准的 modality_labels）
  2. 对比 fused 与 gold：哪个模态以"高置信度错误方向"主导了融合？
  3. 标记破坏者：`confidence ≥ θ_saboteur` 且 `sign(modality - gold) ≠ sign(fused - gold)`

破坏者的判定对融合逻辑调优极其关键。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..contracts import GoldLabel, ModalityAttribution
from .metrics import error_threshold

#: 03 §M-3 模态相对权重（与 Config 默认一致；可被个体画像覆盖）
DEFAULT_MODALITY_WEIGHTS: dict[str, float] = {
    "face": 0.40,
    "speech": 0.35,
    "text": 0.25,
}


@dataclass
class DiagnosticConfig:
    theta_saboteur: float = 0.7
    weights: dict[str, float] | None = None
    modalities: tuple[str, ...] = ("face", "speech", "text")

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = dict(DEFAULT_MODALITY_WEIGHTS)

    def as_dict(self) -> dict:
        return {
            "theta_saboteur": self.theta_saboteur,
            "weights": dict(self.weights),
            "modalities": list(self.modalities),
        }


def _safe_sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def attribute_sample(
    sample: GoldLabel,
    fused: float,
    modality_confidence: dict[str, float] | None = None,
    cfg: DiagnosticConfig | None = None,
) -> ModalityAttribution:
    """对单样本做模态级归因。

    参：
      sample: 金标准（包含 modality_labels）
      fused:  MVA 融合后 stress
      modality_confidence: 各模态置信度；缺省视为 1.0
      cfg: 诊断配置
    返回：ModalityAttribution
    """
    cfg = cfg or DiagnosticConfig()
    ml = dict(sample.modality_labels)
    # 把缺测模态填 0
    for m in cfg.modalities:
        ml.setdefault(m, 0.0)
    face = float(ml.get("face", 0.0))
    speech = float(ml.get("speech", 0.0))
    text = float(ml.get("text", 0.0))
    gold = float(sample.stress)
    dominant = _dominant(face, speech, text, cfg)

    # 错误方向：positive = fused 高估，negative = 低估
    fused_err = fused - gold
    fused_sign = _safe_sign(fused_err)

    conf = modality_confidence or {m: 1.0 for m in cfg.modalities}

    saboteur: str | None = None
    reason = "fused 与 gold 误差在合理区间"
    for m, val in (("face", face), ("speech", speech), ("text", text)):
        mod_err = val - gold
        mod_sign = _safe_sign(mod_err)
        if (
            mod_sign != 0
            and fused_sign != 0
            and mod_sign != fused_sign                       # 方向相反
            and conf.get(m, 0.0) >= cfg.theta_saboteur       # 高置信
        ):
            saboteur = m
            reason = (
                f"{m} 模态以 {conf.get(m, 0.0):.2f} 置信度输出 "
                f"{val:.2f}，与 gold {gold:.2f} 方向相反，"
                f"主导了 fused {fused:.2f}"
            )
            break

    return ModalityAttribution(
        sample_id=sample.sample_id,
        face=face,
        speech=speech,
        text=text,
        fused=fused,
        gold=gold,
        dominant_modality=dominant,
        saboteur=saboteur,
        reason=reason,
    )


def _dominant(face: float, speech: float, text: float, cfg: DiagnosticConfig) -> str:
    """按权重 × 值的合成量取最大者为 dominant。"""
    best = "face"
    best_v = -1e18
    for name, val in (("face", face), ("speech", speech), ("text", text)):
        w = cfg.weights.get(name, 0.0)
        score = w * val
        if score > best_v:
            best_v = score
            best = name
    return best


def attribute_dataset(
    samples: Sequence[GoldLabel],
    fused: Sequence[float],
    modality_confidence: Sequence[dict[str, float]] | None = None,
    cfg: DiagnosticConfig | None = None,
) -> list[ModalityAttribution]:
    """批量归因。"""
    if len(samples) != len(fused):
        raise ValueError("attribute_dataset: samples 与 fused 长度不一致")
    out: list[ModalityAttribution] = []
    for i, s in enumerate(samples):
        mc = modality_confidence[i] if modality_confidence else None
        out.append(attribute_sample(s, fused[i], mc, cfg))
    return out


def saboteur_summary(
    attributions: Sequence[ModalityAttribution],
    error_mask: Sequence[bool] | None = None,
) -> dict[str, int]:
    """统计破坏者分布。`error_mask` 限定只在错误样本里统计。"""
    counter: dict[str, int] = {"face": 0, "speech": 0, "text": 0, "none": 0}
    for i, a in enumerate(attributions):
        if error_mask is not None and not error_mask[i]:
            continue
        counter[a.saboteur or "none"] += 1
    return counter
