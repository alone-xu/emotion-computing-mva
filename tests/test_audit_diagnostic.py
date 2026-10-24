"""audit.modality_diagnostic 单元测试。"""

from __future__ import annotations

import pytest

from emotion_computing.audit.modality_diagnostic import (
    DiagnosticConfig,
    attribute_dataset,
    attribute_sample,
    saboteur_summary,
)
from emotion_computing.contracts import GoldLabel


def _g(sid: str, face: float, speech: float, text: float, stress: float) -> GoldLabel:
    return GoldLabel(
        sample_id=sid,
        modality_labels={"face": face, "speech": speech, "text": text},
        stress=stress,
    )


# ---------- 正例：fused 偏离 gold，且某模态"高置信反向"→ 破坏者 ----------
def test_saboteur_speech():
    # gold=0.6, fused=0.4 (低估)
    # face=0.5 (中性), speech=0.1 (高置信说低), text=0.4
    # speech 与 gold 方向相反（sign(0.1-0.6)=-1, sign(0.4-0.6)=-1）→ 同向
    # 真正反例：face 高置信说高，speech 中性，text 中性
    s = _g("s1", face=0.9, speech=0.5, text=0.5, stress=0.6)
    fused = 0.3
    a = attribute_sample(
        s, fused,
        modality_confidence={"face": 0.9, "speech": 0.5, "text": 0.5},
    )
    # face 0.9 vs gold 0.6 方向正（+1），fused 0.3 vs gold 0.6 方向负（-1）
    # 高置信 face 与 fused 方向相反 → face 是破坏者
    assert a.saboteur == "face"
    assert a.fused == 0.3
    assert a.gold == 0.6


# ---------- 反例：所有模态与 gold 同方向 → 无破坏者 ----------
def test_no_saboteur_all_aligned():
    s = _g("s2", face=0.7, speech=0.6, text=0.5, stress=0.2)
    fused = 0.3
    a = attribute_sample(s, fused,
                         modality_confidence={"face": 0.9, "speech": 0.9, "text": 0.9})
    # fused=0.3 vs gold=0.2 正向；face 0.7 vs 0.2 正向；都同向 → 无破坏者
    assert a.saboteur is None


# ---------- 反例：低置信度 → 不算破坏者 ----------
def test_low_confidence_not_saboteur():
    s = _g("s3", face=0.2, speech=0.9, text=0.3, stress=0.2)
    fused = 0.6
    a = attribute_sample(s, fused,
                         modality_confidence={"face": 0.3, "speech": 0.3, "text": 0.3})
    # 所有置信度都 < 0.7 → 无破坏者
    assert a.saboteur is None


# ---------- dominant ----------
def test_dominant_weighted():
    s = _g("s4", face=0.9, speech=0.3, text=0.3, stress=0.5)
    a = attribute_sample(s, 0.5, modality_confidence={"face": 0.5, "speech": 0.5, "text": 0.5})
    # face: 0.4*0.9 = 0.36; speech: 0.35*0.3=0.105; text: 0.25*0.3=0.075 → face
    assert a.dominant_modality == "face"


# ---------- attribute_dataset ----------
def test_attribute_dataset_basic():
    samples = (
        _g("a", 0.1, 0.1, 0.1, 0.2),
        _g("b", 0.9, 0.9, 0.9, 0.2),
    )
    fused = [0.2, 0.2]  # 两个 fused 都对齐 gold
    atts = attribute_dataset(samples, fused,
                             [{"face": 0.9, "speech": 0.9, "text": 0.9}] * 2)
    assert atts[0].saboteur is None
    assert atts[1].saboteur is None


# ---------- saboteur_summary ----------
def test_saboteur_summary():
    # gold=0.6；s1: fused=0.3, face=0.9 高置信说高 → face 是破坏者
    # gold=0.6; s2: fused=0.3, text=0.9 高置信说高 → text 是破坏者
    s1 = _g("a", 0.9, 0.5, 0.5, 0.6)
    s2 = _g("b", 0.5, 0.5, 0.9, 0.6)
    samples = (s1, s2)
    fused = [0.3, 0.3]  # 都低估
    conf = [
        {"face": 0.9, "speech": 0.3, "text": 0.3},
        {"face": 0.3, "speech": 0.3, "text": 0.9},
    ]
    atts = attribute_dataset(samples, fused, conf)
    err_mask = [True, True]
    summary = saboteur_summary(atts, err_mask)
    assert summary["face"] >= 1
    assert summary["text"] >= 1
