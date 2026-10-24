"""audit.failure_attribution 单元测试。"""

from __future__ import annotations

from emotion_computing.audit.failure_attribution import (
    AttributionEvidence,
    attribute_batch,
    attribute_one,
    build_distribution,
)
from emotion_computing.audit.modality_diagnostic import attribute_sample
from emotion_computing.contracts import GoldLabel, ModalityAttribution


def _g(sid: str, stress: float, state: str = "OK",
       modalities: dict | None = None) -> GoldLabel:
    return GoldLabel(
        sample_id=sid, stress=stress, state=state,
        modality_labels=modalities or {"face": stress, "speech": stress, "text": stress},
    )


def _ev(sid: str, gold: float, fused: float, mva_state: str = "OK",
        mva_action: str = "continue", data_suspect: bool = False,
        att: ModalityAttribution | None = None,
        retrieved=(), relevant=(),
        gold_state: str = "OK") -> AttributionEvidence:
    return AttributionEvidence(
        sample_id=sid, gold_label=_g(sid, gold, state=gold_state), fused=fused,
        mva_state=mva_state, mva_action=mva_action,
        data_suspect=data_suspect, modality_attribution=att,
        memory_retrieved=tuple(retrieved), memory_relevant=tuple(relevant),
    )


# ---------- 步骤 1: data_problem ----------
def test_data_problem():
    ev = _ev("s1", gold=0.5, fused=0.2, data_suspect=True)
    case = attribute_one(ev, error_theta=0.20)
    assert case.system_attribution == "data_problem"


# ---------- 步骤 2: modality_sabotage ----------
def test_modality_sabotage():
    # gold=0.6, fused=0.3 (低估)；face=0.9 高置信说高 → 方向相反 → face 是破坏者
    s = _g("s2", 0.6, modalities={"face": 0.9, "speech": 0.5, "text": 0.5})
    att = attribute_sample(s, 0.3,
                           modality_confidence={"face": 0.9, "speech": 0.5, "text": 0.5})
    assert att.saboteur == "face"
    ev = _ev("s2", gold=0.6, fused=0.3, att=att)
    case = attribute_one(ev, error_theta=0.20)
    assert case.system_attribution == "modality_sabotage"
    assert "face" in (case.evidence.get("saboteur") or "")


# ---------- 步骤 3: perception_failure ----------
def test_perception_failure():
    # face/speech 都与 gold 0.2 同方向（都高于 gold），但 fused 偏离大
    # speech 0.7 vs gold 0.2 差 0.5 > 0.25（步骤 3 触发）
    s = _g("s3", 0.2, modalities={"face": 0.3, "speech": 0.7, "text": 0.3})
    # 给所有模态低置信，避免步骤 2 触发
    att = attribute_sample(s, 0.8, modality_confidence={"face": 0.3, "speech": 0.3, "text": 0.3})
    assert att.saboteur is None
    ev = _ev("s3", gold=0.2, fused=0.8, att=att)
    case = attribute_one(ev, error_theta=0.20, modality_error_theta=0.25)
    assert case.system_attribution == "perception_failure"


# ---------- 步骤 4: slow_eval_failure ----------
def test_slow_eval_failure():
    s = _g("s4", 0.5, state="PAUSE")
    att = ModalityAttribution(
        sample_id="s4", face=0.5, speech=0.5, text=0.5,
        fused=0.6, gold=0.5, dominant_modality="face", saboteur=None, reason=""
    )
    # gold.state=PAUSE, mva_state=OK 跨级；fused 偏离 0.4
    ev = _ev("s4", gold=0.5, fused=0.9, mva_state="OK", att=att, gold_state="PAUSE")
    case = attribute_one(ev, error_theta=0.20, modality_error_theta=0.25)
    assert case.system_attribution == "slow_eval_failure"


# ---------- 步骤 5: memory_interference ----------
def test_memory_interference():
    s = _g("s5", 0.3, state="WATCH")
    att = ModalityAttribution(
        sample_id="s5", face=0.3, speech=0.3, text=0.3,
        fused=0.6, gold=0.3, dominant_modality="face", saboteur=None, reason=""
    )
    ev = _ev("s5", gold=0.3, fused=0.6, mva_state="WATCH", att=att,
             gold_state="WATCH",
             retrieved=("ep_a", "ep_b", "ep_c"), relevant=(False, False, False))
    case = attribute_one(ev, error_theta=0.20, modality_error_theta=0.10)
    assert case.system_attribution == "memory_interference"


# ---------- 步骤 6: action_failure ----------
def test_action_failure():
    s = _g("s6", 0.5, state="PAUSE")
    att = ModalityAttribution(
        sample_id="s6", face=0.5, speech=0.5, text=0.5,
        fused=0.5, gold=0.5, dominant_modality="face", saboteur=None, reason=""
    )
    # gold.state=PAUSE, mva_state=PAUSE（不跨级）, mva_action=continue（应升级但没）
    ev = _ev("s6", gold=0.5, fused=0.8, mva_state="PAUSE", mva_action="continue",
             att=att, gold_state="PAUSE")
    case = attribute_one(ev, error_theta=0.20, modality_error_theta=0.10)
    assert case.system_attribution == "action_failure"


# ---------- 步骤 7: combined_failure ----------
def test_combined_failure():
    s = _g("s7", 0.5, state="OK")
    att = ModalityAttribution(
        sample_id="s7", face=0.5, speech=0.5, text=0.5,
        fused=0.8, gold=0.5, dominant_modality="face", saboteur=None, reason=""
    )
    ev = _ev("s7", gold=0.5, fused=0.8, mva_state="OK", mva_action="continue",
             att=att, gold_state="OK")
    case = attribute_one(ev, error_theta=0.20, modality_error_theta=0.10)
    assert case.system_attribution == "combined_failure"


# ---------- 非错误样本 ----------
def test_ok_case():
    ev = _ev("s8", gold=0.5, fused=0.55)
    case = attribute_one(ev, error_theta=0.20)
    assert case.system_attribution == "ok"


# ---------- F-3 分布 ----------
def test_distribution_percentages_sum_to_100():
    s = _g("s", 0.5, state="OK")
    att = ModalityAttribution(
        sample_id="s", face=0.5, speech=0.5, text=0.5,
        fused=0.5, gold=0.5, dominant_modality="face", saboteur=None, reason=""
    )
    evs = [
        _ev(f"s{i}", gold=0.5, fused=0.1 + 0.01 * i, att=att) for i in range(5)
    ]
    evs[0] = _ev("s0", gold=0.5, fused=0.1, data_suspect=True, att=att)
    cases = attribute_batch(evs, error_theta=0.20, modality_error_theta=0.10)
    dist = build_distribution(cases)
    total = sum(dist.percentages.values())
    assert abs(total - 100.0) < 1e-6


def test_priority_easy_first():
    """修复方向应按"高占比+易修复"排序。"""
    from emotion_computing.audit.failure_attribution import build_distribution
    from emotion_computing.contracts import FailureCase
    cases = [
        FailureCase(sample_id="s1", error=0.3, system_attribution="perception_failure"),
        FailureCase(sample_id="s2", error=0.3, system_attribution="modality_sabotage"),
        FailureCase(sample_id="s3", error=0.3, system_attribution="data_problem"),
    ]
    dist = build_distribution(cases)
    # top_repair 第一个应是 easy
    assert "modality_sabotage" in dist.top_repair[0] or "data_problem" in dist.top_repair[0]
