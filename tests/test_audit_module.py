"""audit.module_evaluator 单元测试。"""

from __future__ import annotations

import pytest

from emotion_computing.audit.module_evaluator import (
    MVASampleOutput,
    ablation,
    evaluate_action,
    evaluate_all,
    evaluate_fast_system,
    evaluate_memory,
    evaluate_perception,
    evaluate_slow_system,
)
from emotion_computing.contracts import GoldDataset, GoldLabel


def _g(sid: str, stress: float, state: str = "OK",
       modalities: dict | None = None, score: int = 3) -> GoldLabel:
    return GoldLabel(
        sample_id=sid, stress=stress, state=state,
        modality_labels=modalities or {"face": stress, "speech": stress, "text": stress},
        action_appropriateness=score,
    )


def _o(sid: str, stress: float, state: str = "OK", action: str = "continue",
        mod_scores: dict | None = None,
        retrieved: tuple = (),
        relevant: tuple = ()) -> MVASampleOutput:
    return MVASampleOutput(
        sample_id=sid, stress=stress, state=state, action=action,
        modality_scores=mod_scores or {"face": stress, "speech": stress, "text": stress},
        retrieved_memory=retrieved, memory_relevant=relevant,
    )


# ---------- ① 感知 ----------
def test_perception_perfect():
    samples = tuple(_g(f"s{i}", 0.5) for i in range(10))
    ds = GoldDataset(name="t", samples=samples)
    outputs = tuple(_o(f"s{i}", 0.5) for i in range(10))
    ev = evaluate_perception(ds, outputs)
    # CCC 应接近 1
    for m in ev.metrics:
        if m.metric_name == "ccc":
            assert m.value > 0.99
            assert m.status == "ok"


def test_perception_failure():
    samples = tuple(_g(f"s{i}", 0.5) for i in range(10))
    ds = GoldDataset(name="t", samples=samples)
    # face 一直输出 0.1 → 差距 0.4 > 0.25
    outputs = []
    for i in range(10):
        outputs.append(_o(f"s{i}", 0.5, mod_scores={"face": 0.1, "speech": 0.5, "text": 0.5}))
    ev = evaluate_perception(ds, outputs)
    perception_face_metric = [m for m in ev.metrics if m.module == "perception.face"][0]
    assert perception_face_metric.status in ("warn", "fail")


# ---------- ② 快系统 ----------
def test_fast_system_perfect():
    samples = tuple(_g(f"s{i}", 0.3) for i in range(20))
    ds = GoldDataset(name="t", samples=samples)
    outputs = tuple(_o(f"s{i}", 0.3) for i in range(20))
    ev = evaluate_fast_system(ds, outputs)
    for m in ev.metrics:
        if m.metric_name == "ccc":
            assert m.value > 0.99
            assert m.status == "ok"


def test_fast_system_low_ccc():
    samples = tuple(_g(f"s{i}", 0.1 + 0.04 * i) for i in range(20))
    ds = GoldDataset(name="t", samples=samples)
    outputs = tuple(_o(f"s{i}", 0.5) for i in range(20))  # 全部 0.5
    ev = evaluate_fast_system(ds, outputs)
    for m in ev.metrics:
        if m.metric_name == "ccc":
            assert m.value < 0.5
            assert m.status == "fail"


# ---------- ③ 慢系统 ----------
def test_slow_system_perfect():
    # 3 个类别都有支持：完美预测 → f1 = 1
    samples = []
    outputs = []
    for i in range(9):
        if i < 3:
            state = "OK"
        elif i < 6:
            state = "WATCH"
        else:
            state = "PAUSE"
        samples.append(_g(f"s{i}", 0.5, state=state))
        outputs.append(_o(f"s{i}", 0.5, state=state))
    ds = GoldDataset(name="t", samples=tuple(samples))
    ev = evaluate_slow_system(ds, tuple(outputs))
    for m in ev.metrics:
        if m.metric_name == "f1_watch_pause":
            assert m.value == 1.0


def test_slow_system_miss():
    samples = (
        _g("s0", 0.5, "PAUSE"),
        _g("s1", 0.5, "PAUSE"),
    )
    ds = GoldDataset(name="t", samples=samples)
    outputs = (
        _o("s0", 0.5, "OK"),
        _o("s1", 0.5, "OK"),
    )
    ev = evaluate_slow_system(ds, outputs)
    miss_metric = [m for m in ev.metrics if m.metric_name == "miss_rate"][0]
    assert miss_metric.value == 1.0
    assert miss_metric.status == "fail"


# ---------- ④ 记忆 ----------
def test_memory_p_at_3():
    samples = tuple(_g(f"s{i}", 0.3) for i in range(5))
    ds = GoldDataset(name="t", samples=samples)
    outputs = [
        _o("s0", 0.3, retrieved=("a", "b", "c"), relevant=(True, True, True)),
        _o("s1", 0.3, retrieved=("a", "b", "c"), relevant=(True, False, False)),
        _o("s2", 0.3, retrieved=("a", "b", "c"), relevant=(True, True, False)),
        _o("s3", 0.3, retrieved=("a", "b", "c"), relevant=(False, False, False)),
        _o("s4", 0.3),  # 无数据 → 默认全相关
    ]
    ev = evaluate_memory(ds, outputs, k=3)
    p = ev.metrics[0]
    # 前 4: 1.0, 0.333, 0.667, 0.0; 第 5 默认 1.0
    expected = (1.0 + 0.333 + 0.667 + 0.0 + 1.0) / 5
    assert p.value == pytest.approx(expected, abs=0.05)


# ---------- ⑤ 行动 ----------
def test_action_score():
    samples = tuple(_g(f"s{i}", 0.5, score=4) for i in range(5))
    ds = GoldDataset(name="t", samples=samples)
    outputs = tuple(_o(f"s{i}", 0.5) for i in range(5))
    ev = evaluate_action(ds, outputs)
    assert ev.metrics[0].value == 4.0
    assert ev.metrics[0].status == "ok"


# ---------- M-2 消融 ----------
def test_ablation_compare():
    samples = tuple(_g(f"s{i}", 0.1 + 0.05 * i) for i in range(20))
    ds = GoldDataset(name="t", samples=samples)
    full = tuple(_o(f"s{i}", 0.1 + 0.05 * i) for i in range(20))
    # 消融 P_face_only：fused stress 改为 face 模态值
    # 这模拟"关闭 speech 和 text 模态，只用 face 做融合"
    p_face = []
    for i in range(20):
        face_v = 0.1 + 0.05 * i
        p_face.append(_o(f"s{i}", face_v,  # fused = face
                         mod_scores={"face": face_v, "speech": 0.5, "text": 0.5}))
    abl = ablation(ds, full, {"P_face_only": p_face})
    a = abl["P_face_only"]
    # face 模态值与 full 模型 stress 完全一致 → 差距 0
    assert a.delta_ccc == 0.0

    # 真正退化：P_face_only 改为 face 加噪声
    p_face_noisy = []
    for i in range(20):
        face_v = 0.1 + 0.05 * i + 0.1  # 加 0.1 偏置
        p_face_noisy.append(_o(f"s{i}", min(1.0, max(0.0, face_v)),
                               mod_scores={"face": face_v, "speech": 0.5, "text": 0.5}))
    abl2 = ablation(ds, full, {"P_face_only_noisy": p_face_noisy})
    a2 = abl2["P_face_only_noisy"]
    # 加偏置后 CCC 应降低
    assert a2.delta_ccc < 0


# ---------- 顶层 ----------
def test_evaluate_all_returns_five():
    samples = tuple(_g(f"s{i}", 0.3) for i in range(5))
    ds = GoldDataset(name="t", samples=samples)
    outputs = tuple(_o(f"s{i}", 0.3) for i in range(5))
    evs = evaluate_all(ds, outputs)
    assert len(evs) == 5
    assert {e.module for e in evs} == {"perception", "fast_system", "slow_system",
                                       "memory", "action"}
