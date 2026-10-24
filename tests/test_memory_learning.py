"""④ 记忆层 · "学得会"的回归测试。

对应评审 P0 的四条与 P1-#7：
  * 阈值真的被学习并写回（含可解释依据）；
  * baseline_sigma 不再永远是 0；
  * 基线是 EMA 平滑，不是单次覆盖；
  * 空会话不计入画像；
  * 记忆命中会改变行动层决策，且写进 rationale（不是只拿去写报告）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from emotion_computing.action import ActionPolicy
from emotion_computing.config import Config
from emotion_computing.contracts import CapacityAssessment, Forecast, StressSample
from emotion_computing.memory import EpisodeStore, summarize
from emotion_computing.pipeline import SessionPipeline, _percentile


# --------------------------------------------------------------------------
# 脚手架
# --------------------------------------------------------------------------


def _pipeline(tmp_path: Path, task: str = "t") -> SessionPipeline:
    cfg = Config(baseline_seconds=1.0, checkpoint_interval=60.0)
    return SessionPipeline(config=cfg, data_dir=str(tmp_path), task_label=task)


def _sample(t: float = 0.0, value: float = 0.5) -> dict:
    return {"t": t, "raw": value, "smoothed": value, "confidence": 1.0,
            "contributions": {"face": 1.0}}


def _seed_episode(store: EpisodeStore, peak: float, task: str = "t", n: int = 40):
    ep = store.new_episode(task)
    ep.baseline_mu = 0.20
    ep.baseline_sigma = 0.02
    ep.samples = [
        {"t": float(i), "raw": 0.2,
         "smoothed": 0.20 + (peak - 0.20) * (i / max(n - 1, 1)),
         "confidence": 0.8, "contributions": {"face": 0.5, "speech": 0.5}}
        for i in range(n)
    ]
    ep.summary = summarize(ep)
    store.append_episode(ep)
    return ep


def _spike_episode(store: EpisodeStore, recovery_s: int, task: str = "t"):
    """构造一个有明确恢复时间的会话（峰值后线性回落）。"""
    ep = store.new_episode(task)
    ep.baseline_mu = 0.20
    ep.baseline_sigma = 0.02
    samples = []
    for i in range(11):  # 0..10s 升到 0.8
        samples.append({"t": float(i), "raw": 0.2, "smoothed": 0.2 + 0.06 * i,
                        "confidence": 0.9, "contributions": {"face": 1.0}})
    for i in range(1, recovery_s + 1):  # 然后回落
        value = 0.8 - 0.6 * (i / recovery_s)
        samples.append({"t": float(10 + i), "raw": value, "smoothed": value,
                        "confidence": 0.9, "contributions": {"face": 1.0}})
    ep.samples = samples
    ep.summary = summarize(ep)
    store.append_episode(ep)
    return ep


def assessment(state: str, point: float = 0.6, slope: float = 0.3) -> CapacityAssessment:
    forecast = Forecast(horizon_s=60.0, point=point, lower=point - 0.05,
                        upper=point + 0.05, slope=slope, baseline=0.3,
                        method="test", validated=False)
    return CapacityAssessment(state=state, score=0.4, forecast=forecast,
                              resilience=0.5, reasons=("预测值高于关注阈值",))


def stress(value: float = 0.6, t: float = 0.0) -> StressSample:
    return StressSample(timestamp=t, raw=value, smoothed=value, confidence=0.8,
                        contributions={"face": 1.0})


# --------------------------------------------------------------------------
# P0-#5 空会话不计入
# --------------------------------------------------------------------------


def test_empty_session_does_not_increment_sessions(tmp_path: Path) -> None:
    pipe = _pipeline(tmp_path)
    pipe.episode.samples = []
    pipe._update_profile()
    assert pipe.profile.sessions == 0
    pipe.store.close()


# --------------------------------------------------------------------------
# P0-#2 / P0-#3 基线：EMA + 补 sigma
# --------------------------------------------------------------------------


def test_baseline_is_ema_and_keeps_sigma(tmp_path: Path) -> None:
    pipe = _pipeline(tmp_path)

    # 第一次会话：直接采用
    pipe.episode.samples = [_sample(0.0, 0.3)]
    pipe.baseline_stress = 0.30
    pipe.episode.baseline_sigma = 0.020
    pipe._update_profile()
    assert pipe.profile.sessions == 1
    assert pipe.profile.baseline_mu == pytest.approx(0.30)
    assert pipe.profile.baseline_sigma == pytest.approx(0.020), "P0-#2：sigma 必须被写回"

    # 第二次会话异常偏高：α = 1/min(1+1,10) = 0.5，只带动一半
    pipe.episode.samples = [_sample(0.0, 0.5)]
    pipe.baseline_stress = 0.50
    pipe.episode.baseline_sigma = 0.040
    pipe._update_profile()
    assert pipe.profile.baseline_mu == pytest.approx(0.40), "单次异常会话不得直接覆盖基线"
    assert pipe.profile.baseline_sigma == pytest.approx(0.030)

    # 第三次：α = 1/3，继续收敛而不是跳变
    pipe.baseline_stress = 0.30
    pipe._update_profile()
    assert pipe.profile.baseline_mu == pytest.approx((1 - 1 / 3) * 0.40 + (1 / 3) * 0.30)
    pipe.store.close()


# --------------------------------------------------------------------------
# P0-#1 阈值学习
# --------------------------------------------------------------------------


def test_thresholds_are_learned_and_written_back(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path)
    for peak in (0.40, 0.50, 0.60, 0.70, 0.80):
        _seed_episode(store, peak=peak)
    store.close()

    pipe = _pipeline(tmp_path)
    pipe.episode.samples = [_sample(0.0, 0.5)]
    pipe.episode.summary = summarize(pipe.episode)
    pipe._update_profile()

    rows = [r for r in pipe.store.load_index() if r["task_label"] == "t"]
    means = [float(r["mean"]) for r in rows]
    expected_mid = _percentile(means, 60.0)

    assert pipe.profile.theta_mid == pytest.approx(expected_mid, abs=1e-9), "P0-#1：阈值必须被学习"
    assert pipe.profile.theta_mid != 0.55, "默认值必须被历史数据顶掉"
    assert pipe.profile.theta_hi > pipe.profile.theta_mid

    evidence = pipe.profile.threshold_evidence
    assert evidence["theta_mid"]["rule"].startswith("p60")
    assert evidence["theta_mid"]["n"] == len(means)
    assert len(evidence["theta_mid"]["inputs"]) == len(means)
    assert evidence["learned_at"]
    pipe.store.close()


def test_tau_rev_learned_from_recovery_median(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path)
    for recovery in (40, 60, 80):
        _spike_episode(store, recovery)
    store.close()

    pipe = _pipeline(tmp_path)
    pipe.episode.samples = [_sample(0.0, 0.5)]
    pipe.episode.summary = summarize(pipe.episode)
    pipe._update_profile()

    evidence = pipe.profile.threshold_evidence["tau_rev"]
    assert evidence["rule"].startswith("median"), "P1 建议：tau_rev 用恢复时间中位数 × 1.2"
    assert evidence["n"] == 3
    assert 30.0 <= pipe.profile.tau_rev <= 600.0
    assert pipe.profile.tau_rev != 90.0
    pipe.store.close()


def test_thresholds_stay_honest_with_too_little_data(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path)
    _seed_episode(store, peak=0.6)  # 只有 1 条同类会话
    store.close()

    pipe = _pipeline(tmp_path)
    pipe.episode.samples = [_sample(0.0, 0.5)]
    pipe.episode.summary = summarize(pipe.episode)
    pipe._update_profile()

    evidence = pipe.profile.threshold_evidence
    assert evidence["theta_mid"]["rule"].startswith("insufficient_data")
    assert pipe.profile.theta_mid == pytest.approx(0.55)
    pipe.store.close()


# --------------------------------------------------------------------------
# P1-#7 记忆回流到决策
# --------------------------------------------------------------------------


def test_history_prior_escalates_watch_earlier() -> None:
    policy = ActionPolicy(
        Config(t_watch=60.0),
        history_prior={"sessions": 4, "pause_rate": 0.75, "task_label": "t"},
    )
    decision = policy.decide(stress(0.6), assessment("WATCH"), now=0.0)
    assert decision.action == "micro_break", "同类任务历史风险高时应提前升级"
    assert "记忆" in decision.rationale


def test_memory_hint_escalates_and_is_cited() -> None:
    policy = ActionPolicy(Config(t_watch=60.0))
    hint = {"episode_id": "abc123", "similarity": 0.8, "max": 0.9, "distance": 0.7}
    decision = policy.decide(stress(0.6), assessment("WATCH"), now=0.0, memory_hint=hint)
    assert decision.action == "micro_break"
    assert "abc123" in decision.rationale, "必须能说清是被哪条历史影响的"


def test_low_similarity_hint_does_not_escalate() -> None:
    policy = ActionPolicy(Config(t_watch=60.0))
    hint = {"episode_id": "far", "similarity": 0.2, "max": 0.95, "distance": 1.4}
    decision = policy.decide(stress(0.6), assessment("WATCH"), now=0.0, memory_hint=hint)
    assert decision.action == "coach"


def test_memory_does_not_bypass_cooldown() -> None:
    policy = ActionPolicy(Config(), history_prior={"sessions": 3, "pause_rate": 1.0})
    first = policy.decide(stress(0.6, 0.0), assessment("WATCH"), now=0.0)
    assert first.action == "micro_break"
    second = policy.decide(stress(0.6, 10.0), assessment("WATCH"), now=10.0)
    assert second.action == "continue", "记忆不得绕过迟滞"


def test_pipeline_retrieves_partial_situation_from_history(tmp_path: Path) -> None:
    pipe = _pipeline(tmp_path)
    for t in range(0, 120):
        pipe.step({"t": float(t)})
    assert pipe.episode.samples, "先攒下真实样本"

    # 用当前会话的样本造一条"历史"，保证与当前情境完全相似
    hist = pipe.store.new_episode("t")
    hist.samples = list(pipe.episode.samples)
    hist.baseline_mu = pipe.episode.baseline_mu
    hist.summary = summarize(hist)
    pipe.store.append_episode(hist)
    pipe.index_rows = pipe.store.load_index()

    hint = pipe._memory_hint(120.0)
    assert hint is not None, "当前情境应能检索到刚写入的相似历史"
    assert hint["episode_id"] == hist.episode_id
    assert hint["similarity"] >= 0.5
    pipe.store.close()
