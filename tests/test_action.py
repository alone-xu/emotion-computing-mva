"""⑤ 行动层 · 分级决策与迟滞测试。"""

from __future__ import annotations

from emotion_computing.action import ActionPolicy
from emotion_computing.config import Config
from emotion_computing.contracts import ACTION_ORDER, CapacityAssessment, Forecast, StressSample


def assessment(state: str, point: float = 0.8, slope: float = 0.3) -> CapacityAssessment:
    forecast = Forecast(
        horizon_s=60.0,
        point=point,
        lower=max(0.0, point - 0.05),
        upper=min(1.0, point + 0.05),
        slope=slope,
        baseline=0.3,
        method="test",
        validated=False,
    )
    return CapacityAssessment(
        state=state, score=0.3, forecast=forecast, resilience=0.5,
        reasons=("预测 60s 后紧张度 0.80 超过阈值 0.75", "当前已连续 25s 高于 0.75"),
    )


def stress(t: float = 0.0, value: float = 0.9) -> StressSample:
    return StressSample(
        timestamp=t, raw=value, smoothed=value, confidence=0.8,
        contributions={"face": 1.0},
    )


def test_pause_fires_on_pause_state() -> None:
    policy = ActionPolicy(Config())
    decision = policy.decide(stress(0.0), assessment("PAUSE"), now=0.0)
    assert decision.action == "pause"
    assert decision.urgent is True
    assert decision.rationale
    assert decision.message
    assert decision.intervention


def test_ok_state_stays_silent() -> None:
    policy = ActionPolicy(Config())
    decision = policy.decide(stress(0.0, 0.2), assessment("OK", point=0.2, slope=-0.1), now=0.0)
    assert decision.action == "continue"
    assert decision.intervention is None
    assert decision.message == ""


def test_watch_escalates_after_duration() -> None:
    cfg = Config(t_watch=60.0)
    policy = ActionPolicy(cfg)
    first = policy.decide(stress(0.0, 0.6), assessment("WATCH", 0.6), now=0.0)
    assert first.action == "coach"

    # 超过 t_watch 后应升级为 micro_break
    later = policy.decide(stress(200.0, 0.6), assessment("WATCH", 0.6), now=200.0)
    assert later.action == "micro_break"


def test_cooldown_suppresses_repeat_action() -> None:
    """需求 A-3：迟滞必须防止抖动式反复提示。"""
    policy = ActionPolicy(Config())
    first = policy.decide(stress(0.0), assessment("PAUSE"), now=0.0)
    assert first.action == "pause"

    second = policy.decide(stress(10.0), assessment("PAUSE"), now=10.0)
    assert second.action == "continue"
    assert "冷却" in second.rationale


def test_action_reappears_after_cooldown() -> None:
    cfg = Config(cooldown_pause=90.0)
    policy = ActionPolicy(cfg)
    policy.decide(stress(0.0), assessment("PAUSE"), now=0.0)
    later = policy.decide(stress(200.0), assessment("PAUSE"), now=200.0)
    assert later.action == "pause"


def test_all_actions_are_in_known_set() -> None:
    cfg = Config()
    policy = ActionPolicy(cfg)
    seen = {
        policy.decide(stress(0.0, 0.2), assessment("OK", 0.2, -0.1), now=0.0).action,
        policy.decide(stress(1.0, 0.6), assessment("WATCH", 0.6), now=1.0).action,
        policy.decide(stress(2.0, 0.9), assessment("PAUSE", 0.9), now=2.0).action,
    }
    assert seen <= set(ACTION_ORDER)


def test_watch_resets_when_state_returns_to_ok() -> None:
    cfg = Config(t_watch=60.0)
    policy = ActionPolicy(cfg)
    policy.decide(stress(0.0, 0.6), assessment("WATCH", 0.6), now=0.0)
    # 回到 OK，计时器应重置
    policy.decide(stress(30.0, 0.2), assessment("OK", 0.2, -0.1), now=30.0)
    again = policy.decide(stress(100.0, 0.6), assessment("WATCH", 0.6), now=100.0)
    assert again.action == "coach", "重置后不应立刻升级为 micro_break"


def test_intervention_effectiveness_influences_choice() -> None:
    from emotion_computing.action import pick_intervention

    # coach 场景下有两个候选：breathing_478 与 pacing 与 reframe
    plain = pick_intervention("coach", ["0.7"])
    assert plain is not None

    boosted = pick_intervention("coach", ["0.7"], {"pacing": 5.0})
    assert boosted is not None
    assert boosted.id == "pacing", "历史有效的干预应被优先选择"
