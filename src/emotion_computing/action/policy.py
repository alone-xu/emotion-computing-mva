"""⑤ 行动层 · 分级决策与迟滞。

实现需求 A-1 ~ A-4、A-6。

迟滞的两个机制（架构文档 §5.5.1）：
  1. **冷却时间**：同一动作在 cooldown 内不重复输出；
  2. **阈值分离**：由慢系统状态机承担（theta_mid/theta_hi 之间留有带宽）。

被冷却抑制时返回 ``continue``，并在 rationale 中说明"已在冷却期"，
这样"系统为什么没说话"同样可追溯。

记忆回流（评审 P1-#7）
----------------------
决策会读取两路历史证据，并把"因为哪条历史"写进 rationale：

  * ``history_prior``：会话开始时算好的同类任务先验（历史上有多大比例冲进 PAUSE）；
  * ``memory_hint``：当前情境检索到的**最相似历史会话**（由 pipeline 每个慢周期传入）。

历史风险高时，WATCH 阶段会更早升级为 micro_break —— 但**不会**绕过冷却、
也不会凭空把状态改成 PAUSE：阈值永远由慢系统说了算，记忆只影响"多快出手"。
"""

from __future__ import annotations

from typing import Any

from ..contracts import ActionDecision, CapacityAssessment, StressSample
from .interventions import pick_intervention


class ActionPolicy:
    def __init__(self, config: Any, profile: Any = None, history_prior: dict | None = None) -> None:
        self.config = config
        self.profile = profile
        self.history_prior = dict(history_prior or {})
        self._last_action_at: dict[str, float] = {}
        self._watch_since: float | None = None
        self._last_action: str = "continue"
        self._last_intervention_id: str | None = None

    # ------------------------------------------------------------------
    def _cooldown_for(self, action: str) -> float:
        cfg = self.config
        return {
            "coach": cfg.cooldown_coach,
            "micro_break": cfg.cooldown_micro_break,
            "pause": cfg.cooldown_pause,
        }.get(action, 0.0)

    def _in_cooldown(self, action: str, now: float) -> bool:
        last = self._last_action_at.get(action)
        if last is None:
            return False
        return (now - last) < self._cooldown_for(action)

    def _effectiveness(self) -> dict[str, float]:
        if self.profile is None:
            return {}
        return dict(getattr(self.profile, "intervention_effectiveness", {}) or {})

    # ------------------------------------------------------------------
    def _memory_risk(self, memory_hint: dict | None) -> str | None:
        """当前情境命中了"进过高位"的历史会话时，返回一句可追溯的说明。"""
        if not memory_hint:
            return None
        cfg = self.config
        similarity = float(memory_hint.get("similarity", 0.0) or 0.0)
        peak = float(memory_hint.get("max", 0.0) or 0.0)
        if similarity < 0.5 or peak < float(cfg.theta_hi):
            return None
        return (
            f"历史相似情境 {memory_hint.get('episode_id', '?')} 峰值 {peak:.2f}"
            f"（相似度 {similarity:.2f}、距离 {float(memory_hint.get('distance', 0.0) or 0.0):.2f}）"
            "当时进入过高位，故提前一档介入"
        )

    def _prior_risk(self) -> str | None:
        """同类任务的历史先验：多数会话都冲进过 PAUSE 时，缩短 WATCH 观察期。"""
        prior = self.history_prior or {}
        sessions = int(prior.get("sessions", 0) or 0)
        pause_rate = float(prior.get("pause_rate", 0.0) or 0.0)
        if sessions < 2 or pause_rate < 0.5:
            return None
        return (
            f"同类任务历史 {sessions} 次中 {pause_rate * 100:.0f}% 曾进入高位（PAUSE 区间），"
            "WATCH 观察期按 60% 折算"
        )

    # ------------------------------------------------------------------
    def decide(
        self,
        sample: StressSample,
        assessment: CapacityAssessment,
        now: float | None = None,
        memory_hint: dict | None = None,
    ) -> ActionDecision:
        cfg = self.config
        now = float(now if now is not None else sample.timestamp)
        state = assessment.state
        memory_note = self._memory_risk(memory_hint)
        prior_note = self._prior_risk()

        # ---- 跟踪 WATCH 持续时长 ----
        if state == "WATCH":
            if self._watch_since is None:
                self._watch_since = now
        else:
            self._watch_since = None

        # ---- 候选动作 ----
        if state == "PAUSE":
            candidate = "pause"
        elif state == "WATCH":
            watched = (now - self._watch_since) if self._watch_since is not None else 0.0
            # 历史风险高 → WATCH 观察期缩短；命中了"进过高位"的相似情境 → 当场升级
            watch_limit = cfg.t_watch * (0.6 if (prior_note or memory_note) else 1.0)
            escalate = watched >= watch_limit or bool(memory_note) or bool(prior_note)
            candidate = "micro_break" if escalate else "coach"
        else:
            candidate = "continue"

        # ---- 迟滞：冷却抑制 ----
        if candidate != "continue" and self._in_cooldown(candidate, now):
            last = self._last_action_at.get(candidate, now)
            reason = (
                f"状态为 {state}，但动作 {candidate} 仍在冷却期"
                f"（剩余 {self._cooldown_for(candidate) - (now - last):.0f}s），本轮不重复提示。"
            )
            if memory_note:
                reason += f"（记忆提示：{memory_note}）"
            return ActionDecision(
                timestamp=now,
                action="continue",
                intervention=None,
                message="",
                rationale=reason,
                urgent=False,
            )

        # ---- 组装决策 ----
        if candidate == "continue":
            # 仅在"刚从高位回落"时给一次正向确认
            recovering = (
                self._last_action in ("coach", "micro_break", "pause")
                and sample.smoothed < cfg.theta_mid * 0.8
            )
            if recovering:
                intervention = pick_intervention(
                    "continue",
                    [f"{sample.smoothed:.2f}"],
                    self._effectiveness(),
                )
                self._last_action_at["continue"] = now
                self._last_action = "continue"
                self._last_intervention_id = intervention.id if intervention else None
                return ActionDecision(
                    timestamp=now,
                    action="continue",
                    intervention=intervention.id if intervention else None,
                    message=intervention.message if intervention else "",
                    rationale="紧张度已回落至关注阈值以下（需求 A-6：默认静默，仅在此时给予确认）。",
                    urgent=False,
                )
            rationale = f"状态 {state}：紧张度 {sample.smoothed:.2f} 未见恶化。"
            if memory_note:
                rationale += f"（记忆提示：{memory_note}）"
            return ActionDecision(
                timestamp=now,
                action="continue",
                intervention=None,
                message="",
                rationale=rationale,
                urgent=False,
            )

        signals = [f"{sample.smoothed:.2f}", *assessment.reasons[:2]]
        intervention = pick_intervention(candidate, signals, self._effectiveness())

        rationale = "；".join(assessment.reasons[:3])
        if memory_note:
            rationale += f"；【记忆】{memory_note}"
        elif prior_note:
            rationale += f"；【记忆】{prior_note}"
        message = intervention.message if intervention else ""
        if intervention:
            rationale = (
                f"{rationale}　→ 选择干预「{intervention.title}」："
                + intervention.rationale_template.format(signals=", ".join(signals[:2]))
            )

        self._last_action_at[candidate] = now
        self._last_action = candidate
        self._last_intervention_id = intervention.id if intervention else None

        return ActionDecision(
            timestamp=now,
            action=candidate,
            intervention=intervention.id if intervention else None,
            message=message,
            rationale=rationale,
            urgent=(candidate == "pause"),
        )
