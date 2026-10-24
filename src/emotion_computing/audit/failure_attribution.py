"""F-# 失败归因（03-audit-requirements.md §4）。

7 类失败（FAILURE_TYPES）：
  perception_failure / modality_sabotage / slow_eval_failure
  memory_interference / action_failure / data_problem / combined_failure

归因操作流程（F-2）按顺序排查：
  1. 金标准本身有问题？        → data_problem
  2. M-3 已识别出破坏者？     → modality_sabotage
  3. 某模态独立预测就是错的？   → perception_failure
  4. 慢系统状态机判断错？      → slow_eval_failure
  5. 记忆检索干扰？            → memory_interference
  6. 干预策略不当？            → action_failure
  7. 兜底：组合失败            → combined_failure
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..contracts import (
    FAILURE_REPAIR,
    FAILURE_TYPES,
    FailureCase,
    FailureDistribution,
    GoldLabel,
    ModalityAttribution,
)
from .metrics import error_threshold


@dataclass
class AttributionEvidence:
    """每个错误样本的归因证据。"""

    sample_id: str
    gold_label: GoldLabel
    fused: float
    mva_state: str = "OK"
    mva_action: str = "continue"
    memory_retrieved: tuple[str, ...] = ()
    memory_relevant: tuple[bool, ...] = ()        # 人工判断 top-3 是否相关
    data_suspect: bool = False                     # 审核者认为金标准可能错
    modality_attribution: ModalityAttribution | None = None
    extra: dict = field(default_factory=dict)


def _has_bad_modality(att: ModalityAttribution | None, theta: float) -> bool:
    """F-2 步骤 3：某模态独立预测就是错的（即与 gold 差距大）。"""
    if att is None:
        return False
    gold = att.gold
    for v in (att.face, att.speech, att.text):
        if abs(v - gold) > theta:
            return True
    return False


def _state_wrong(gold_state: str, mva_state: str) -> bool:
    """F-2 步骤 4：慢系统状态机错（用 OK/WATCH/PAUSE 等级判定）。"""
    order = {"OK": 0, "WATCH": 1, "PAUSE": 2}
    a = order.get(mva_state, 0)
    b = order.get(gold_state, 0)
    return abs(a - b) >= 2  # 跨级判错（OK vs PAUSE 视为失败）


def _memory_interference(retrieved: tuple[str, ...],
                         relevant: tuple[bool, ...],
                         threshold: float = 0.5) -> bool:
    """F-2 步骤 5：检索到的 top-K 中相关比例 < threshold 视为干扰。"""
    if not retrieved or not relevant:
        return False
    k = min(len(retrieved), len(relevant))
    rel = sum(1 for r in relevant[:k] if r)
    return (rel / k) < threshold


def attribute_one(ev: AttributionEvidence,
                  error_theta: float = 0.20,
                  modality_error_theta: float = 0.25) -> FailureCase:
    """对单条证据做归因（F-2 顺序）。"""
    err = abs(ev.fused - ev.gold_label.stress)
    if err <= error_theta:
        # 非错误样本；返回 system 归因为空
        return FailureCase(
            sample_id=ev.sample_id,
            error=err,
            system_attribution="ok",
            modality_attribution=ev.modality_attribution,
            evidence={"fused": ev.fused, "gold": ev.gold_label.stress},
        )

    # F-2 步骤 1：金标准问题
    if ev.data_suspect:
        return FailureCase(
            sample_id=ev.sample_id,
            error=err,
            system_attribution="data_problem",
            modality_attribution=ev.modality_attribution,
            evidence={"reason": "审核者标记金标准可疑"},
        )

    # F-2 步骤 2：M-3 识别到破坏者
    if ev.modality_attribution and ev.modality_attribution.saboteur:
        return FailureCase(
            sample_id=ev.sample_id,
            error=err,
            system_attribution="modality_sabotage",
            modality_attribution=ev.modality_attribution,
            evidence={"saboteur": ev.modality_attribution.saboteur,
                      "reason": ev.modality_attribution.reason},
        )

    # F-2 步骤 3：模态独立预测错
    if _has_bad_modality(ev.modality_attribution, modality_error_theta):
        return FailureCase(
            sample_id=ev.sample_id,
            error=err,
            system_attribution="perception_failure",
            modality_attribution=ev.modality_attribution,
            evidence={"reason": "某模态独立预测与 gold 差距大"},
        )

    # F-2 步骤 4：慢系统状态机跨级错
    if _state_wrong(ev.gold_label.state, ev.mva_state):
        return FailureCase(
            sample_id=ev.sample_id,
            error=err,
            system_attribution="slow_eval_failure",
            modality_attribution=ev.modality_attribution,
            evidence={"gold_state": ev.gold_label.state, "mva_state": ev.mva_state},
        )

    # F-2 步骤 5：记忆干扰
    if _memory_interference(ev.memory_retrieved, ev.memory_relevant):
        return FailureCase(
            sample_id=ev.sample_id,
            error=err,
            system_attribution="memory_interference",
            modality_attribution=ev.modality_attribution,
            evidence={"retrieved": list(ev.memory_retrieved),
                      "relevant": list(ev.memory_relevant)},
        )

    # F-2 步骤 6：策略失败
    # 启发式：action=continue 但 gold.state=PAUSE → 严重欠报 → strategy fail
    if ev.gold_label.state == "PAUSE" and ev.mva_action in ("continue", "coach"):
        return FailureCase(
            sample_id=ev.sample_id,
            error=err,
            system_attribution="action_failure",
            modality_attribution=ev.modality_attribution,
            evidence={"reason": f"PAUSE 状态但行动 = {ev.mva_action}"},
        )

    # F-2 步骤 7：兜底
    return FailureCase(
        sample_id=ev.sample_id,
        error=err,
        system_attribution="combined_failure",
        modality_attribution=ev.modality_attribution,
        evidence={"reason": "各模块单独看都合理但组合结果错"},
    )


def attribute_batch(
    evidence_list: Sequence[AttributionEvidence],
    error_theta: float = 0.20,
    modality_error_theta: float = 0.25,
) -> list[FailureCase]:
    """批量归因。"""
    return [attribute_one(e, error_theta, modality_error_theta) for e in evidence_list]


def build_distribution(
    cases: Sequence[FailureCase],
    feasibility: dict[str, str] | None = None,
) -> FailureDistribution:
    """F-3 量化归因：失败分布 + 推荐修复。"""
    feas = feasibility or {
        "modality_sabotage": "easy",
        "data_problem":      "easy",
        "perception_failure": "hard",
        "combined_failure":  "hard",
        "slow_eval_failure": "hard",
        "memory_interference": "easy",
        "action_failure":    "hard",
    }
    relevant = [c for c in cases if c.system_attribution != "ok"]
    counter: Counter[str] = Counter(c.system_attribution for c in relevant)
    total = sum(counter.values())
    pct: dict[str, float] = {k: (counter.get(k, 0) / total * 100 if total else 0.0)
                              for k in FAILURE_TYPES}
    # top_repair：按 03 §I-1 "高占比+易修复优先"
    from .metrics import repair_priority
    prio = repair_priority(pct, feas)
    top_repair = tuple(f"{t} → {FAILURE_REPAIR.get(t, '?')}" for t, _, _ in prio[:3])
    return FailureDistribution(
        counts={k: counter.get(k, 0) for k in FAILURE_TYPES},
        percentages={k: round(pct.get(k, 0.0), 2) for k in FAILURE_TYPES},
        top_repair=top_repair,
    )
