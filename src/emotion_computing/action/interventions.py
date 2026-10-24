"""⑤ 行动层 · 干预库。

实现需求 A-4、A-6。

设计立场（需求文档 §3）：受试者本人也是使用者。
干预的目标是"降低负荷"，不是"纠正表现"。
因此所有措辞避免评价性表达（不要说"你太紧张了"），
只描述客观观察 + 给出可选建议。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Intervention:
    id: str
    title: str
    message: str
    applicable_actions: tuple[str, ...]
    rationale_template: str


INTERVENTIONS: dict[str, Intervention] = {
    "breathing_478": Intervention(
        id="breathing_478",
        title="4-7-8 呼吸",
        message="现在做一次 4-7-8 呼吸：吸气 4 秒、屏息 7 秒、呼气 8 秒，重复两轮。",
        applicable_actions=("coach", "pause"),
        rationale_template="高唤醒信号（{signals}）通常对延长呼气最敏感。",
    ),
    "pacing": Intervention(
        id="pacing",
        title="放慢节奏",
        message="试着把语速和动作放慢一档——放慢本身往往能带动紧张度下降。",
        applicable_actions=("coach",),
        rationale_template="语音/动作节律加快（{signals}），放慢节奏通常有即时效果。",
    ),
    "task_downgrade": Intervention(
        id="task_downgrade",
        title="任务降级",
        message="考虑把当前目标拆成更小的一步，先只完成最必要的那一环。",
        applicable_actions=("micro_break", "pause"),
        rationale_template="紧张度已持续处于高位（{signals}），降低单步难度比硬撑更划算。",
    ),
    "micro_reset": Intervention(
        id="micro_reset",
        title="20 秒微休息",
        message="离开屏幕 20 秒：站起来、伸展一下、看向远处。回来再继续。",
        applicable_actions=("micro_break",),
        rationale_template="连续处于关注区间（{signals}），短暂抽离通常能重置状态。",
    ),
    "reframe": Intervention(
        id="reframe",
        title="认知重构",
        message="把注意力从「我做不完」换到「我现在只做下一步」——只处理眼前这一步。",
        applicable_actions=("coach", "pause"),
        rationale_template="语义信号以负向自我评价为主（{signals}）。",
    ),
    "encourage": Intervention(
        id="encourage",
        title="正向确认",
        message="紧张度正在回落，你刚才的处理方式有效，继续保持。",
        applicable_actions=("continue",),
        rationale_template="紧张度已从峰值回落（{signals}），此时确认有效策略有助于巩固。",
    ),
}


def pick_intervention(
    action: str,
    signals: Sequence[str],
    effectiveness: Mapping[str, float] | None = None,
) -> Intervention | None:
    """为给定动作挑一条干预。

    若历史有效性能数据（记忆层回写）可用，则在适用的候选中优先选择
    历史上更有效的干预（需求 A-5 的闭环）。
    """
    candidates = [
        iv for iv in INTERVENTIONS.values() if action in iv.applicable_actions
    ]
    if not candidates:
        return None

    score_map = dict(effectiveness or {})
    # 稳定排序：效果分降序，id 升序（保证可复现）
    candidates.sort(key=lambda iv: (-score_map.get(iv.id, 0.0), iv.id))
    return candidates[0]
