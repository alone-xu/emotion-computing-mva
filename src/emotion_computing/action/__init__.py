"""⑤ 行动层：分级干预决策。

实现需求 A-1 ~ A-6。

要点：
  * 默认**静默运行**，仅在必要时才建议动作；
  * 迟滞（冷却 + 上下阈值分离）防止抖动式反复提示；
  * 每次决策都必须给出理由（rationale）。
"""

from __future__ import annotations

from .interventions import INTERVENTIONS, pick_intervention
from .policy import ActionPolicy

__all__ = ["ActionPolicy", "INTERVENTIONS", "pick_intervention"]
