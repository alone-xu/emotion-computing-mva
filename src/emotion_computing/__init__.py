"""多模态情感计算系统（MVA）。

五层顺序：感知 → 快系统 → 轻量慢系统 → 记忆 → 行动。

诚实性约定（见 docs/01-requirements.md §7.2）：
    本系统输出的紧张度是**个体内的相对量**，未经真实数据校准。
    所有与准确率相关的结论都由 UNVALIDATED 常量显式标记。
"""

from __future__ import annotations

__version__ = "0.1.0"

#: 全局诚实标记：当前版本的紧张度估计与预测**未经真实数据验证**。
#: 任何声称准确率的表述，都必须先解除该标记（见需求文档 §7.2）。
UNVALIDATED: bool = True

__all__ = ["__version__", "UNVALIDATED"]
