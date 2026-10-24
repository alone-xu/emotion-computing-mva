"""审核子包：MVA-Audit 流水线（详见 docs/03-audit-requirements.md）。

公开 API（沿用架构文档 §2 契约优先原则）：
  * 数据契约：见 ``emotion_computing.contracts``
  * 评价指标：``metrics``（CCC / RMSE / F1 / Krippendorff α / …）
  * 数据层审核：``data_audit``（D-1 / D-2 / D-3）
  * 模块评估：``module_evaluator``（M-1 / M-2）
  * 模态诊断：``modality_diagnostic``（M-3）
  * 失败归因：``failure_attribution``（F-1 / F-2 / F-3）
  * 审核日志：``log``（L-1）
  * 报告与可视化：``report`` / ``viz``
  * 编排：``runner.AuditRunner``
"""

from __future__ import annotations

from .metrics import (
    bias,
    ccc,
    cohens_kappa,
    confusion_matrix,
    error_threshold,
    krippendorff_alpha_interval,
    krippendorff_alpha_nominal,
    mae,
    macro_f1,
    metric_delta,
    miss_rate,
    precision_at_k,
    precision_recall_f1,
    repair_priority,
    rmse,
)
from .runner import AuditRunner, AuditRunnerConfig, build_default_runner
from .log import AuditLogStore, log_action

__all__ = [
    # metrics
    "ccc", "rmse", "mae", "bias",
    "confusion_matrix", "precision_recall_f1", "macro_f1", "miss_rate",
    "precision_at_k",
    "cohens_kappa", "krippendorff_alpha_nominal", "krippendorff_alpha_interval",
    "repair_priority", "error_threshold", "metric_delta",
    # runner
    "AuditRunner", "AuditRunnerConfig", "build_default_runner",
    # log
    "AuditLogStore", "log_action",
]
