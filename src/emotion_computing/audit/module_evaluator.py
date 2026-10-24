"""M-1 / M-2 模块级独立评估（03-audit-requirements.md §3）。

设计要点：
  * 接收 ``GoldDataset`` 与 MVA 业务输出（每样本的 stress/state/action/…）
  * 逐模块独立计算指标（不混合），输出 ModuleEvaluation
  * 消融实验（M-2）通过 ``AblationResult`` 数据类承载
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..contracts import (
    FAILURE_TYPES,
    GoldDataset,
    GoldLabel,
    ModuleEvaluation,
    ModuleMetric,
)
from . import metrics


# --------------------------------------------------------------------------
# 阈值与档位（03 §M-1）
# --------------------------------------------------------------------------

DEFAULT_THRESHOLDS: dict[str, dict[str, tuple[float, str, str]]] = {
    # module -> {metric: (ok_below_or_above, ok_status, fail_status)}
    # value > threshold ⇒ ok，< threshold ⇒ fail（CCC 越大约好）
    "perception": {
        "ccc":       (0.5, "ok", "fail"),     # 模态 CCC ≥ 0.5 才 ok
        "mae":       (0.20, "fail", "ok"),    # MAE 越低越好
    },
    "fast_system": {
        "ccc":       (0.5, "ok", "fail"),
        "rmse":      (0.25, "fail", "ok"),
    },
    "slow_system": {
        "f1_watch_pause": (0.5, "ok", "fail"),
        "miss_rate":      (0.30, "fail", "ok"),
    },
    "memory": {
        "precision_at_3": (0.5, "ok", "fail"),
    },
    "action": {
        "mean_score":      (3.0, "ok", "fail"),
    },
}


# --------------------------------------------------------------------------
# 输入契约：每条样本的 MVA 业务输出
# --------------------------------------------------------------------------


@dataclass
class MVASampleOutput:
    """一条样本的 MVA 业务输出。"""

    sample_id: str
    stress: float                       # fused smoothed stress
    state: str = "OK"                   # 慢系统输出状态
    action: str = "continue"            # 行动层输出
    modality_scores: dict[str, float] = field(default_factory=dict)  # 各模态独立 score（可选）
    modality_confidence: dict[str, float] = field(default_factory=dict)
    retrieved_memory: tuple[str, ...] = ()
    memory_relevant: tuple[bool, ...] = ()


@dataclass
class AblationResult:
    name: str                # "P" | "F" | "S" | "M" | "A"
    description: str
    ccc: float
    rmse: float
    mae: float
    n_samples: int
    delta_ccc: float = 0.0   # 相对完整模型的差
    delta_rmse: float = 0.0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "ccc": round(self.ccc, 4),
            "rmse": round(self.rmse, 4),
            "mae": round(self.mae, 4),
            "n_samples": self.n_samples,
            "delta_ccc": round(self.delta_ccc, 4),
            "delta_rmse": round(self.delta_rmse, 4),
        }


# --------------------------------------------------------------------------
# ① 感知层：模态独立 CCC/MAE
# --------------------------------------------------------------------------


def evaluate_perception(
    dataset: GoldDataset,
    outputs: Sequence[MVASampleOutput],
) -> ModuleEvaluation:
    by_modality: dict[str, tuple[list[float], list[float]]] = {}
    for sample, out in zip(dataset.samples, outputs):
        for m, v in sample.modality_labels.items():
            if m not in out.modality_scores:
                continue
            true, pred = by_modality.setdefault(m, ([], []))
            true.append(float(v))
            pred.append(float(out.modality_scores[m]))
    ev_metrics: list[ModuleMetric] = []
    failures: list[str] = []
    for m, (t, p) in by_modality.items():
        ccc_v = metrics.ccc(t, p)
        mae_v = metrics.mae(t, p)
        # CCC < 0.5 视为该模态失败
        ccc_status = _status(ccc_v, 0.5, ">")
        mae_status = _status(mae_v, 0.20, "<")
        m_status = _combine(ccc_status, mae_status)
        ev_metrics.append(ModuleMetric(
            module=f"perception.{m}", metric_name="ccc",
            value=ccc_v, n=len(t), threshold=0.5, status=ccc_status,
        ))
        ev_metrics.append(ModuleMetric(
            module=f"perception.{m}", metric_name="mae",
            value=mae_v, n=len(t), threshold=0.20, status=mae_status,
        ))
        if m_status == "fail":
            failures.extend(_aligned_ids(dataset, outputs, m))
    return ModuleEvaluation(
        module="perception", metrics=tuple(ev_metrics), sample_failures=tuple(failures)
    )


def _aligned_ids(
    dataset: GoldDataset,
    outputs: Sequence[MVASampleOutput],
    modality: str,
    theta: float = 0.25,
) -> list[str]:
    """返回 |modality_score - gold| > theta 的样本 ID。"""
    out: list[str] = []
    for s, o in zip(dataset.samples, outputs):
        gold_v = s.modality_labels.get(modality)
        pred_v = o.modality_scores.get(modality)
        if gold_v is None or pred_v is None:
            continue
        if abs(float(gold_v) - float(pred_v)) > theta:
            out.append(s.sample_id)
    return out


# --------------------------------------------------------------------------
# ② 快系统：CCC / RMSE
# --------------------------------------------------------------------------


def evaluate_fast_system(
    dataset: GoldDataset,
    outputs: Sequence[MVASampleOutput],
) -> ModuleEvaluation:
    y_true = [s.stress for s in dataset.samples]
    y_pred = [o.stress for o in outputs]
    ccc_v = metrics.ccc(y_true, y_pred)
    rmse_v = metrics.rmse(y_true, y_pred)
    mae_v = metrics.mae(y_true, y_pred)
    ccc_status = _status(ccc_v, 0.5, ">")
    rmse_status = _status(rmse_v, 0.25, "<")
    failures = [s.sample_id for s, o, err
                in zip(dataset.samples, outputs, metrics.error_threshold(y_true, y_pred, 0.20))
                if err]
    return ModuleEvaluation(
        module="fast_system",
        metrics=(
            ModuleMetric("fast_system", "ccc", ccc_v, len(y_true), 0.5, ccc_status),
            ModuleMetric("fast_system", "rmse", rmse_v, len(y_true), 0.25, rmse_status),
            ModuleMetric("fast_system", "mae", mae_v, len(y_true), 0.20, _status(mae_v, 0.20, "<")),
        ),
        sample_failures=tuple(failures),
    )


# --------------------------------------------------------------------------
# ③ 慢系统：F1 / 漏报率
# --------------------------------------------------------------------------


def evaluate_slow_system(
    dataset: GoldDataset,
    outputs: Sequence[MVASampleOutput],
    labels: Sequence[str] = ("OK", "WATCH", "PAUSE"),
) -> ModuleEvaluation:
    y_true = [s.state for s in dataset.samples]
    y_pred = [o.state for o in outputs]
    prf = metrics.precision_recall_f1(y_true, y_pred, list(labels))
    f1_w_p = (prf.get("WATCH", {}).get("f1", 0.0) + prf.get("PAUSE", {}).get("f1", 0.0)) / 2
    miss = metrics.miss_rate(y_true, y_pred, pos_labels=["WATCH", "PAUSE"])
    f1_status = _status(f1_w_p, 0.5, ">")
    miss_status = _status(miss, 0.30, "<")
    failures = [s.sample_id for s, o in zip(dataset.samples, outputs)
                if s.state != o.state]
    ev_metrics = (
        ModuleMetric("slow_system", "f1_watch_pause", f1_w_p, len(y_true), 0.5, f1_status),
        ModuleMetric("slow_system", "miss_rate", miss, len(y_true), 0.30, miss_status),
    )
    return ModuleEvaluation(
        module="slow_system",
        metrics=ev_metrics,
        sample_failures=tuple(failures),
    )


# --------------------------------------------------------------------------
# ④ 记忆层：P@3
# --------------------------------------------------------------------------


def evaluate_memory(
    dataset: GoldDataset,
    outputs: Sequence[MVASampleOutput],
    k: int = 3,
) -> ModuleEvaluation:
    retrieved: list[list[int]] = []
    for o in outputs:
        if not o.memory_relevant:
            retrieved.append([1])    # 没数据时按"全相关"占位，避免无限除零
        else:
            retrieved.append([1 if r else 0 for r in o.memory_relevant])
    p_at_k = metrics.precision_at_k(retrieved, k)
    status = _status(p_at_k, 0.5, ">")
    return ModuleEvaluation(
        module="memory",
        metrics=(
            ModuleMetric("memory", f"precision_at_{k}", p_at_k, len(outputs), 0.5, status),
        ),
    )


# --------------------------------------------------------------------------
# ⑤ 行动层：人工评分均值
# --------------------------------------------------------------------------


def evaluate_action(
    dataset: GoldDataset,
    outputs: Sequence[MVASampleOutput],
) -> ModuleEvaluation:
    scores = [s.action_appropriateness for s in dataset.samples if s.action_appropriateness]
    if not scores:
        return ModuleEvaluation(
            module="action",
            metrics=(ModuleMetric("action", "mean_score", 0.0, 0, 3.0, "unverified"),),
        )
    mean = sum(scores) / len(scores)
    return ModuleEvaluation(
        module="action",
        metrics=(
            ModuleMetric("action", "mean_score", mean, len(scores), 3.0,
                         _status(mean, 3.0, ">")),
        ),
    )


# --------------------------------------------------------------------------
# 顶层入口
# --------------------------------------------------------------------------


def evaluate_all(
    dataset: GoldDataset,
    outputs: Sequence[MVASampleOutput],
) -> tuple[ModuleEvaluation, ...]:
    if len(dataset.samples) != len(outputs):
        raise ValueError("evaluate_all: samples 与 outputs 长度不一致")
    return (
        evaluate_perception(dataset, outputs),
        evaluate_fast_system(dataset, outputs),
        evaluate_slow_system(dataset, outputs),
        evaluate_memory(dataset, outputs),
        evaluate_action(dataset, outputs),
    )


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------


def _status(value: float, threshold: float, op: str) -> str:
    """根据 op (">" 或 "<") 比较 value 与 threshold，返回 status。"""
    if op == ">" and value >= threshold:
        return "ok"
    if op == "<" and value <= threshold:
        return "ok"
    # 中间档：以阈值的 80% / 120% 划分 warn
    if op == ">":
        return "warn" if value >= threshold * 0.8 else "fail"
    return "warn" if value <= threshold * 1.2 else "fail"


def _combine(*statuses: str) -> str:
    order = {"ok": 0, "warn": 1, "unverified": 2, "fail": 3}
    if not statuses:
        return "unverified"
    return max(statuses, key=lambda s: order.get(s, 2))


# --------------------------------------------------------------------------
# M-2 消融实验
# --------------------------------------------------------------------------


def ablation(
    dataset: GoldDataset,
    full_outputs: Sequence[MVASampleOutput],
    ablated_outputs: dict[str, Sequence[MVASampleOutput]],
) -> dict[str, AblationResult]:
    """M-2 消融实验。

    `ablated_outputs` 形如 {"P_face_only": [...], "S_off": [...], "M_off": [...]}。
    与 full_outputs 比较，输出每个消融版本的指标。
    """
    full_metrics = _basic_metrics(dataset, full_outputs)
    out: dict[str, AblationResult] = {}
    for name, outputs in ablated_outputs.items():
        m = _basic_metrics(dataset, outputs)
        out[name] = AblationResult(
            name=name,
            description=_ablation_description(name),
            ccc=m["ccc"],
            rmse=m["rmse"],
            mae=m["mae"],
            n_samples=len(outputs),
            delta_ccc=m["ccc"] - full_metrics["ccc"],
            delta_rmse=m["rmse"] - full_metrics["rmse"],
        )
    return out


def _basic_metrics(
    dataset: GoldDataset, outputs: Sequence[MVASampleOutput]
) -> dict[str, float]:
    y_true = [s.stress for s in dataset.samples]
    y_pred = [o.stress for o in outputs]
    return {
        "ccc": metrics.ccc(y_true, y_pred),
        "rmse": metrics.rmse(y_true, y_pred),
        "mae": metrics.mae(y_true, y_pred),
    }


def _ablation_description(name: str) -> str:
    table = {
        "P_face_only":   "感知层只用 face 模态",
        "P_speech_only": "感知层只用 speech 模态",
        "P_text_only":   "感知层只用 text 模态",
        "F_off":         "快系统不平滑（tau_fast=0）",
        "S_off":         "慢系统不评估（state 永远 OK）",
        "M_off":         "记忆层不检索（个性化失效）",
        "A_off":         "行动层总是 continue",
    }
    return table.get(name, name)
