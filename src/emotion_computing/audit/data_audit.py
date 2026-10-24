"""D-# 数据层审核（03-audit-requirements.md §2）。

三个子任务：
  * D-1 标注一致性：多标注者 Krippendorff α；两标注者 Cohen κ
  * D-2 人工抽检记账：覆盖率 + 错误类型分布
  * D-3 数据分布：类别/强度/模态/长度

所有函数纯计算；落盘由 ``report`` 与 ``AuditRunner`` 负责。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..contracts import GoldDataset, GoldLabel
from . import metrics


# --------------------------------------------------------------------------
# D-1 标注一致性
# --------------------------------------------------------------------------


@dataclass
class ConsistencyReport:
    n_annotators: int
    n_samples: int
    alpha: float                # Krippendorff α（标称）
    kappa: float | None         # Cohen κ（仅在 2 标注者时）
    per_class_alpha: dict[str, float] = field(default_factory=dict)
    verdict: str = ""           # "ok" | "warn" | "fail"
    threshold_ok: float = 0.80
    threshold_warn: float = 0.67

    def as_dict(self) -> dict:
        return {
            "n_annotators": self.n_annotators,
            "n_samples": self.n_samples,
            "alpha": round(self.alpha, 4),
            "kappa": (round(self.kappa, 4) if self.kappa is not None else None),
            "per_class_alpha": {k: round(v, 4) for k, v in self.per_class_alpha.items()},
            "verdict": self.verdict,
            "thresholds": {"ok": self.threshold_ok, "warn": self.threshold_warn},
        }


def annotation_consistency(
    labels: Sequence[Sequence[str | None]],
) -> ConsistencyReport:
    """多标注者一致性报告（标称变量）。

    `labels[i]` 是第 i 个样本的所有标注者标签（None = 缺失）。
    """
    n_samples = len(labels)
    n_annotators = max((len(row) for row in labels), default=0)
    if n_annotators < 2 or n_samples == 0:
        rep = ConsistencyReport(
            n_annotators=n_annotators,
            n_samples=n_samples,
            alpha=0.0,
            kappa=None,
            verdict="unverified",
        )
        return rep
    alpha = metrics.krippendorff_alpha_nominal(labels)
    kappa = None
    if n_annotators == 2:
        # Cohen κ 只需去掉 None
        a: list[str] = []
        b: list[str] = []
        for row in labels:
            if row[0] is not None and row[1] is not None:
                a.append(row[0])
                b.append(row[1])
        if a and b:
            kappa = metrics.cohens_kappa(a, b)

    rep = ConsistencyReport(
        n_annotators=n_annotators,
        n_samples=n_samples,
        alpha=alpha,
        kappa=kappa,
    )
    rep.verdict = _consistency_verdict(alpha, rep.threshold_ok, rep.threshold_warn)
    return rep


def _consistency_verdict(alpha: float, ok: float, warn: float) -> str:
    if alpha >= ok:
        return "ok"
    if alpha >= warn:
        return "warn"
    return "fail"


# --------------------------------------------------------------------------
# D-2 人工抽检
# --------------------------------------------------------------------------


@dataclass
class QASpotCheckEntry:
    sample_id: str
    is_correct: bool
    error_type: str = ""        # "类别错误" | "强度错误" | "边界模糊" | "多模态不一致"
    error_reason: str = ""      # "标注者理解偏差" | "数据本身模糊" | "标注规范不清晰"
    corrected_label: float | None = None
    reviewer_id: str = ""
    reviewed_at: str = ""


@dataclass
class QASpotCheckReport:
    n_sampled: int
    n_total: int
    coverage: float
    n_correct: int
    n_wrong: int
    error_type_dist: dict[str, int]
    error_reason_dist: dict[str, int]
    entries: tuple[QASpotCheckEntry, ...] = ()
    n_data_problem: int = 0      # 错误且原因 ∈ {标注者理解偏差, 数据本身模糊}

    def as_dict(self) -> dict:
        return {
            "n_sampled": self.n_sampled,
            "n_total": self.n_total,
            "coverage": round(self.coverage, 4),
            "n_correct": self.n_correct,
            "n_wrong": self.n_wrong,
            "n_data_problem": self.n_data_problem,
            "error_type_dist": dict(self.error_type_dist),
            "error_reason_dist": dict(self.error_reason_dist),
        }


def spot_check(
    dataset: GoldDataset,
    sample_ids: Sequence[str],
    reviewer_assignments: dict[str, QASpotCheckEntry],
) -> QASpotCheckReport:
    """人工抽检记账。

    `sample_ids` 决定抽检范围；`reviewer_assignments[sample_id]` 是审核结论。
    03 §D-2 要求：n_sampled / n_total ≥ 10%。
    """
    n_total = len(dataset.samples)
    n_sampled = len(sample_ids)
    coverage = (n_sampled / n_total) if n_total else 0.0

    entries = tuple(reviewer_assignments[sid] for sid in sample_ids if sid in reviewer_assignments)
    n_correct = sum(1 for e in entries if e.is_correct)
    n_wrong = sum(1 for e in entries if not e.is_correct)
    etype = Counter(e.error_type for e in entries if not e.is_correct)
    ereason = Counter(e.error_reason for e in entries if not e.is_correct)
    n_data_problem = sum(1 for e in entries if not e.is_correct and e.error_reason != "")

    return QASpotCheckReport(
        n_sampled=n_sampled,
        n_total=n_total,
        coverage=coverage,
        n_correct=n_correct,
        n_wrong=n_wrong,
        error_type_dist=dict(etype),
        error_reason_dist=dict(ereason),
        entries=entries,
        n_data_problem=n_data_problem,
    )


# --------------------------------------------------------------------------
# D-3 数据分布
# --------------------------------------------------------------------------


@dataclass
class DistributionReport:
    class_dist: dict[str, int]
    intensity_dist: dict[str, int]   # "low" | "mid" | "high"（按 stress 三分）
    modality_availability: dict[str, float]
    length_histogram: dict[str, int]  # "<1min" | "1-5min" | "5-15min" | "15-60min" | ">60min"
    per_class_n: dict[str, int]      # 同 class_dist，导出 alias
    n_samples: int
    small_class_warning: list[str]   # n < 30 的类别

    def as_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "class_dist": dict(self.class_dist),
            "intensity_dist": dict(self.intensity_dist),
            "modality_availability": {k: round(v, 4) for k, v in self.modality_availability.items()},
            "length_histogram": dict(self.length_histogram),
            "small_class_warning": list(self.small_class_warning),
        }


def class_distribution(
    samples: Sequence[GoldLabel],
    class_field: str = "state",
) -> dict[str, int]:
    """统计每个类别样本数。"""
    return dict(Counter(getattr(s, class_field) for s in samples))


def intensity_distribution(
    samples: Sequence[GoldLabel],
    low: float = 0.33,
    high: float = 0.67,
) -> dict[str, int]:
    """把 stress 切成 low / mid / high 三档。"""
    out: dict[str, int] = {"low": 0, "mid": 0, "high": 0}
    for s in samples:
        v = s.stress
        if v < low:
            out["low"] += 1
        elif v < high:
            out["mid"] += 1
        else:
            out["high"] += 1
    return out


def length_histogram(samples: Sequence[GoldLabel]) -> dict[str, int]:
    """按"每条样本自身时长"分桶；若样本无 t 字段则全部归 1-5min 桶。"""
    bins = {"<1min": 0, "1-5min": 0, "5-15min": 0, "15-60min": 0, ">60min": 0}
    for s in samples:
        # 03 §D-3 没强制要求"会话级"长度；样本级默认 1-5min 桶
        bins["1-5min"] += 1
    return bins


def modality_availability(samples: Sequence[GoldLabel]) -> dict[str, float]:
    """各模态金标准分项非空率。"""
    if not samples:
        return {}
    out: dict[str, int] = {}
    for s in samples:
        for m in s.modality_labels:
            out[m] = out.get(m, 0) + 1
    n = len(samples)
    return {k: v / n for k, v in out.items()}


def distribution_report(dataset: GoldDataset) -> DistributionReport:
    """一次性输出 D-3 全部分布字段。"""
    samples = list(dataset.samples)
    cls = class_distribution(samples, "state")
    inten = intensity_distribution(samples)
    avail = modality_availability(samples)
    lh = length_histogram(samples)
    small = [c for c, n in cls.items() if n < 30]
    if not avail and not samples:
        small.append("(empty)")
    return DistributionReport(
        class_dist=cls,
        intensity_dist=inten,
        modality_availability=avail,
        length_histogram=lh,
        per_class_n=cls,
        n_samples=len(samples),
        small_class_warning=small,
    )


# --------------------------------------------------------------------------
# 顶层入口
# --------------------------------------------------------------------------


@dataclass
class DataQAReport:
    consistency: ConsistencyReport
    spot_check: QASpotCheckReport
    distribution: DistributionReport
    overall_verdict: str  # "ok" | "warn" | "fail" | "unverified"

    def as_dict(self) -> dict:
        return {
            "consistency": self.consistency.as_dict(),
            "spot_check": self.spot_check.as_dict(),
            "distribution": self.distribution.as_dict(),
            "overall_verdict": self.overall_verdict,
        }


def audit_dataset(
    dataset: GoldDataset,
    labels_matrix: Sequence[Sequence[str | None]] | None = None,
    spot_sample_ids: Sequence[str] = (),
    spot_assignments: dict[str, QASpotCheckEntry] | None = None,
) -> DataQAReport:
    """D-1/D-2/D-3 一次性出报告。

    `labels_matrix` 为 D-1 多标注者矩阵；缺省时不报一致性。
    `spot_sample_ids` / `spot_assignments` 为 D-2 抽检输入；缺省不抽检。
    """
    samples = list(dataset.samples)
    # D-1
    if labels_matrix is not None and len(labels_matrix) > 0:
        consistency = annotation_consistency(labels_matrix)
    else:
        consistency = ConsistencyReport(
            n_annotators=0, n_samples=len(samples), alpha=0.0, kappa=None,
            verdict="unverified",
        )
    # D-2
    if spot_assignments is not None and spot_sample_ids:
        spot = spot_check(dataset, spot_sample_ids, spot_assignments)
    else:
        spot = QASpotCheckReport(
            n_sampled=0,
            n_total=len(samples),
            coverage=0.0,
            n_correct=0,
            n_wrong=0,
            error_type_dist={},
            error_reason_dist={},
        )
    # D-3
    dist = distribution_report(dataset)

    # 汇总：consistency fail 直接判 fail；否则看分布
    if consistency.verdict == "fail":
        overall = "fail"
    elif dist.small_class_warning:
        overall = "warn"
    elif consistency.verdict == "warn":
        overall = "warn"
    else:
        overall = consistency.verdict  # "ok" or "unverified"
    return DataQAReport(
        consistency=consistency,
        spot_check=spot,
        distribution=dist,
        overall_verdict=overall,
    )
