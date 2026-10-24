"""审核报告生成（03-audit-requirements.md §5 / §6 / §8）。

包含：
  * Markdown 报告（人工可读、可在无浏览器环境打开）
  * JSON 机器可读副本
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from ..contracts import (
    FAILURE_LABELS,
    FAILURE_REPAIR,
    FAILURE_TYPES,
    AuditReport,
    FailureCase,
    ModuleEvaluation,
)


DISCLAIMER = (
    "⚠️ 审核报告基于有限测试集与启发式指标。统计结论不构成对模型『准确率』或"
    "个体紧张度的绝对声明。任何对外发布前的最终判断须由人工复核。"
)


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


def to_markdown(report: AuditReport) -> str:
    L: list[str] = []
    L.append(f"# 审核报告 · {report.audit_id}")
    L.append("")
    L.append(f"- 生成时间：{report.generated_at}")
    L.append(f"- 数据集：{report.dataset_name}")
    L.append(f"- 样本数：{report.n_samples}")
    L.append(f"- 错误样本：{report.n_errors}（{report.n_errors / max(report.n_samples, 1) * 100:.1f}%）")
    L.append("")

    L.append("## 1. 数据层审核（D-#）")
    L.append("")
    dq = report.data_qa or {}
    cons = dq.get("consistency", {}) or {}
    spot = dq.get("spot_check", {}) or {}
    dist = dq.get("distribution", {}) or {}
    L.append(f"- 标注一致性 (Krippendorff α)：{cons.get('alpha', float('nan')):.4f}"
             f" → **{cons.get('verdict', 'unverified')}**")
    if cons.get("kappa") is not None:
        L.append(f"- Cohen κ：{cons['kappa']:.4f}")
    L.append(f"- 抽检覆盖率：{spot.get('coverage', 0) * 100:.1f}% "
             f"（{spot.get('n_sampled', 0)}/{spot.get('n_total', 0)}）")
    L.append(f"- 抽检错误数：{spot.get('n_wrong', 0)}，数据问题：{spot.get('n_data_problem', 0)}")
    L.append("")
    L.append("### 数据分布")
    L.append("")
    L.append("| 类别 | 样本数 |")
    L.append("|---|---:|")
    for c, n in (dist.get("class_dist", {}) or {}).items():
        L.append(f"| {c} | {n} |")
    L.append("")
    L.append("| 强度档 | 样本数 |")
    L.append("|---|---:|")
    for k, n in (dist.get("intensity_dist", {}) or {}).items():
        L.append(f"| {k} | {n} |")
    L.append("")
    L.append("| 模态 | 可用率 |")
    L.append("|---|---:|")
    for m, v in (dist.get("modality_availability", {}) or {}).items():
        L.append(f"| {m} | {v * 100:.1f}% |")
    if dist.get("small_class_warning"):
        L.append("")
        L.append(f"⚠️ 类别样本数 < 30：{dist['small_class_warning']}")
    L.append("")

    L.append("## 2. 模块级评估（M-1）")
    L.append("")
    L.extend(_module_table(report.module_evaluations))
    L.append("")

    L.append("## 3. 消融实验（M-2）")
    L.append("")
    L.extend(_ablation_table(report.ablation))
    L.append("")

    L.append("## 4. 失败归因（F-3）")
    L.append("")
    L.extend(_failure_distribution_table(report.failure_distribution))
    L.append("")
    L.append(f"**优先修复方向**（按 03 §I-1 排序）：")
    L.append("")
    for t in report.failure_distribution.top_repair:
        L.append(f"- {t}")
    L.append("")

    if report.iterations:
        L.append("## 5. 跨迭代对比（I-2）")
        L.append("")
        L.append("| 指标 | 当前 | 历史最近 | Δ |")
        L.append("|---|---:|---:|---:|")
        for it in report.iterations:
            L.append(f"| {it.get('metric', '?')} | {it.get('curr', '-')} | "
                     f"{it.get('prev', '-')} | {it.get('delta', '-')} |")
        L.append("")

    if report.failure_cases:
        L.append("## 6. 错误样本清单（前 10 条）")
        L.append("")
        L.append("| 样本ID | 误差 | 系统归因 | 主导模态 | 破坏者 |")
        L.append("|---|---:|---|---|---|")
        for c in report.failure_cases[:10]:
            ma = c.modality_attribution
            dom = ma.dominant_modality if ma else "-"
            sab = ma.saboteur or "-" if ma else "-"
            L.append(f"| {c.sample_id} | {c.error:.3f} | "
                     f"{FAILURE_LABELS.get(c.system_attribution, c.system_attribution)} | "
                     f"{dom} | {sab} |")
        L.append("")

    L.append("---")
    L.append("")
    L.append(DISCLAIMER)
    return "\n".join(L)


def _module_table(evals: Sequence[ModuleEvaluation]) -> list[str]:
    L: list[str] = []
    L.append("| 模块 | 指标 | 值 | 阈值 | 状态 |")
    L.append("|---|---|---:|---:|---|")
    for ev in evals:
        for m in ev.metrics:
            v = m.value
            if m.metric_name in ("f1_watch_pause", "precision_at_3", "mean_score", "ccc"):
                v_str = f"{v:.3f}"
            else:
                v_str = f"{v:.3f}"
            t = m.threshold if m.threshold is not None else "-"
            L.append(f"| {m.module} | {m.metric_name} | {v_str} | {t} | {m.status} |")
    return L


def _ablation_table(abl: dict[str, float]) -> list[str]:
    if not abl:
        return ["_无消融数据_"]
    L: list[str] = []
    L.append("| 消融 | CCC | RMSE | MAE | ΔCCC |")
    L.append("|---|---:|---:|---:|---:|")
    for k, v in abl.items():
        ccc = v.get("ccc", 0.0) if isinstance(v, dict) else 0.0
        rmse = v.get("rmse", 0.0) if isinstance(v, dict) else 0.0
        mae = v.get("mae", 0.0) if isinstance(v, dict) else 0.0
        d_ccc = v.get("delta_ccc", 0.0) if isinstance(v, dict) else 0.0
        L.append(f"| {k} | {ccc:.3f} | {rmse:.3f} | {mae:.3f} | {d_ccc:+.3f} |")
    return L


def _failure_distribution_table(fd) -> list[str]:
    L: list[str] = []
    L.append("| 失败类型 | 数量 | 占比 | 推荐修复 |")
    L.append("|---|---:|---:|---|")
    for t in FAILURE_TYPES:
        c = fd.counts.get(t, 0)
        p = fd.percentages.get(t, 0.0)
        repair = FAILURE_REPAIR.get(t, "-")
        L.append(f"| {FAILURE_LABELS.get(t, t)} | {c} | {p:.1f}% | {repair} |")
    return L


# --------------------------------------------------------------------------
# JSON 落盘
# --------------------------------------------------------------------------


def write_report(
    report: AuditReport,
    output_dir: str | Path,
) -> dict[str, Path]:
    """把 report 写到 output_dir，返回产物路径字典。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    md_path = out / "audit_report.md"
    md_path.write_text(to_markdown(report), encoding="utf-8")
    paths["markdown"] = md_path

    json_path = out / "audit_report.json"
    json_path.write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["json"] = json_path
    return paths


# --------------------------------------------------------------------------
# 辅助：失败样本摘要
# --------------------------------------------------------------------------


def failure_summary(cases: Sequence[FailureCase]) -> dict[str, int]:
    return dict(Counter(c.system_attribution for c in cases))
