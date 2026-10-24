"""审核流水线编排（04-audit-architecture.md §5）。

AuditRunner 串联 D-# / M-# / F-# / V-# / L-#，
输入 GoldDataset + MVA 业务输出，输出：
  * AuditReport（机器可读 + Markdown）
  * data/audit/<audit_id>/dashboard.html（可视化）
  * data/audit/_audit_log.jsonl（追溯）
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from ..contracts import (
    AuditReport,
    FailureCase,
    GoldDataset,
    GoldLabel,
    ModalityAttribution,
    ModuleEvaluation,
)
from . import data_audit, failure_attribution, metrics, module_evaluator, report, viz
from .data_audit import QASpotCheckEntry
from .failure_attribution import AttributionEvidence, build_distribution
from .log import AuditLogStore, log_action
from .modality_diagnostic import DiagnosticConfig, attribute_dataset
from .module_evaluator import MVASampleOutput


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------


@dataclass
class AuditRunnerConfig:
    audit_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    output_dir: str = "data/audit"
    error_theta: float = 0.20          # |fused - gold| > theta 判错
    modality_error_theta: float = 0.25
    theta_saboteur: float = 0.7
    abort_on_consistency_fail: bool = True
    consistency_warn_threshold: float = 0.67
    consistency_ok_threshold: float = 0.80
    spot_check_required_coverage: float = 0.10
    chartjs_source: str | None = None   # None = 用占位
    reviewer_id: str = "system"

    def as_dict(self) -> dict:
        return {
            "audit_id": self.audit_id,
            "output_dir": self.output_dir,
            "error_theta": self.error_theta,
            "modality_error_theta": self.modality_error_theta,
            "theta_saboteur": self.theta_saboteur,
            "abort_on_consistency_fail": self.abort_on_consistency_fail,
            "consistency_thresholds": {
                "ok": self.consistency_ok_threshold,
                "warn": self.consistency_warn_threshold,
            },
            "spot_check_required_coverage": self.spot_check_required_coverage,
        }


# --------------------------------------------------------------------------
# 主类
# --------------------------------------------------------------------------


class AuditRunner:
    def __init__(self, cfg: AuditRunnerConfig | None = None) -> None:
        self.cfg = cfg or AuditRunnerConfig()
        self.audit_dir = Path(self.cfg.output_dir) / self.cfg.audit_id
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.log_store = AuditLogStore(Path(self.cfg.output_dir) / "_audit_log.jsonl")

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------
    def run(
        self,
        dataset: GoldDataset,
        outputs: Sequence[MVASampleOutput],
        labels_matrix: Sequence[Sequence[str | None]] | None = None,
        spot_sample_ids: Sequence[str] = (),
        spot_assignments: dict[str, QASpotCheckEntry] | None = None,
        ablation_outputs: dict[str, Sequence[MVASampleOutput]] | None = None,
        iterations: Sequence[dict] = (),
    ) -> AuditReport:
        cfg = self.cfg
        # 1. 数据层审核
        data_qa = data_audit.audit_dataset(
            dataset,
            labels_matrix=labels_matrix,
            spot_sample_ids=spot_sample_ids,
            spot_assignments=spot_assignments,
        )
        # 落盘
        self._write_json(self.audit_dir / "data_qa.json", data_qa.as_dict())

        if data_qa.consistency.verdict == "fail" and cfg.abort_on_consistency_fail:
            # 03 §D-1 失败信号 → 不进入 M 层
            self._write_text(
                self.audit_dir / "audit_report.md",
                report.to_markdown(self._abort_report(dataset, data_qa)),
            )
            return self._abort_report(dataset, data_qa)

        # 2. 模态级诊断（M-3）—— 需要每样本 fused + 金标准
        y_pred = [o.stress for o in outputs]
        mc = [o.modality_confidence for o in outputs]
        attributions = attribute_dataset(
            list(dataset.samples), y_pred, mc,
            DiagnosticConfig(theta_saboteur=cfg.theta_saboteur),
        )
        self._write_json(
            self.audit_dir / "modality_attributions.json",
            [a.as_dict() for a in attributions],
        )

        # 3. 模块级评估（M-1）
        evals = module_evaluator.evaluate_all(dataset, outputs)
        self._write_json(
            self.audit_dir / "module_evaluations.json",
            [e.as_dict() for e in evals],
        )

        # 4. 消融（M-2）
        abl_results: dict[str, dict[str, float]] = {}
        if ablation_outputs:
            abl = module_evaluator.ablation(dataset, outputs, ablation_outputs)
            abl_results = {k: v.as_dict() for k, v in abl.items()}
        self._write_json(self.audit_dir / "ablation_results.json", abl_results)

        # 5. 失败归因（F-1/F-2/F-3）
        cases, evidence_list = self._build_failures(
            dataset, outputs, attributions,
            spot_assignments=spot_assignments,
        )
        distribution = build_distribution(cases)
        self._write_jsonl(
            self.audit_dir / "failure_cases.jsonl",
            [c.as_dict() for c in cases],
        )
        self._write_json(
            self.audit_dir / "failure_distribution.json",
            distribution.as_dict(),
        )

        # 6. 写审核日志（L-1）
        for c in cases:
            log_action(
                self.log_store,
                audit_id=cfg.audit_id,
                reviewer_id=cfg.reviewer_id,
                sample_id=c.sample_id,
                system_attribution=c.system_attribution,
                human_attribution=c.human_attribution,
                correction_reason=c.correction_reason,
                follow_up="none",
            )

        # 7. 报告 + 可视化
        rep = AuditReport(
            audit_id=cfg.audit_id,
            generated_at=_now_iso(),
            dataset_name=dataset.name,
            n_samples=len(dataset.samples),
            n_errors=len(cases),
            data_qa=data_qa.as_dict(),
            module_evaluations=tuple(evals),
            ablation=abl_results,
            failure_distribution=distribution,
            failure_cases=tuple(cases),
            iterations=tuple(iterations),
            disclaimer=report.DISCLAIMER,
            metadata=cfg.as_dict(),
        )
        paths = report.write_report(rep, self.audit_dir)
        # dashboard
        dashboard_path = self.audit_dir / "dashboard.html"
        viz.generate_dashboard(
            rep, dashboard_path, chartjs_source=cfg.chartjs_source,
        )
        # 迭代历史
        self._append_iteration(rep)
        return rep

    # ------------------------------------------------------------------
    def _build_failures(
        self,
        dataset: GoldDataset,
        outputs: Sequence[MVASampleOutput],
        attributions: list[ModalityAttribution],
        spot_assignments: dict[str, QASpotCheckEntry] | None,
    ) -> tuple[list[FailureCase], list[AttributionEvidence]]:
        # 构造 evidence list
        evs: list[AttributionEvidence] = []
        spot_assignments = spot_assignments or {}
        att_by_id = {a.sample_id: a for a in attributions}
        for sample, out in zip(dataset.samples, outputs):
            spot = spot_assignments.get(sample.sample_id)
            data_suspect = (spot is not None and not spot.is_correct)
            evs.append(AttributionEvidence(
                sample_id=sample.sample_id,
                gold_label=sample,
                fused=out.stress,
                mva_state=out.state,
                mva_action=out.action,
                memory_retrieved=out.retrieved_memory,
                memory_relevant=out.memory_relevant,
                data_suspect=data_suspect,
                modality_attribution=att_by_id.get(sample.sample_id),
            ))
        cases = failure_attribution.attribute_batch(
            evs,
            error_theta=self.cfg.error_theta,
            modality_error_theta=self.cfg.modality_error_theta,
        )
        # 过滤掉非错误的（"ok"）
        cases = [c for c in cases if c.system_attribution != "ok"]
        return cases, evs

    # ------------------------------------------------------------------
    def _abort_report(self, dataset: GoldDataset, data_qa) -> AuditReport:
        return AuditReport(
            audit_id=self.cfg.audit_id,
            generated_at=_now_iso(),
            dataset_name=dataset.name,
            n_samples=len(dataset.samples),
            n_errors=0,
            data_qa=data_qa.as_dict(),
            module_evaluations=(),
            ablation={},
            failure_distribution=failure_attribution.build_distribution([]),
            failure_cases=(),
            disclaimer=report.DISCLAIMER + "\n\n⚠️ 本次审核因标注一致性不足被中止。",
            metadata={"aborted": True, "reason": "consistency_fail"},
        )

    # ------------------------------------------------------------------
    def _write_json(self, path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _write_text(self, path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")

    def _append_iteration(self, rep: AuditReport) -> None:
        path = Path(self.cfg.output_dir) / "_iterations.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        # 提取关键指标
        ccc_v = 0.0
        rmse_v = 0.0
        for ev in rep.module_evaluations:
            if ev.module == "fast_system":
                for m in ev.metrics:
                    if m.metric_name == "ccc":
                        ccc_v = m.value
                    elif m.metric_name == "rmse":
                        rmse_v = m.value
        record = {
            "audit_id": rep.audit_id,
            "generated_at": rep.generated_at,
            "n_samples": rep.n_samples,
            "n_errors": rep.n_errors,
            "ccc": ccc_v,
            "rmse": rmse_v,
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# 便捷工厂
# --------------------------------------------------------------------------


def build_default_runner(
    audit_id: str | None = None,
    output_dir: str = "data/audit",
    chartjs_source: str | None = None,
) -> AuditRunner:
    cfg = AuditRunnerConfig(
        audit_id=audit_id or uuid.uuid4().hex[:12],
        output_dir=output_dir,
        chartjs_source=chartjs_source,
    )
    return AuditRunner(cfg)
