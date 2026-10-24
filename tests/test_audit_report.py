"""audit.report + viz 单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

from emotion_computing.audit import report, viz
from emotion_computing.audit.failure_attribution import build_distribution
from emotion_computing.contracts import (
    AuditReport,
    FailureCase,
    FailureDistribution,
    ModuleEvaluation,
    ModuleMetric,
)
from emotion_computing.audit.modality_diagnostic import ModalityAttribution


def _mod_eval() -> tuple[ModuleEvaluation, ...]:
    return (
        ModuleEvaluation(
            module="perception",
            metrics=(
                ModuleMetric("perception.face", "ccc", 0.7, 10, 0.5, "ok"),
                ModuleMetric("perception.speech", "ccc", 0.4, 10, 0.5, "fail"),
            ),
        ),
        ModuleEvaluation(
            module="fast_system",
            metrics=(ModuleMetric("fast_system", "ccc", 0.6, 20, 0.5, "ok"),),
        ),
    )


def _att(sample_id: str, fused: float, gold: float, saboteur: str | None = None):
    return ModalityAttribution(
        sample_id=sample_id, face=0.3, speech=0.7, text=0.4,
        fused=fused, gold=gold, dominant_modality="speech",
        saboteur=saboteur, reason="",
    )


def _cases() -> tuple[FailureCase, ...]:
    return (
        FailureCase(sample_id="s1", error=0.3, system_attribution="modality_sabotage",
                    modality_attribution=_att("s1", 0.6, 0.3, "speech")),
        FailureCase(sample_id="s2", error=0.4, system_attribution="perception_failure",
                    modality_attribution=_att("s2", 0.7, 0.3, None)),
    )


def test_to_markdown_contains_key_sections():
    fd = build_distribution(_cases())
    rep = AuditReport(
        audit_id="abc123",
        generated_at="2026-01-01T00:00:00",
        dataset_name="t",
        n_samples=20,
        n_errors=2,
        data_qa={
            "consistency": {"alpha": 0.85, "kappa": 0.80, "verdict": "ok"},
            "spot_check": {"coverage": 0.2, "n_sampled": 4, "n_total": 20,
                           "n_wrong": 0, "n_data_problem": 0,
                           "error_type_dist": {}, "error_reason_dist": {}},
            "distribution": {"class_dist": {"OK": 15, "WATCH": 5},
                             "intensity_dist": {"low": 5, "mid": 10, "high": 5},
                             "modality_availability": {"face": 0.9, "speech": 0.8, "text": 0.7},
                             "small_class_warning": []},
        },
        module_evaluations=_mod_eval(),
        ablation={"P_face_only": {"ccc": 0.5, "rmse": 0.2, "mae": 0.18, "delta_ccc": -0.1}},
        failure_distribution=fd,
        failure_cases=_cases(),
        disclaimer=report.DISCLAIMER,
    )
    md = report.to_markdown(rep)
    assert "审核报告 · abc123" in md
    assert "数据层审核" in md
    assert "模块级评估" in md
    assert "消融实验" in md
    assert "失败归因" in md
    assert "模态破坏" in md
    assert "感知失败" in md


def test_write_report_creates_files(tmp_path: Path):
    fd = build_distribution(_cases())
    rep = AuditReport(
        audit_id="xyz", generated_at="2026-01-01T00:00:00",
        dataset_name="t", n_samples=10, n_errors=2,
        data_qa={}, module_evaluations=(), ablation={},
        failure_distribution=fd, failure_cases=_cases(),
        disclaimer=report.DISCLAIMER,
    )
    paths = report.write_report(rep, tmp_path)
    assert paths["markdown"].exists()
    assert paths["json"].exists()
    # JSON 包含 n_samples
    j = json.loads(paths["json"].read_text("utf-8"))
    assert j["n_samples"] == 10
    assert j["audit_id"] == "xyz"


def test_dashboard_creates_html(tmp_path: Path):
    fd = build_distribution(_cases())
    rep = AuditReport(
        audit_id="d1", generated_at="2026-01-01T00:00:00",
        dataset_name="t", n_samples=10, n_errors=2,
        data_qa={}, module_evaluations=_mod_eval(), ablation={},
        failure_distribution=fd, failure_cases=_cases(),
        disclaimer=report.DISCLAIMER,
    )
    p = viz.generate_dashboard(rep, tmp_path / "dash.html")
    assert p.exists()
    html = p.read_text("utf-8")
    assert "MVA 审核仪表盘" in html
    assert "DATA =" in html
    assert "modality_sabotage" in html  # 数据 JSON 中的 key
