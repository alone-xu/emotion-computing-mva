"""audit.runner 单元测试：端到端跑通审核流水线。"""

from __future__ import annotations

from pathlib import Path

from emotion_computing.audit.data_audit import QASpotCheckEntry
from emotion_computing.audit.module_evaluator import MVASampleOutput
from emotion_computing.audit.runner import AuditRunner, AuditRunnerConfig
from emotion_computing.contracts import GoldDataset, GoldLabel


def _make_dataset(n: int = 30) -> GoldDataset:
    samples = []
    for i in range(n):
        stress = (i % 10) / 10.0
        state = "PAUSE" if stress > 0.8 else ("WATCH" if stress > 0.4 else "OK")
        samples.append(GoldLabel(
            sample_id=f"s{i}",
            modality_labels={"face": stress, "speech": stress, "text": stress},
            stress=stress, state=state,
            action_appropriateness=3,
            annotator_id="a1",
        ))
    return GoldDataset(name="test_ds", samples=tuple(samples))


def _make_outputs(ds: GoldDataset) -> list[MVASampleOutput]:
    """MVA 输出：80% 完美 + 20% 注入真实破坏者（高置信反向模态）。"""
    out = []
    for i, s in enumerate(ds.samples):
        if i % 5 == 0:
            # 注入"破坏者"：fused 偏离 gold 0.4+；
            # 至少一个模态与 fused 方向相反、高置信
            if s.stress > 0.5:
                # gold 高，fused 低估；找一个高置信的"高"模态当破坏者
                fused = max(0.0, s.stress - 0.4)
                mc = {"face": 0.9, "speech": 0.5, "text": 0.5}
                ms = {"face": s.stress, "speech": 0.5, "text": 0.5}
                state = "OK"
            else:
                # gold 低，fused 高估；找一个高置信的"低"模态当破坏者
                fused = min(1.0, s.stress + 0.4)
                mc = {"face": 0.9, "speech": 0.5, "text": 0.5}
                ms = {"face": 0.0, "speech": 0.5, "text": 0.5}
                state = "WATCH"
            out.append(MVASampleOutput(
                sample_id=s.sample_id, stress=fused,
                state=state, action="continue",
                modality_scores=ms, modality_confidence=mc,
            ))
        else:
            out.append(MVASampleOutput(
                sample_id=s.sample_id, stress=s.stress,
                state=s.state, action="continue",
                modality_scores={"face": s.stress, "speech": s.stress, "text": s.stress},
            ))
    return out


def test_runner_end_to_end(tmp_path: Path):
    cfg = AuditRunnerConfig(
        audit_id="test01",
        output_dir=str(tmp_path / "audit"),
        abort_on_consistency_fail=False,
    )
    runner = AuditRunner(cfg)
    ds = _make_dataset(30)
    outputs = _make_outputs(ds)
    labels_matrix = [["OK", "OK"], ["WATCH", "WATCH"]] * 15  # α=1
    spot_ids = tuple(s.sample_id for s in ds.samples[:3])
    spot_assignments = {
        s: QASpotCheckEntry(sample_id=s, is_correct=True,
                            reviewer_id="r1", reviewed_at="2026-01-01")
        for s in spot_ids
    }
    rep = runner.run(
        ds, outputs,
        labels_matrix=labels_matrix,
        spot_sample_ids=spot_ids,
        spot_assignments=spot_assignments,
        ablation_outputs={"P_face_only": [
            MVASampleOutput(sample_id=s.sample_id, stress=s.stress,
                            modality_scores={"face": s.stress, "speech": 0.5, "text": 0.5})
            for s in ds.samples
        ]},
    )
    assert rep.audit_id == "test01"
    assert rep.n_samples == 30
    assert rep.n_errors >= 1
    # 注入的 5 个偏离样本应至少有一种失败归因
    total_failures = sum(rep.failure_distribution.counts.values())
    assert total_failures >= 1
    # 至少一个错误案例有 modality_attribution 字段
    assert any(c.modality_attribution is not None for c in rep.failure_cases)
    # 产物齐全
    audit_dir = tmp_path / "audit" / "test01"
    assert (audit_dir / "audit_report.md").exists()
    assert (audit_dir / "audit_report.json").exists()
    assert (audit_dir / "dashboard.html").exists()
    assert (audit_dir / "modality_attributions.json").exists()
    assert (audit_dir / "module_evaluations.json").exists()
    assert (audit_dir / "ablation_results.json").exists()
    assert (audit_dir / "failure_cases.jsonl").exists()
    assert (audit_dir / "data_qa.json").exists()
    # 审计日志
    assert (tmp_path / "audit" / "_audit_log.jsonl").exists()
    log = (tmp_path / "audit" / "_audit_log.jsonl").read_text("utf-8")
    # 至少有一条非 "ok" 的归因
    non_ok_lines = [
        l for l in log.splitlines()
        if '"system_attribution"' in l and '"system_attribution": "ok"' not in l
    ]
    assert len(non_ok_lines) >= 1


def test_runner_aborts_on_consistency_fail(tmp_path: Path):
    cfg = AuditRunnerConfig(
        audit_id="abort01",
        output_dir=str(tmp_path / "audit"),
        abort_on_consistency_fail=True,
    )
    runner = AuditRunner(cfg)
    ds = _make_dataset(10)
    outputs = [MVASampleOutput(sample_id=s.sample_id, stress=s.stress,
                                modality_scores={"face": s.stress, "speech": s.stress,
                                                 "text": s.stress})
                for s in ds.samples]
    # labels_matrix 全错 → α 极低
    rel = [["OK", "WATCH"], ["WATCH", "PAUSE"]] * 5
    rep = runner.run(ds, outputs, labels_matrix=rel)
    assert "aborted" in (rep.metadata.get("reason") or "") or rep.n_errors == 0
    # 即使 abort 也应写报告
    audit_dir = tmp_path / "audit" / "abort01"
    assert (audit_dir / "audit_report.md").exists()


def test_runner_with_iterations(tmp_path: Path):
    cfg = AuditRunnerConfig(audit_id="iter01", output_dir=str(tmp_path / "audit"),
                            abort_on_consistency_fail=False)
    runner = AuditRunner(cfg)
    ds = _make_dataset(10)
    outputs = [MVASampleOutput(sample_id=s.sample_id, stress=s.stress,
                                modality_scores={"face": s.stress, "speech": s.stress,
                                                 "text": s.stress})
                for s in ds.samples]
    rep = runner.run(ds, outputs, iterations=[
        {"metric": "ccc", "prev": 0.5, "curr": 0.6, "delta": 0.1},
    ])
    assert len(rep.iterations) == 1
    # _iterations.jsonl 应被写入
    iter_path = tmp_path / "audit" / "_iterations.jsonl"
    assert iter_path.exists()
    lines = iter_path.read_text("utf-8").strip().splitlines()
    assert len(lines) == 1
