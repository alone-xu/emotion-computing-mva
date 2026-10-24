"""audit.data_audit 单元测试。"""

from __future__ import annotations

from emotion_computing.audit.data_audit import (
    annotation_consistency,
    audit_dataset,
    class_distribution,
    distribution_report,
    intensity_distribution,
    modality_availability,
    spot_check,
    QASpotCheckEntry,
)
from emotion_computing.contracts import GoldDataset, GoldLabel


def _g(sid: str, stress: float, state: str = "OK",
       modalities: dict | None = None, annot: str = "a1",
       score: int = 3) -> GoldLabel:
    return GoldLabel(
        sample_id=sid,
        modality_labels=modalities or {"face": 0.5, "speech": 0.5, "text": 0.5},
        stress=stress, state=state,
        action_appropriateness=score,
        annotator_id=annot,
    )


# ---------- D-1 ----------
def test_consistency_perfect():
    rel = [["OK", "OK"], ["WATCH", "WATCH"], ["PAUSE", "PAUSE"]]
    rep = annotation_consistency(rel)
    assert rep.alpha == 1.0
    assert rep.verdict == "ok"


def test_consistency_fail():
    rel = [["OK", "WATCH"]] * 5
    rep = annotation_consistency(rel)
    assert rep.alpha < 0.67
    assert rep.verdict == "fail"


def test_consistency_warn():
    rel = [["OK", "OK"], ["WATCH", "WATCH"], ["WATCH", "WATCH"],
           ["OK", "PAUSE"], ["PAUSE", "OK"], ["PAUSE", "PAUSE"]]
    # 6 个样本，6 对中 4 对一致，2 对不一致 → α 应为正
    rep = annotation_consistency(rel)
    assert 0.4 < rep.alpha < 0.8


# ---------- D-2 ----------
def test_spot_check_coverage():
    samples = tuple(_g(f"s{i}", 0.3) for i in range(10))
    ds = GoldDataset(name="t", samples=samples)
    spot_ids = [f"s{i}" for i in range(0, 2)]
    assignments = {sid: QASpotCheckEntry(sample_id=sid, is_correct=True,
                                          reviewer_id="r1", reviewed_at="2026-01-01T00:00:00")
                   for sid in spot_ids}
    rep = spot_check(ds, spot_ids, assignments)
    assert rep.coverage == pytest_approx(0.2)
    assert rep.n_correct == 2
    assert rep.n_wrong == 0


# ---------- D-3 ----------
def test_class_distribution():
    samples = (
        _g("s1", 0.1, "OK"),
        _g("s2", 0.5, "WATCH"),
        _g("s3", 0.5, "WATCH"),
        _g("s4", 0.9, "PAUSE"),
    )
    cls = class_distribution(samples, "state")
    assert cls == {"OK": 1, "WATCH": 2, "PAUSE": 1}


def test_intensity_distribution():
    samples = tuple(_g(f"s{i}", v) for i, v in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]))
    out = intensity_distribution(samples)
    assert out == {"low": 2, "mid": 1, "high": 2}


def test_modality_availability():
    samples = (
        _g("s1", 0.1, modalities={"face": 0.1}),
        _g("s2", 0.2, modalities={"face": 0.1, "speech": 0.2}),
        _g("s3", 0.3, modalities={"face": 0.1, "speech": 0.2, "text": 0.3}),
    )
    avail = modality_availability(samples)
    assert avail["face"] == pytest_approx(1.0)
    assert avail["speech"] == pytest_approx(2 / 3)
    assert avail["text"] == pytest_approx(1 / 3)


def test_distribution_report_small_class_warning():
    samples = tuple(_g(f"s{i}", 0.5, "PAUSE") for i in range(5))
    ds = GoldDataset(name="t", samples=samples)
    rep = distribution_report(ds)
    assert "PAUSE" in rep.small_class_warning


# ---------- 顶层 audit_dataset ----------
def test_audit_dataset_consistency_fail_aborts():
    samples = tuple(_g(f"s{i}", 0.3) for i in range(20))
    ds = GoldDataset(name="t", samples=samples)
    rel = [["OK", "WATCH"]] * 20
    rep = audit_dataset(ds, labels_matrix=rel)
    assert rep.overall_verdict == "fail"
    assert rep.consistency.verdict == "fail"


def test_audit_dataset_no_consistency_unverified():
    # 90 个样本（每类 30+，避免 small_class_warning）
    samples = []
    for i in range(90):
        state = ["OK", "WATCH", "PAUSE"][i % 3]
        samples.append(_g(f"s{i}", 0.3, state=state))
    ds = GoldDataset(name="t", samples=tuple(samples))
    rep = audit_dataset(ds)
    assert rep.consistency.verdict == "unverified"
    # 没分布 warning → 应是 unverified
    assert rep.overall_verdict in ("unverified", "ok")


from pytest import approx as pytest_approx  # type: ignore
