"""audit.log 单元测试。"""

from __future__ import annotations

from pathlib import Path

from emotion_computing.audit.log import AuditLogStore, log_action


def test_log_store_append_and_load(tmp_path: Path):
    p = tmp_path / "log.jsonl"
    store = AuditLogStore(p)
    e1 = log_action(store, audit_id="a1", reviewer_id="r1", sample_id="s1",
                    system_attribution="modality_sabotage", human_attribution=None,
                    correction_reason=None, follow_up="none")
    e2 = log_action(store, audit_id="a1", reviewer_id="r2", sample_id="s2",
                    system_attribution="perception_failure",
                    human_attribution="data_problem",
                    correction_reason="标注者理解偏差", follow_up="bug_filed")
    rows = store.load()
    assert len(rows) == 2
    assert rows[0].log_id == e1.log_id
    assert rows[1].human_attribution == "data_problem"


def test_audit_chain(tmp_path: Path):
    p = tmp_path / "log.jsonl"
    store = AuditLogStore(p)
    log_action(store, "a1", "r1", "s1", "x", "y", "reason1", "none")
    log_action(store, "a1", "r2", "s1", "x", "z", "reason2", "config_tuned")
    chain = store.audit_chain("s1")
    assert len(chain) == 2
    assert chain[0]["reviewer_id"] == "r1"
    assert chain[1]["reviewer_id"] == "r2"


def test_find_by_sample(tmp_path: Path):
    p = tmp_path / "log.jsonl"
    store = AuditLogStore(p)
    log_action(store, "a1", "r1", "s1", "x")
    log_action(store, "a1", "r1", "s2", "x")
    rows = store.find_by_sample("s1")
    assert len(rows) == 1
    assert rows[0].sample_id == "s1"


def test_audit_chain_empty(tmp_path: Path):
    p = tmp_path / "log.jsonl"
    store = AuditLogStore(p)
    assert store.audit_chain("nonexistent") == []
