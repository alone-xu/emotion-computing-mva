"""L-# 审核日志（03-audit-requirements.md §7）。

追加写 JSONL（never overwrite），用于事后归因与审计链。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from ..contracts import AuditLogEntry


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class AuditLogStore:
    """审核日志的 JSONL 追加存储。"""

    def __init__(self, log_path: str | Path) -> None:
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: AuditLogEntry) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.as_dict(), ensure_ascii=False) + "\n")

    def load(self) -> list[AuditLogEntry]:
        if not self.path.exists():
            return []
        out: list[AuditLogEntry] = []
        for line in self.path.read_text("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out.append(AuditLogEntry(**d))
            except (json.JSONDecodeError, TypeError):
                continue
        return out

    def find_by_sample(self, sample_id: str) -> list[AuditLogEntry]:
        return [e for e in self.load() if e.sample_id == sample_id]

    def audit_chain(self, sample_id: str) -> list[dict[str, Any]]:
        """追溯单个样本的完整审计链（L-2）。"""
        out: list[dict[str, Any]] = []
        for e in self.find_by_sample(sample_id):
            out.append({
                "log_id": e.log_id,
                "reviewer_id": e.reviewer_id,
                "system_attribution": e.system_attribution,
                "human_attribution": e.human_attribution,
                "correction_reason": e.correction_reason,
                "follow_up": e.follow_up,
                "timestamp": e.timestamp,
            })
        return out


def log_action(
    store: AuditLogStore,
    audit_id: str,
    reviewer_id: str,
    sample_id: str,
    system_attribution: str,
    human_attribution: str | None = None,
    correction_reason: str | None = None,
    follow_up: str = "none",
) -> AuditLogEntry:
    """便捷的写日志函数（L-1）。"""
    entry = AuditLogEntry(
        log_id=uuid.uuid4().hex[:12],
        audit_id=audit_id,
        reviewer_id=reviewer_id,
        sample_id=sample_id,
        system_attribution=system_attribution,
        human_attribution=human_attribution,
        correction_reason=correction_reason,
        follow_up=follow_up,
        timestamp=_now_iso(),
    )
    store.append(entry)
    return entry
