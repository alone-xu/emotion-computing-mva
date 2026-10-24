"""④ 记忆层 · SQLite 检索索引。

需求 M-2 原文："存储格式须人类可读且可增量追加（JSONL），**另行提供 SQLite 索引**"。

分工（不要搞混，两者职责不同）：
  * ``episodes.jsonl`` / ``index.jsonl`` —— **事实来源**，人类可读、可 grep、可手工修；
  * ``index.sqlite`` —— **查询索引**，由 JSONL 派生，可随时用 ``reindex()`` 重建，
    丢了也不影响数据。

设计原则：索引永远是可丢弃的派生物。任何写索引失败都只降级为 warning，
绝不让"索引坏了"变成"数据丢了"。
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

LOGGER = logging.getLogger(__name__)

#: 索引表的标量列（其余字段塞进 payload JSON，检索时无需展开）
_COLUMNS: tuple[str, ...] = (
    "episode_id",
    "task_label",
    "started_at",
    "task_outcome",
    "mean",
    "max",
    "min",
    "std",
    "n_samples",
    "n_spikes",
    "n_events",
    "duration_min",
    "peak_time_ratio",
    "slope_early",
    "slope_late",
    "recovery_time",
    "mean_confidence",
    "n_interventions",
)

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS episode_index (
    episode_id      TEXT PRIMARY KEY,
    task_label      TEXT NOT NULL,
    started_at      TEXT,
    task_outcome    REAL,
    mean            REAL,
    max             REAL,
    min             REAL,
    std             REAL,
    n_samples       INTEGER,
    n_spikes        INTEGER,
    n_events        INTEGER,
    duration_min    REAL,
    peak_time_ratio REAL,
    slope_early     REAL,
    slope_late      REAL,
    recovery_time   REAL,
    mean_confidence REAL,
    n_interventions INTEGER,
    payload         TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episode_index_task ON episode_index(task_label);
CREATE INDEX IF NOT EXISTS idx_episode_index_started ON episode_index(started_at);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class SqliteIndex:
    """``index.sqlite`` 的薄封装。

    所有方法在数据库不可用时**不抛异常**，只把 ``self.available`` 置 False 并记 warning，
    保证索引故障不会拖垮主流程（索引是派生物，主存储才是事实）。
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.available = False
        self.last_error: str | None = None
        self._conn: sqlite3.Connection | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), timeout=5.0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
            self.available = True
        except sqlite3.Error as exc:  # pragma: no cover - 环境相关
            self._degrade(exc)

    # ------------------------------------------------------------------
    def _degrade(self, exc: Exception) -> None:
        self.available = False
        self.last_error = f"{type(exc).__name__}: {exc}"
        LOGGER.warning("SQLite 索引不可用（%s）；检索将回落到 index.jsonl。", self.last_error)
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None

    # ------------------------------------------------------------------
    def upsert(self, record: dict) -> None:
        if not self._conn:
            return
        values = {c: record.get(c) for c in _COLUMNS}
        values["task_label"] = values["task_label"] or "default"
        values["payload"] = json.dumps(record, ensure_ascii=False)
        values["updated_at"] = _now_iso()
        placeholders = ", ".join("?" for _ in (*_COLUMNS, "payload", "updated_at"))
        columns = ", ".join((*_COLUMNS, "payload", "updated_at"))
        try:
            self._conn.execute(
                f"INSERT INTO episode_index ({columns}) VALUES ({placeholders}) "
                f"ON CONFLICT(episode_id) DO UPDATE SET "
                + ", ".join(f"{c}=excluded.{c}" for c in (*_COLUMNS[1:], "payload", "updated_at")),
                [values[c] for c in (*_COLUMNS, "payload", "updated_at")],
            )
            self._conn.commit()
        except sqlite3.Error as exc:  # pragma: no cover - 环境相关
            self._degrade(exc)

    # ------------------------------------------------------------------
    def rows(self, task_label: str | None = None, exclude_id: str | None = None) -> list[dict]:
        if not self._conn:
            return []
        sql = "SELECT payload FROM episode_index"
        clauses: list[str] = []
        params: list[object] = []
        if task_label is not None:
            clauses.append("task_label = ?")
            params.append(task_label)
        if exclude_id is not None:
            clauses.append("episode_id != ?")
            params.append(exclude_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY started_at ASC"
        try:
            return [json.loads(r["payload"]) for r in self._conn.execute(sql, params)]
        except sqlite3.Error as exc:  # pragma: no cover
            self._degrade(exc)
            return []

    # ------------------------------------------------------------------
    def delete(self, episode_id: str) -> None:
        if not self._conn:
            return
        try:
            self._conn.execute("DELETE FROM episode_index WHERE episode_id = ?", (episode_id,))
            self._conn.commit()
        except sqlite3.Error as exc:  # pragma: no cover
            self._degrade(exc)

    # ------------------------------------------------------------------
    def rebuild(self, records: Iterable[dict]) -> int:
        """整表重建（``reindex`` 用）。返回写入行数。"""
        if not self._conn:
            return 0
        n = 0
        try:
            self._conn.execute("DELETE FROM episode_index")
            for record in records:
                self.upsert(record)
                n += 1
            self._conn.commit()
        except sqlite3.Error as exc:  # pragma: no cover
            self._degrade(exc)
        return n

    # ------------------------------------------------------------------
    def count(self) -> int:
        if not self._conn:
            return 0
        try:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM episode_index").fetchone()
            return int(row["n"]) if row else 0
        except sqlite3.Error as exc:  # pragma: no cover
            self._degrade(exc)
            return 0

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None

    def __enter__(self) -> "SqliteIndex":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def read_index_rows(path: str | Path) -> list[dict]:
    """直接读 ``index.jsonl``（不经过 Store，供离线脚本使用）。"""
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def feature_keys_of(rows: Sequence[dict]) -> dict[str, int]:
    """统计每行包含的特征字段情况，供 reindex/健康检查使用。"""
    counts: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            counts[key] = counts.get(key, 0) + 1
    return counts
