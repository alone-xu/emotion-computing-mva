"""④ 记忆层 · 存储。

实现需求 M-1 ~ M-3、M-5、M-6。

存储布局（全部位于 data/ 下）::

    data/
      episodes.jsonl    # 一行一个 Episode 快照（主存储，人类可读，追加写）
      index.jsonl       # 一行一个聚合特征向量（检索用，可重建）
      index.sqlite      # 检索索引（M-2 要求；纯派生物，丢了可用 reindex 重建）
      profile.json      # 跨会话个体画像（M-5 回写，原子写）
      reports/
        <episode_id>.md  # 复盘报告

为什么用 JSONL 而不是数据库：MVA 阶段的可读性与可调试性 > 查询性能；
任何一行都能用文本编辑器看懂，出问题时可人工核查。

崩溃安全（评审 P0-#4）
----------------------
``episodes.jsonl`` 是**追加式日志**，不是"结束时写一次"：

  * 会话进行中每 ``checkpoint_interval`` 秒追加一条 ``status="running"`` 快照；
  * ``finalize()`` 追加 ``status="done"`` 的终态快照，然后**压缩**日志
    （每个 episode_id 只保留最后一行，顺序按最后一次出现）；
  * 进程异常退出后，下一次 ``EpisodeStore()`` 构造会调用 ``recover()``，
    把残留的 ``running`` 标为 ``aborted`` 并落到日志里 —— 39 分钟不是白跑。

并发与原子性：
  * 所有读写走同一把**跨进程文件锁**（``episodes.jsonl.lock``），进程内用
    ``RLock`` 保证可重入，避免两个进程 append 出交错半行；
  * 整文件重写（压缩 / 重建索引 / 画像）一律"临时文件 + fsync + ``os.replace``"。

隐私红线（M-6）
---------------
本层只存储**特征、统计量与结论**，不存储原始视频帧、音频波形或完整原始文本。
``assert_privacy_safe()`` 是这条红线的执行点：``notes`` 只能是结构化短标签，
写一句用户原话会直接抛 ``PrivacyViolation``。
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..config import load_profile, save_profile, write_text_atomic
from ..contracts import EPISODE_SCHEMA_VERSION, Episode, PersonalProfile
from .index_db import SqliteIndex

LOGGER = logging.getLogger(__name__)

#: 参与相似度计算的数值特征（顺序固定，不要随意插队）
FEATURE_KEYS: tuple[str, ...] = (
    "mean",
    "max",
    "std",
    "slope_early",
    "slope_late",
    "peak_time_ratio",
    "n_spikes",
    "duration_min",
)

#: 特征权重：紧张度/离散度/斜率 > 峰值位置 > 尖峰数 > 时长。
#: 时长权重刻意压到最低 —— 否则"8 分钟 vs 8.2 分钟"会比"紧张度 0.5 vs 0.8"更"像"。
FEATURE_WEIGHTS: dict[str, float] = {
    "mean": 1.0,
    "max": 0.8,
    "std": 1.2,
    "slope_early": 1.2,
    "slope_late": 1.2,
    "peak_time_ratio": 0.6,
    "n_spikes": 0.5,
    "duration_min": 0.4,
}

#: 每个特征的固定物理尺度（用于标准化）：距离因此是**绝对量**，跨查询可比。
#: 取值含义：1 个尺度 ≈ "明显不同"。例：mean 差 0.20、时长差 5 分钟。
FEATURE_SCALE: dict[str, float] = {
    "mean": 0.20,
    "max": 0.20,
    "std": 0.15,
    "slope_early": 0.50,
    "slope_late": 0.50,
    "peak_time_ratio": 0.30,
    "n_spikes": 2.0,
    "duration_min": 5.0,
}

#: samples 里允许出现的字段（白名单）。新增字段必须同步这里，否则视为隐私违规。
PRIVACY_ALLOWED_SAMPLE_KEYS: frozenset[str] = frozenset(
    {"t", "raw", "smoothed", "confidence", "contributions"}
)

#: 隐私红线：禁止出现的原始媒体字段名（序列化后的 JSON 里做子串检查）
PRIVACY_FORBIDDEN_TOKENS: tuple[str, ...] = ("frame", "audio", "video", "waveform", "transcript")

#: notes 只允许结构化短标签（``backlog_high`` / ``task:midterm`` 这种），
#: **不允许**自由文本 —— 否则红线会被一句用户原话捅穿（评审 #15）。
#: 人类可读的描述请写进复盘报告（由模板生成），不要写进记忆。
PRIVACY_MAX_NOTE_CHARS = 40
PRIVACY_NOTE_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,40}$")


class PrivacyViolation(ValueError):
    """写入记忆的内容越过 M-6 红线。"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _slope(times: list[float], values: list[float]) -> float:
    """简单线性斜率（每分钟）；用于聚合特征，不追求稳健性。"""
    n = len(times)
    if n < 2:
        return 0.0
    dt = times[-1] - times[0]
    if dt <= 1e-6:
        return 0.0
    return (values[-1] - values[0]) / dt * 60.0


def summarize(episode: Episode) -> dict:
    """从 Episode 的样本序列中提取聚合统计量（供检索、复盘、回写使用）。"""
    samples = episode.samples or []
    values = [float(s.get("smoothed", 0.0)) for s in samples]
    times = [float(s.get("t", 0.0)) for s in samples]
    confs = [float(s.get("confidence", 0.0)) for s in samples]
    n = len(values)

    if n == 0:
        return {
            "mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0,
            "n_samples": 0, "n_spikes": 0, "n_events": 0,
            "duration_min": 0.0, "peak_time_ratio": 0.0,
            "slope_early": 0.0, "slope_late": 0.0,
            "recovery_time": None, "mean_confidence": 0.0,
            "n_interventions": 0,
        }

    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
    peak_idx = max(range(n), key=lambda i: values[i])
    duration = times[-1] - times[0] if n > 1 else 0.0

    half = max(n // 2, 1)
    slope_early = _slope(times[:half], values[:half])
    slope_late = _slope(times[half:], values[half:])

    # 恢复时间：峰值之后首次回落到 baseline 与峰值中点所需的时间
    recovery_time = None
    if n > 1:
        baseline = float(episode.baseline_mu)
        half_level = baseline + 0.5 * (values[peak_idx] - baseline)
        for i in range(peak_idx + 1, n):
            if values[i] <= half_level:
                recovery_time = float(times[i] - times[peak_idx])
                break

    events = episode.events or []
    return {
        "mean": float(mean),
        "max": float(values[peak_idx]),
        "min": float(min(values)),
        "std": float(std),
        "n_samples": n,
        "n_spikes": sum(1 for e in events if e.get("kind") == "spike"),
        "n_events": len(events),
        "duration_min": float(duration / 60.0),
        "peak_time_ratio": float(peak_idx / max(n - 1, 1)),
        "slope_early": float(slope_early),
        "slope_late": float(slope_late),
        "recovery_time": recovery_time,
        "mean_confidence": float(sum(confs) / n),
        "n_interventions": len(episode.interventions or []),
    }


def assert_privacy_safe(episode: Episode) -> None:
    """M-6 红线的执行点：越线直接拒绝写入，不做"先存后删"。

    检查三件事：
      1. ``samples`` 只能有白名单字段（不许混进原始帧/波形/转写文本）；
      2. ``notes`` 只能是短结构化标签（不许写用户原话，评审 #15）；
      3. 序列化结果里不许出现 frame/audio/video/waveform/transcript 这些字段名。
    """
    for i, sample in enumerate(episode.samples or []):
        if not isinstance(sample, dict):
            raise PrivacyViolation(f"samples[{i}] 不是 dict")
        extra = set(sample) - PRIVACY_ALLOWED_SAMPLE_KEYS
        if extra:
            raise PrivacyViolation(
                f"samples[{i}] 含白名单外字段 {sorted(extra)}；"
                "记忆层只允许存特征与统计量（M-6）"
            )

    for i, note in enumerate(episode.notes or []):
        if not isinstance(note, str):
            raise PrivacyViolation(f"notes[{i}] 不是字符串")
        if "\n" in note or len(note) > PRIVACY_MAX_NOTE_CHARS or not PRIVACY_NOTE_PATTERN.match(note):
            raise PrivacyViolation(
                f"notes[{i}] 只能是结构化短标签（如 backlog_high / task:midterm），"
                f"收到 {note[:20]!r}…（长度 {len(note)}）；"
                "禁止在记忆里写用户原话或自由文本（M-6）"
            )

    blob = json.dumps(episode.as_dict(), ensure_ascii=False).lower()
    for token in PRIVACY_FORBIDDEN_TOKENS:
        if f'"{token}' in blob:
            raise PrivacyViolation(f"记忆内容里出现了原始媒体字段名 {token!r}（M-6）")


# --------------------------------------------------------------------------
# 跨进程文件锁
# --------------------------------------------------------------------------


@contextmanager
def _os_file_lock(fh) -> Iterator[None]:
    """在已经打开的句柄上加排他锁（Windows: msvcrt / POSIX: flock）。"""
    if os.name == "nt":  # pragma: no cover - 平台相关
        import msvcrt

        fh.seek(0)
        if fh.read(1) == b"":
            fh.write(b"\0")
            fh.flush()
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            fh.seek(0)
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
    else:  # pragma: no cover - 平台相关
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class EpisodeStore:
    def __init__(self, data_dir: str | Path = "data", *, auto_recover: bool = True) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.data_dir / "episodes.jsonl"
        self.index_path = self.data_dir / "index.jsonl"
        self.index_sqlite_path = self.data_dir / "index.sqlite"
        self.profile_path = self.data_dir / "profile.json"
        self.reports_dir = self.data_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.data_dir / "episodes.jsonl.lock"

        self.index_db = SqliteIndex(self.index_sqlite_path)

        self._rlock = threading.RLock()
        self._lock_depth = 0

        #: 上一次 load_index() 跳过的旧行数（缺特征字段，P1-#9）
        self.last_index_skipped = 0
        #: 上一次 recover() 标记为 aborted 的会话数
        self.recovered_aborted = 0

        if auto_recover:
            self.recovered_aborted = self.recover()

    # ------------------------------------------------------------------
    # 锁
    # ------------------------------------------------------------------
    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._rlock:
            if self._lock_depth > 0:  # 进程内可重入：不再二次加 OS 锁
                self._lock_depth += 1
                try:
                    yield
                finally:
                    self._lock_depth -= 1
                return
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            with self.lock_path.open("a+b") as fh:
                with _os_file_lock(fh):
                    self._lock_depth += 1
                    try:
                        yield
                    finally:
                        self._lock_depth -= 1

    # ------------------------------------------------------------------
    # 基础读写
    # ------------------------------------------------------------------
    def _read_lines(self, path: Path) -> list[str]:
        if not path.exists():
            return []
        try:
            return path.read_text("utf-8").splitlines()
        except OSError:
            return []

    def _append_jsonl(self, path: Path, payload: dict) -> None:
        data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        with self._locked():
            with path.open("ab") as fh:
                fh.write(data)  # 单次 write，避免 Windows 上的交错半行
                fh.flush()
                os.fsync(fh.fileno())

    def _load_latest(self) -> list[Episode]:
        """读日志并解析为"每个 episode_id 的最后一行"（顺序 = 最后一次出现）。"""
        latest: dict[str, Episode] = {}
        for line in self._read_lines(self.episodes_path):
            line = line.strip()
            if not line:
                continue
            try:
                ep = Episode.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue
            latest.pop(ep.episode_id, None)
            latest[ep.episode_id] = ep
        return list(latest.values())

    def _write_journal(self, episodes: Iterable[Episode]) -> None:
        text = "".join(
            json.dumps(ep.as_dict(), ensure_ascii=False) + "\n" for ep in episodes
        )
        write_text_atomic(self.episodes_path, text)

    # ------------------------------------------------------------------
    # 会话生命周期
    # ------------------------------------------------------------------
    def new_episode(self, task_label: str = "default") -> Episode:
        return Episode(
            episode_id=uuid.uuid4().hex[:12],
            started_at=_now_iso(),
            task_label=task_label,
            status="running",
        )

    def save_checkpoint(self, episode: Episode) -> None:
        """会话进行中的检查点（``status="running"``）。

        崩溃重启后 ``recover()`` 会把它标成 ``aborted``，样本不会丢。
        """
        episode.status = "running"
        episode.schema_version = EPISODE_SCHEMA_VERSION
        self._append_jsonl(self.episodes_path, episode.as_dict())

    def append_episode(self, episode: Episode) -> Path:
        """落盘一次会话终态（``status="done"``），压缩日志并写索引。"""
        assert_privacy_safe(episode)
        episode.status = "done"
        episode.ended_at = _now_iso()
        with self._locked():
            self._append_jsonl(self.episodes_path, episode.as_dict())
            # 压缩：每个 episode_id 只留最后一行，文件不会随检查点无限膨胀
            self._write_journal(self._load_latest())
            self._write_index_row(episode)
        return self.episodes_path

    def recover(self) -> int:
        """把崩溃残留的 ``running`` 会话标为 ``aborted``；返回处理条数。

        保留样本（诊断价值），但明确它不是一次完整会话，不参与画像学习。

        快速路径：没有 ``running`` 行就直接返回，不加锁、不写盘 ——
        纯读取（``episodes`` / ``profile`` 这类命令）不该产生任何写入副作用。
        """
        if not self.episodes_path.exists():
            return 0
        if not any('"running"' in line for line in self._read_lines(self.episodes_path)):
            return 0
        with self._locked():
            episodes = self._load_latest()
            aborted = 0
            for ep in episodes:
                if ep.status == "running":
                    ep.status = "aborted"
                    ep.ended_at = ep.ended_at or _now_iso()
                    aborted += 1
            if aborted:
                self._write_journal(episodes)
                LOGGER.warning(
                    "发现 %d 个未正常结束的会话，已标记为 aborted（样本保留，可用于诊断）。",
                    aborted,
                )
            return aborted

    # ------------------------------------------------------------------
    # 索引
    # ------------------------------------------------------------------
    def _index_record(self, episode: Episode) -> dict:
        summary = episode.summary or summarize(episode)
        return {
            "episode_id": episode.episode_id,
            "task_label": episode.task_label,
            "started_at": episode.started_at,
            "task_outcome": episode.task_outcome,
            "status": episode.status,
            **summary,
        }

    def _write_index_row(self, episode: Episode) -> None:
        """索引 upsert：同名 episode 覆盖（重放/重跑不会写重复行）。"""
        record = self._index_record(episode)
        rows = [r for r in self._read_index_raw() if r.get("episode_id") != episode.episode_id]
        rows.append(record)
        self._write_index_raw(rows)
        self.index_db.upsert(record)

    def _read_index_raw(self) -> list[dict]:
        out: list[dict] = []
        for line in self._read_lines(self.index_path):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def _write_index_raw(self, rows: Iterable[dict]) -> None:
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        write_text_atomic(self.index_path, text)

    def load_index(self) -> list[dict]:
        """读索引行；缺特征字段的旧行会被跳过并**计数告警**（P1-#9）。"""
        rows: list[dict] = []
        skipped: list[str] = []
        for row in self._read_index_raw():
            if all(k in row for k in FEATURE_KEYS):
                rows.append(row)
            else:
                skipped.append(str(row.get("episode_id", "?")))
        self.last_index_skipped = len(skipped)
        if skipped:
            LOGGER.warning(
                "index.jsonl 有 %d 行缺少 FEATURE_KEYS（%s）；这些历史会话检索时不可见，"
                "请运行 `emotion-computing reindex` 重建索引。",
                len(skipped),
                ", ".join(skipped[:5]) + ("…" if len(skipped) > 5 else ""),
            )
        return rows

    def index_health(self) -> dict:
        raw = self._read_index_raw()
        usable = [r for r in raw if all(k in r for k in FEATURE_KEYS)]
        missing_keys: dict[str, int] = {}
        for row in raw:
            for key in FEATURE_KEYS:
                if key not in row:
                    missing_keys[key] = missing_keys.get(key, 0) + 1
        return {
            "jsonl_rows": len(raw),
            "usable_rows": len(usable),
            "skipped_rows": len(raw) - len(usable),
            "missing_feature_counts": missing_keys,
            "sqlite_rows": self.index_db.count(),
            "sqlite_available": self.index_db.available,
            "sqlite_error": self.index_db.last_error,
            "feature_keys": list(FEATURE_KEYS),
        }

    # ------------------------------------------------------------------
    def load_all(self) -> list[Episode]:
        episodes = self._load_latest()
        for ep in episodes:
            if ep.status == "running":
                ep.status = "aborted"  # 内存态兜底：调用方永远看不到"还在跑"的历史
        return episodes

    def load(self, episode_id: str) -> Episode | None:
        for ep in self.load_all():
            if ep.episode_id == episode_id:
                return ep
        return None

    # ------------------------------------------------------------------
    def write_report(self, episode_id: str, markdown: str) -> Path:
        path = self.reports_dir / f"{episode_id}.md"
        write_text_atomic(path, markdown)
        return path

    # ------------------------------------------------------------------
    def load_profile(self) -> PersonalProfile:
        return load_profile(self.profile_path)

    def save_profile(self, profile: PersonalProfile) -> None:
        save_profile(self.profile_path, profile)

    # ------------------------------------------------------------------
    # 维护操作
    # ------------------------------------------------------------------
    def reindex(self) -> dict:
        """从 ``episodes.jsonl`` 重建 index.jsonl + index.sqlite（P1-#9 的修复命令）。"""
        with self._locked():
            episodes = self._load_latest()
            records = [
                self._index_record(ep) for ep in episodes if ep.status in ("done", "aborted")
            ]
            self._write_index_raw(records)
            n = self.index_db.rebuild(records)
        return {
            "episodes": len(episodes),
            "index_rows": len(records),
            "sqlite_rows": n,
            "sqlite_available": self.index_db.available,
        }

    def delete_episode(self, episode_id: str) -> bool:
        """删除单条会话（P2-#12：不再只能全清）。"""
        with self._locked():
            episodes = self._load_latest()
            kept = [ep for ep in episodes if ep.episode_id != episode_id]
            if len(kept) == len(episodes):
                return False
            self._write_journal(kept)
            self._write_index_raw(
                [r for r in self._read_index_raw() if r.get("episode_id") != episode_id]
            )
            self.index_db.delete(episode_id)
            report = self.reports_dir / f"{episode_id}.md"
            if report.exists():
                report.unlink()
        return True

    def prune(self, keep_last: int | None = None, max_age_days: float | None = None) -> int:
        """保留策略（P2-#12）：按条数/天数清理旧会话，返回删除条数。"""
        if keep_last is None and max_age_days is None:
            return 0
        episodes = self._load_latest()
        order = sorted(episodes, key=lambda e: e.started_at or "")
        drop: set[str] = set()
        if keep_last is not None and keep_last >= 0:
            for ep in order[: max(len(order) - keep_last, 0)]:
                drop.add(ep.episode_id)
        if max_age_days is not None:
            cutoff = datetime.now(timezone.utc).timestamp() - max_age_days * 86400.0
            for ep in order:
                try:
                    ts = datetime.fromisoformat(str(ep.started_at)).timestamp()
                except ValueError:
                    continue
                if ts < cutoff:
                    drop.add(ep.episode_id)
        for episode_id in drop:
            self.delete_episode(episode_id)
        return len(drop)

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """清空所有记忆产物（需求文档：数据可一键删除）。"""
        with self._locked():
            # 先关索引连接：Windows 上文件被占用时 unlink 会失败
            self.index_db.close()
            paths = [
                self.episodes_path,
                self.index_path,
                self.index_sqlite_path,
                Path(str(self.index_sqlite_path) + "-wal"),
                Path(str(self.index_sqlite_path) + "-shm"),
                self.profile_path,
            ]
            for p in paths:
                if p.exists():
                    p.unlink()
            if self.reports_dir.exists():
                for f in self.reports_dir.iterdir():
                    if f.is_file():
                        f.unlink()
        self.index_db = SqliteIndex(self.index_sqlite_path)

    # ------------------------------------------------------------------
    def close(self) -> None:
        self.index_db.close()

    def __enter__(self) -> "EpisodeStore":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
