"""④ 记忆层 · 持久化可靠性 / 索引 / 检索 回归测试。

覆盖评审里点名的四个缺口：
  1. 崩溃中断后能恢复（checkpoint + recover）；
  2. 画像原子写 + 损坏不静默；
  3. 缺字段旧索引被识别并计数告警；
  4. 检索的绝对距离 / 阈值 / 权重 / 时间衰减。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from emotion_computing.contracts import Episode
from emotion_computing.memory import (
    FEATURE_KEYS,
    FEATURE_WEIGHTS,
    EpisodeStore,
    PrivacyViolation,
    find_similar,
    summarize,
)
from emotion_computing.memory.retrieval import feature_vector


def make_episode(store: EpisodeStore, peak: float = 0.5, n: int = 30, task: str = "t") -> Episode:
    episode = store.new_episode(task)
    episode.baseline_mu = 0.20
    episode.baseline_sigma = 0.02
    episode.samples = [
        {
            "t": float(i),
            "raw": 0.2,
            "smoothed": 0.20 + (peak - 0.20) * (i / max(n - 1, 1)),
            "confidence": 0.8,
            "contributions": {"face": 0.5, "speech": 0.5},
        }
        for i in range(n)
    ]
    episode.events = [{"timestamp": 10.0, "kind": "spike", "magnitude": 0.3,
                       "dominant_modality": "face", "detail": ""}]
    episode.summary = summarize(episode)
    return episode


# --------------------------------------------------------------------------
# 崩溃安全
# --------------------------------------------------------------------------


def test_checkpoint_survives_crash(tmp_path: Path) -> None:
    """39 分钟崩溃不该让前面 39 分钟归零（P0-#4）。"""
    store = EpisodeStore(tmp_path)
    episode = make_episode(store, peak=0.7, n=20)
    store.save_checkpoint(episode)
    assert episode.status == "running"
    store.close()

    # 模拟进程被杀后重新启动
    restarted = EpisodeStore(tmp_path)
    assert restarted.recovered_aborted == 1
    loaded = restarted.load_all()
    assert len(loaded) == 1
    assert loaded[0].status == "aborted"
    assert len(loaded[0].samples) == 20, "检查点里的样本必须还在"
    restarted.close()


def test_finalize_compacts_journal(tmp_path: Path) -> None:
    """多次检查点 + 终态写入后，日志只保留每个会话最后一行。"""
    store = EpisodeStore(tmp_path)
    episode = make_episode(store, peak=0.6, n=10)
    store.save_checkpoint(episode)
    store.save_checkpoint(episode)
    store.append_episode(episode)

    lines = [ln for ln in store.episodes_path.read_text("utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["status"] == "done"
    assert EpisodeStore(tmp_path).load_all()[0].status == "done"
    store.close()


def test_recover_keeps_aborted_out_of_profile(tmp_path: Path) -> None:
    """aborted 会话保留样本但不参与画像学习（由 pipeline 的样本门控保证）。"""
    store = EpisodeStore(tmp_path)
    episode = make_episode(store, peak=0.5, n=5)
    store.save_checkpoint(episode)
    store.close()

    restarted = EpisodeStore(tmp_path)
    assert restarted.load_all()[0].status == "aborted"
    assert restarted.load_index() == [], "aborted 会话不该进索引"
    restarted.close()


# --------------------------------------------------------------------------
# 画像持久化
# --------------------------------------------------------------------------


def test_profile_save_is_atomic(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path)
    profile = store.load_profile()
    store.save_profile(profile)
    assert (tmp_path / "profile.json").exists()
    assert not list(tmp_path.glob(".profile.json.*.tmp")), "临时文件必须被 replace 掉"
    store.close()


def test_corrupt_profile_is_backed_up_and_warned(tmp_path: Path, caplog) -> None:
    (tmp_path / "profile.json").write_text("{ this is not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="emotion_computing.config"):
        profile = EpisodeStore(tmp_path).load_profile()

    assert profile.sessions == 0
    backups = list(tmp_path.glob("profile.json.corrupt-*"))
    assert backups, "损坏的画像必须留证据，不能静默消失"
    assert any("画像损坏" in rec.message for rec in caplog.records)


# --------------------------------------------------------------------------
# 索引
# --------------------------------------------------------------------------


def test_sqlite_index_mirrors_jsonl(tmp_path: Path) -> None:
    """M-2：JSONL 是人类可读主存储，SQLite 是另一份检索索引。"""
    store = EpisodeStore(tmp_path)
    episode = make_episode(store, peak=0.55)
    store.append_episode(episode)

    assert (tmp_path / "index.sqlite").exists()
    assert store.index_db.available is True
    rows = store.index_db.rows("t")
    assert len(rows) == 1
    assert rows[0]["episode_id"] == episode.episode_id
    store.close()


def test_missing_feature_rows_are_flagged(tmp_path: Path, caplog) -> None:
    """改 FEATURE_KEYS 后旧索引不能"无声缩水"（P1-#9）。"""
    store = EpisodeStore(tmp_path)
    store.append_episode(make_episode(store))
    with store.index_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"episode_id": "old_row", "task_label": "t", "mean": 0.5}) + "\n")

    with caplog.at_level(logging.WARNING, logger="emotion_computing.memory.store"):
        rows = store.load_index()

    assert len(rows) == 1, "只有完整行参与检索"
    assert store.last_index_skipped == 1
    assert store.index_health()["skipped_rows"] == 1
    assert any("FEATURE_KEYS" in rec.message for rec in caplog.records)
    store.close()


def test_reindex_rebuilds_lost_index(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path)
    store.append_episode(make_episode(store, peak=0.5))
    store.index_path.write_text("", encoding="utf-8")
    assert store.load_index() == []

    stats = store.reindex()
    assert stats["episodes"] == 1
    assert stats["index_rows"] == 1
    assert len(store.load_index()) == 1
    assert store.index_health()["sqlite_rows"] == 1
    store.close()


def test_delete_single_episode(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path)
    keep = make_episode(store, peak=0.3)
    drop = make_episode(store, peak=0.9)
    store.append_episode(keep)
    store.append_episode(drop)
    store.write_report(drop.episode_id, "x")

    assert store.delete_episode(drop.episode_id) is True
    assert [e.episode_id for e in store.load_all()] == [keep.episode_id]
    assert [r["episode_id"] for r in store.load_index()] == [keep.episode_id]
    assert not (store.reports_dir / f"{drop.episode_id}.md").exists()
    assert store.delete_episode("nope") is False
    store.close()


# --------------------------------------------------------------------------
# 检索
# --------------------------------------------------------------------------


def test_similarity_is_comparable_across_top_k(tmp_path: Path) -> None:
    """同一对会话，top_k=1 与 top_k=3 算出的 distance/similarity 必须一致。"""
    store = EpisodeStore(tmp_path)
    low = make_episode(store, peak=0.30, n=40)
    mid = make_episode(store, peak=0.55, n=40)
    high = make_episode(store, peak=0.90, n=40)
    for ep in (low, mid, high):
        store.append_episode(ep)

    rows = store.load_index()
    query = feature_vector(mid.summary)
    one = find_similar(rows, query, task_label="t", top_k=1, exclude_id=mid.episode_id)
    three = find_similar(rows, query, task_label="t", top_k=3, exclude_id=mid.episode_id)

    assert one and three
    assert one[0]["episode_id"] == three[0]["episode_id"]
    assert one[0]["distance"] == pytest.approx(three[0]["distance"])
    assert one[0]["similarity"] == pytest.approx(three[0]["similarity"])
    store.close()


def test_unrelated_history_is_filtered_by_threshold(tmp_path: Path) -> None:
    """历史全是毫不相干的会话时，不许硬凑 3 条"相似"（P1-#8）。"""
    store = EpisodeStore(tmp_path)
    store.append_episode(make_episode(store, peak=0.95, n=40))
    rows = store.load_index()

    # 查询一个"很平静、很快结束"的情境
    calm = {
        "mean": 0.05, "max": 0.06, "std": 0.005, "slope_early": 0.0,
        "slope_late": 0.0, "peak_time_ratio": 0.2, "n_spikes": 0.0,
        "duration_min": 1.0,
    }
    assert find_similar(rows, feature_vector(calm), task_label="t") == []
    store.close()


def test_duration_weighs_less_than_stress() -> None:
    """时长权重必须低于紧张度权重（P1-#8：8 分钟 vs 8.2 分钟不算"更像"）。"""
    assert FEATURE_WEIGHTS["duration_min"] < FEATURE_WEIGHTS["mean"]
    assert FEATURE_WEIGHTS["duration_min"] < FEATURE_WEIGHTS["std"]
    assert FEATURE_WEIGHTS["duration_min"] <= min(
        FEATURE_WEIGHTS[k] for k in FEATURE_KEYS if k != "duration_min"
    )


def test_old_sessions_decay() -> None:
    """两年前的会话不该和昨天等权（P2-#13）。"""
    now = datetime.now(timezone.utc)
    base = {
        "task_label": "t", "task_outcome": 0.5,
        "mean": 0.5, "max": 0.7, "std": 0.1, "slope_early": 0.1,
        "slope_late": -0.1, "peak_time_ratio": 0.5, "n_spikes": 0.0,
        "duration_min": 8.0,
    }
    rows = [
        {"episode_id": "old", "started_at": (now - timedelta(days=730)).isoformat(), **base},
        {"episode_id": "new", "started_at": (now - timedelta(days=1)).isoformat(), **base},
    ]
    hits = find_similar(rows, feature_vector(base), task_label="t", top_k=2, now=now)

    assert [h["episode_id"] for h in hits] == ["new", "old"]
    assert hits[0]["similarity"] == pytest.approx(hits[1]["similarity"])
    assert hits[0]["decay"] > hits[1]["decay"]
    assert hits[0]["score"] > hits[1]["score"]


# --------------------------------------------------------------------------
# 隐私门
# --------------------------------------------------------------------------


def test_notes_free_text_is_rejected(tmp_path: Path) -> None:
    """M-6：notes 不许写用户原话（P2-#15）。"""
    store = EpisodeStore(tmp_path)
    episode = make_episode(store)
    episode.notes = ["用户原话：我今天真的撑不住了，感觉整个人都要垮了，这个任务我根本做不下去啊"]
    with pytest.raises(PrivacyViolation):
        store.append_episode(episode)
    assert not store.episodes_path.exists() or store.episodes_path.read_text("utf-8") == ""
    store.close()


def test_short_structured_note_is_allowed(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path)
    episode = make_episode(store)
    episode.notes = ["backlog_high"]
    store.append_episode(episode)
    assert store.load_all()[0].notes == ["backlog_high"]
    store.close()


def test_raw_sample_field_is_rejected(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path)
    episode = make_episode(store)
    episode.samples[0]["frame"] = 1
    with pytest.raises(PrivacyViolation):
        store.append_episode(episode)
    store.close()


# --------------------------------------------------------------------------
# schema 兼容
# --------------------------------------------------------------------------


def test_unknown_fields_are_preserved_not_dropped() -> None:
    episode = Episode.from_dict(
        {"episode_id": "a", "started_at": "2026-01-01T00:00:00+08:00", "future_field": 42}
    )
    assert episode.extra["future_field"] == 42
    assert episode.status == "done", "旧数据没有 status，默认视为正常结束"
    assert episode.schema_version == 1
