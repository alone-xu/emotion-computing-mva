"""④ 记忆层 · 存储、检索、复盘测试。"""

from __future__ import annotations

from pathlib import Path

from emotion_computing.contracts import Episode
from emotion_computing.memory import EpisodeStore, build_report, summarize
from emotion_computing.memory.retrieval import FEATURE_KEYS, find_similar


def make_episode(store: EpisodeStore, peak: float = 0.5, n: int = 30, task: str = "t") -> Episode:
    episode = store.new_episode(task)
    episode.baseline_mu = 0.20
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


def test_store_roundtrip(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path)
    episode = make_episode(store, peak=0.7)
    store.append_episode(episode)

    loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0].episode_id == episode.episode_id
    assert loaded[0].task_label == "t"
    assert len(loaded[0].samples) == len(episode.samples)


def test_episode_jsonl_is_human_readable(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path)
    store.append_episode(make_episode(store))
    text = store.episodes_path.read_text("utf-8")
    assert "\n" in text
    assert "episode_id" in text
    assert len(text.splitlines()) == 1


def test_summarize_extracts_statistics(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path)  # 仅用于生成对象
    episode = make_episode(store, peak=0.8, n=50)
    summary = summarize(episode)
    assert summary["n_samples"] == 50
    assert summary["max"] > summary["mean"] > 0.0
    assert summary["n_spikes"] == 1
    assert summary["duration_min"] > 0


def test_summarize_handles_empty_episode(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path)
    empty = store.new_episode("t")
    summary = summarize(empty)
    assert summary["n_samples"] == 0
    assert summary["max"] == 0.0


def test_index_is_written(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path)
    store.append_episode(make_episode(store, peak=0.6))
    rows = store.load_index()
    assert len(rows) == 1
    assert all(k in rows[0] for k in FEATURE_KEYS)


def test_find_similar_ranks_closest_first(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path)
    low = make_episode(store, peak=0.30, n=40)
    mid = make_episode(store, peak=0.55, n=40)
    high = make_episode(store, peak=0.90, n=40)
    for ep in (low, mid, high):
        store.append_episode(ep)

    query = [float(mid.summary[k]) for k in FEATURE_KEYS]
    results = find_similar(store.load_index(), query, task_label="t", top_k=3,
                           exclude_id=mid.episode_id)
    assert results, "应检索到同类历史会话"
    assert results[0]["episode_id"] in (low.episode_id, high.episode_id)
    assert all("similarity" in r for r in results)


def test_find_similar_respects_task_label(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path)
    store.append_episode(make_episode(store, peak=0.5, task="A"))
    query = [0.0] * len(FEATURE_KEYS)
    assert find_similar(store.load_index(), query, task_label="B") == []


def test_report_contains_required_sections_and_disclaimer(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path)
    episode = make_episode(store, peak=0.85, n=60)
    episode.task_outcome = 0.6
    markdown = build_report(episode, backends={"face": "synthetic"}).to_markdown()

    for section in ("时间线", "峰值归因", "恢复画像", "建议"):
        assert section in markdown, f"报告缺少章节：{section}"
    assert "诚实性声明" in markdown, "报告必须包含诚实性声明"
    assert "个体内相对量" in markdown
    assert "不得" in markdown


def test_report_includes_cross_session_rows(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path)
    episode = make_episode(store, peak=0.7)
    cross = [{"episode_id": "abc123", "mean": 0.4, "max": 0.8, "n_spikes": 2,
              "task_outcome": 0.7}]
    markdown = build_report(episode, cross_session=cross).to_markdown()
    assert "abc123" in markdown


def test_profile_roundtrip(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path)
    profile = store.load_profile()
    assert profile.sessions == 0
    profile.sessions = 3
    profile.resilience = 0.42
    profile.intervention_effectiveness["breathing_478"] = 0.15
    store.save_profile(profile)

    reloaded = EpisodeStore(tmp_path).load_profile()
    assert reloaded.sessions == 3
    assert abs(reloaded.resilience - 0.42) < 1e-9
    assert reloaded.intervention_effectiveness["breathing_478"] == 0.15


def test_clear_removes_artifacts(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path)
    episode = make_episode(store)
    store.append_episode(episode)
    store.write_report(episode.episode_id, "hi")
    store.clear()
    assert not store.episodes_path.exists()
    assert not store.index_path.exists()
    assert not (store.reports_dir / f"{episode.episode_id}.md").exists()


def test_samples_contain_no_raw_media_fields(tmp_path: Path) -> None:
    """需求 M-6：记忆里不允许出现原始音视频/文本。"""
    store = EpisodeStore(tmp_path)
    episode = make_episode(store)
    store.append_episode(episode)
    text = store.episodes_path.read_text("utf-8")
    for forbidden in ("frame", "audio", "video", "waveform"):
        assert forbidden not in text
