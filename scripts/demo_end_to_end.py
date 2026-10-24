"""端到端演示：一条命令跑通五层，并验证跨会话记忆确实生效。

用法::

    python scripts/demo_end_to_end.py

它会：
  1. 跑第 1 次会话（合成信号）→ 产出 Episode + 复盘报告
  2. 跑第 2 次同类任务会话 → 报告中应出现「跨会话对比」
  3. 断言产物落盘、结构合法

注意：这里用的是**合成信号**，它只用于验证管线连通性，
**不能用来评估任何效果指标**（见 docs/01-requirements.md §7.2）。

产物写在 `data/demo/` 而不是 `data/`：演示每次都要从干净状态起步（会先 clear），
放在独立子目录里就不会把你自己真实跑过的会话数据一起清掉。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emotion_computing.config import Config  # noqa: E402
from emotion_computing.pipeline import SessionPipeline  # noqa: E402
from emotion_computing.perception.synthetic import SyntheticStressDriver  # noqa: E402

DATA_DIR = ROOT / "data" / "demo"
DURATION = 540.0


def sparkline(values: list[float], width: int = 56) -> str:
    """把紧张度序列压成一行 ASCII 曲线。"""
    if not values:
        return "(无数据)"
    step = max(1, len(values) // width)
    sampled = values[::step]
    blocks = " ▁▂▃▄▅▆▇█"
    return "".join(blocks[min(len(blocks) - 1, int(v * (len(blocks) - 1)))] for v in sampled)


def run_one(label: str, outcome: float, seed: int) -> dict:
    print("=" * 78)
    print(f"会话：{label}    （合成信号，仅供管线验证）")
    print("=" * 78)

    cfg = Config()
    pipe = SessionPipeline(
        config=cfg,
        data_dir=str(DATA_DIR),
        task_label="demo_task",
        driver=SyntheticStressDriver(seed=seed),
    )
    print(f"感知后端：{pipe.backends()}")

    logged: list[tuple[float, float, str, str]] = []

    def on_step(result) -> None:
        if result.state is not None:
            logged.append((result.t, result.stress or 0.0, result.state, result.action or "-"))

    result = pipe.run_offline(duration=DURATION, task_outcome=outcome, on_step=on_step)
    episode = result["episode"]

    values = [float(s["smoothed"]) for s in episode.samples]
    print(f"标定后样本数：{len(values)}")
    print(f"紧张度曲线：|{sparkline(values)}|")
    print(f"  低 0.0 ─────────────────────────────── 高 1.0")

    print("\n慢系统 + 行动层关键决策：")
    print(f"  {'t(s)':>7}  {'紧张度':>7}  {'状态':<7}  动作")
    for t, stress, state, action in logged[:: max(1, len(logged) // 12)]:
        print(f"  {t:>7.0f}  {stress:>7.3f}  {state:<7}  {action}")

    print("\n会话摘要：")
    summary = episode.summary
    for key in ("mean", "max", "std", "n_spikes", "duration_min", "recovery_time"):
        print(f"  {key:<16}= {summary.get(key)}")

    print(f"\n复盘报告：{result['report_path']}")
    return result


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

    # 干净起步，保证演示可复现
    from emotion_computing.memory import EpisodeStore

    EpisodeStore(str(DATA_DIR)).clear()

    first = run_one("第 1 次（task_outcome=0.62）", 0.62, seed=7)
    second = run_one("第 2 次（task_outcome=0.78）", 0.78, seed=21)

    # ---------------- 验证 ----------------
    print("=" * 78)
    print("验证")
    print("=" * 78)
    ep1 = first["episode"]
    ep2 = second["episode"]

    assert (DATA_DIR / "episodes.jsonl").exists(), "episodes.jsonl 未生成"
    assert (DATA_DIR / "index.jsonl").exists(), "index.jsonl 未生成"
    assert (DATA_DIR / "profile.json").exists(), "profile.json 未生成"
    assert first["report_path"].exists(), "第 1 次报告未生成"
    assert second["report_path"].exists(), "第 2 次报告未生成"
    print("[✓] 产物落盘：episodes.jsonl / index.jsonl / profile.json / reports/*.md")

    assert ep1.episode_id != ep2.episode_id
    assert len(ep2.samples) > 0 and len(ep1.samples) > 0
    print(f"[✓] 两次会话独立：{ep1.episode_id} / {ep2.episode_id}")

    cross = second["report"].cross_session
    assert cross, "第 2 次会话的报告中未出现跨会话对比 —— 记忆层没生效"
    print(f"[✓] 跨会话记忆生效：检索到 {len(cross)} 条同类历史会话")

    profile = EpisodeStore(str(DATA_DIR)).load_profile()
    assert profile.sessions >= 2, f"画像 sessions 应 ≥2，实际 {profile.sessions}"
    print(f"[✓] 个性化回写生效：sessions={profile.sessions} "
          f"resilience={profile.resilience:.3f}")

    assert ep2.summary["n_samples"] > 0 and ep2.summary["max"] > 0.0
    print("[✓] 摘要统计非空")

    # 隐私：样本里不允许出现原始数据字段
    forbidden = {"frame", "audio", "raw_text", "video"}
    for s in ep2.samples + ep1.samples:
        assert not (forbidden & set(s.keys())), f"样本中出现原始数据字段：{s.keys()}"
    print("[✓] 隐私约束：样本中不含任何原始音视频/文本字段")

    print("\n演示完成。用下面命令查看复盘报告：")
    print(f"  python -m emotion_computing.cli report {ep2.episode_id} --data-dir {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
