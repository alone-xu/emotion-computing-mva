"""④ 记忆层 · 复盘报告生成。

实现需求 M-4。

复盘回答四个问题（需求文档 §5.4 / 场景 2）：
  1. 这次紧张度是怎么变化的？（时间线）
  2. 峰值出现在什么时候、由什么驱动？（峰值归因）
  3. 平复得快不快？（恢复画像）
  4. 和任务完成度、和历史上同类任务相比如何？（关联与对比）
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from ..contracts import Episode, ReviewReport
from .store import summarize

DISCLAIMER = (
    "> **诚实性声明（必读）**\n"
    "> 本报告中的紧张度为**个体内相对量**：可以在同一次会话的不同时刻之间比较，\n"
    "> 但**不是**心理学或医学意义上的测量结果，也不支持跨人比较。\n"
    "> 预测模型未经真实数据回测（`validated=False`）。\n"
    "> 本系统不提供任何诊断结论，**不得**用于考核、评价或淘汰个人。"
)

_MAX_TIMELINE_ROWS = 24


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _find_peaks(times: list[float], values: list[float], n: int = 3) -> list[int]:
    """找局部极大值索引，按高度取前 n 个（彼此至少间隔 10% 时长）。"""
    if len(values) < 3:
        return list(range(len(values))) if values else []
    min_gap = max(1, int(0.10 * len(values)))
    candidates: list[int] = []
    for i in range(1, len(values) - 1):
        if values[i] >= values[i - 1] and values[i] > values[i + 1]:
            candidates.append(i)
    if not candidates:
        candidates = [max(range(len(values)), key=lambda i: values[i])]
    candidates.sort(key=lambda i: values[i], reverse=True)

    chosen: list[int] = []
    for idx in candidates:
        if all(abs(idx - c) >= min_gap for c in chosen):
            chosen.append(idx)
        if len(chosen) >= n:
            break
    return sorted(chosen)


def build_report(
    episode: Episode,
    cross_session: Sequence[dict] | None = None,
    backends: dict[str, str] | None = None,
) -> ReviewReport:
    samples = episode.samples or []
    times = [float(s.get("t", 0.0)) for s in samples]
    values = [float(s.get("smoothed", 0.0)) for s in samples]
    summary = episode.summary or summarize(episode)

    # ---- 1. 时间线（均匀降采样，控制在 ~24 行）----
    rows: list[dict] = []
    if samples:
        step = max(1, len(samples) // _MAX_TIMELINE_ROWS)
        events = episode.events or []
        for i in range(0, len(samples), step):
            s = samples[i]
            t = float(s.get("t", 0.0))
            contrib = s.get("contributions") or {}
            top_modality = max(contrib.items(), key=lambda kv: kv[1])[0] if contrib else "-"
            matched = [
                e for e in events if abs(float(e.get("timestamp", -1e9)) - t) <= step / 2.0
            ]
            rows.append(
                {
                    "t": t,
                    "stress": float(s.get("smoothed", 0.0)),
                    "top_modality": top_modality,
                    "event": ",".join(e.get("kind", "") for e in matched),
                }
            )

    # ---- 2. 峰值归因 ----
    peaks: list[dict] = []
    for idx in _find_peaks(times, values):
        contrib = (samples[idx].get("contributions") or {}) if idx < len(samples) else {}
        peaks.append(
            {
                "t": times[idx],
                "stress": values[idx],
                "dominant": max(contrib.items(), key=lambda kv: kv[1])[0] if contrib else "-",
                "contributions": {
                    k: round(float(v), 3) for k, v in sorted(
                        contrib.items(), key=lambda kv: kv[1], reverse=True
                    )
                },
            }
        )

    # ---- 3. 恢复画像 ----
    recovery = {
        "峰值": round(summary.get("max", 0.0), 3),
        "均值": round(summary.get("mean", 0.0), 3),
        "标准差": round(summary.get("std", 0.0), 3),
        "峰值后恢复耗时(s)": (
            round(summary["recovery_time"], 1)
            if summary.get("recovery_time") is not None
            else "未在会话内回落到半程"
        ),
        "前半段斜率(/min)": round(summary.get("slope_early", 0.0), 3),
        "后半段斜率(/min)": round(summary.get("slope_late", 0.0), 3),
        "尖峰次数": summary.get("n_spikes", 0),
        "干预次数": summary.get("n_interventions", 0),
    }

    # ---- 4. 与任务完成度的关联 ----
    outcome_link: dict = {}
    if episode.task_outcome is not None:
        outcome_link = {
            "task_outcome": round(float(episode.task_outcome), 3),
            "本会话平均紧张度": round(summary.get("mean", 0.0), 3),
            "本会话峰值": round(summary.get("max", 0.0), 3),
            "说明": (
                "单个会话无法得出统计相关性；此处仅并列展示，"
                "跨会话累积后才有讨论价值。"
            ),
        }

    # ---- 5. 跨会话对比 ----
    cross: list[dict] = [dict(r) for r in (cross_session or [])]

    # ---- 6. 建议 ----
    suggestions: list[str] = []
    if summary.get("peak_time_ratio", 0.0) > 0.6:
        suggestions.append(
            "紧张峰值出现在任务后段——考虑在后段之前主动安排一次短休息，"
            "而不是等到峰值出现再补救。"
        )
    if peaks and peaks[0]["dominant"] == "speech":
        suggestions.append(
            "最强峰值主要由语音（音调/语速）驱动——练习放慢语速与延长呼气，"
            "对这类峰值更有效。"
        )
    if peaks and peaks[0]["dominant"] == "face":
        suggestions.append(
            "最强峰值主要由面部活动（皱眉/唇部紧张）驱动——这类信号往往比主观感受更早出现，"
            "可作为自我觉察的早期提示。"
        )
    rt = summary.get("recovery_time")
    if rt is not None and rt > 90:
        suggestions.append(
            f"峰值后恢复耗时约 {rt:.0f}s，偏长——建议在峰值后安排明确的缓冲时段，"
            "而不是立刻投入下一个子任务。"
        )
    if summary.get("n_interventions", 0) == 0 and summary.get("max", 0.0) >= 0.55:
        suggestions.append(
            "本次会话出现过中高位紧张但未触发干预——若希望更早介入，"
            "可下调 theta_mid 或放宽行动层冷却时间。"
        )
    if not suggestions:
        suggestions.append("本次会话未见明显恶化模式，保持当前节奏即可。")

    return ReviewReport(
        episode_id=episode.episode_id,
        generated_at=_now_iso(),
        timeline_rows=rows,
        peaks=peaks,
        recovery=recovery,
        outcome_link=outcome_link,
        cross_session=cross,
        suggestions=suggestions,
        backends=dict(backends or episode.backends or {}),
        disclaimer=DISCLAIMER,
    )
