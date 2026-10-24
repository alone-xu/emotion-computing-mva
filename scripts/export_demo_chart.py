"""导出一张会话曲线图（纯 SVG，无 matplotlib 依赖，离线可复现）。

用法::

    # 直接把某次历史会话（episodes.jsonl 里的一行）画出来 —— README 那张图就是这么来的
    python scripts/export_demo_chart.py --episodes data/real_data_runs/episodes.jsonl \\
        --episode-id ef18721606fb --out docs/assets/session-timeline.svg

    # 或者现跑一次合成会话再出图
    python scripts/export_demo_chart.py --out chart.svg --duration 540

为什么要自己拼 SVG：这仓库的定位是"零重依赖、本地能跑"，为了给 README 画一张图
去引入 matplotlib 不划算；而且手写的 SVG 是纯文本，diff 友好、能直接进版本库。

副作用：`--episodes` 模式只读文件，不产生临时目录；合成模式会在当前目录建一个
`_chartdata/`（已被 .gitignore 忽略，删掉即可）。

图里的东西全部来自**真实跑出来的一次会话**：紧张度曲线、两档阈值、静息基线、
尖峰事件、干预触发点。信号来源（合成驱动 / 真实音频）会写在图的标题里。
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emotion_computing.config import Config  # noqa: E402
from emotion_computing.pipeline import SessionPipeline  # noqa: E402

# ---------------------------------------------------------------- 画布参数
W, H = 960, 440
ML, MR, MT, MB = 64, 210, 56, 48  # 右边留给图例
PW, PH = W - ML - MR, H - MT - MB

COLORS = {
    "bg": "#ffffff",
    "frame": "#e5e7eb",
    "grid": "#f1f5f9",
    "axis": "#94a3b8",
    "text": "#334155",
    "muted": "#64748b",
    "curve": "#2563eb",
    "curve_fill": "#dbeafe",
    "mid": "#f59e0b",
    "hi": "#dc2626",
    "base": "#6b7280",
    "event": "#7c3aed",
    "interv": "#059669",
}
FONT = "'Segoe UI','PingFang SC','Microsoft YaHei',system-ui,sans-serif"


def collect(duration: float, stride: int) -> dict:
    """跑一次离线会话，把画图要用的数据全捞出来。"""
    pipeline = SessionPipeline(
        config=Config(), data_dir="_chartdata", task_label="chart-demo"
    )
    times: list[float] = []
    values: list[float] = []
    states: list[str | None] = []
    for t in range(int(duration) + 1):
        r = pipeline.step({"t": float(t)})
        if r.calibrated and r.stress is not None:
            times.append(float(t))
            values.append(float(r.stress))
            states.append(r.state)
    result = pipeline.finalize(task_outcome=0.72)
    episode = result["episode"]

    idx = list(range(0, len(times), max(stride, 1)))
    if idx and idx[-1] != len(times) - 1:
        idx.append(len(times) - 1)

    return {
        "t": [times[i] for i in idx],
        "s": [values[i] for i in idx],
        "t_full": times,
        "s_full": values,
        "duration": times[-1] if times else duration,
        "baseline": float(episode.baseline_mu),
        "theta_mid": float(pipeline.config.theta_mid),
        "theta_hi": float(pipeline.config.theta_hi),
        "events": [
            (float(e["timestamp"]), float(e.get("magnitude", 0.0)), str(e.get("kind", "")))
            for e in episode.events
        ],
        "interventions": [
            (float(i["timestamp"]), str(i.get("action", "")))
            for i in episode.interventions
        ],
        "n_samples": len(times),
        "mean": sum(values) / len(values) if values else 0.0,
        "peak": max(values) if values else 0.0,
        "source": "合成信号",
    }


def load_episode(path: str | Path, episode_id: str | None, stride: int = 1) -> dict:
    """从 episodes.jsonl 里读一次已落盘的会话，转成画图要用的结构。

    这是 README 那张图的来源：用一次**真实音频**（RAVDESS）驱动的会话，
    而不是现跑一段合成信号。
    """
    rows: list[dict] = []
    for line in Path(path).read_text("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        raise SystemExit(f"{path} 里没有可解析的会话")

    if episode_id:
        matches = [r for r in rows if r.get("episode_id") == episode_id]
        if not matches:
            raise SystemExit(f"没找到 episode_id={episode_id}")
        episode = matches[-1]  # 同一 id 多行时取最后一行（终态）
    else:
        episode = max(rows, key=lambda r: len(r.get("samples") or []))

    samples = [s for s in (episode.get("samples") or []) if "smoothed" in s]
    if not samples:
        raise SystemExit("这条会话没有样本，画不了")

    times = [float(s.get("t", 0.0)) for s in samples]
    values = [float(s["smoothed"]) for s in samples]

    # 时间轴归零：episodes.jsonl 里的 t 是"会话内绝对秒数"（含基线期），
    # 画图时统一减掉起点，否则横坐标会溢出画布。
    t0 = times[0]
    times = [t - t0 for t in times]

    idx = list(range(0, len(times), max(stride, 1)))
    if idx and idx[-1] != len(times) - 1:
        idx.append(len(times) - 1)

    cfg = Config()  # 那次会话用的是默认阈值（报告里不含阈值，这里如实标注）
    events = [
        (float(e.get("timestamp", 0.0)) - t0, float(e.get("magnitude", 0.0)),
         str(e.get("kind", "")))
        for e in (episode.get("events") or [])
    ]
    interventions = [
        (float(i.get("timestamp", 0.0)) - t0, str(i.get("action", "")))
        for i in (episode.get("interventions") or [])
    ]

    return {
        "t": [times[i] for i in idx],
        "s": [values[i] for i in idx],
        "t_full": times,
        "s_full": values,
        "duration": times[-1] if len(times) > 1 else 1.0,
        "baseline": float(episode.get("baseline_mu", 0.0) or 0.0),
        "theta_mid": float(cfg.theta_mid),
        "theta_hi": float(cfg.theta_hi),
        "events": events,
        "interventions": interventions,
        "n_samples": len(times),
        "mean": sum(values) / len(values),
        "peak": max(values),
        "source": "真实音频 · RAVDESS",
        "episode_id": str(episode.get("episode_id", "")),
        "backends": dict(episode.get("backends") or {}),
    }


def _x(t: float, duration: float) -> float:
    return ML + (t / duration) * PW if duration > 0 else ML


def _y(v: float) -> float:
    return MT + (1.0 - max(0.0, min(1.0, v))) * PH


def render(data: dict) -> str:
    duration = max(data["duration"], 1.0)
    parts: list[str] = []
    add = parts.append

    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="{FONT}">'
    )
    add(f'<rect width="{W}" height="{H}" fill="{COLORS["bg"]}" rx="10"/>')
    add(
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" fill="none" '
        f'stroke="{COLORS["frame"]}" rx="10"/>'
    )

    # 标题 + 副标题
    source = str(data.get("source", "合成信号"))
    add(
        f'<text x="{ML}" y="30" font-size="19" font-weight="600" '
        f'fill="{COLORS["text"]}">一次会话的紧张度曲线（{html.escape(source)}）</text>'
    )
    subtitle = (
        f'{int(duration)}s · {data["n_samples"]} 个已标定样本 · 均值 {data["mean"]:.2f} · '
        f'峰值 {data["peak"]:.2f}'
    )
    backends = data.get("backends") or {}
    if backends:
        subtitle += "　｜　后端 " + " / ".join(f"{k}={v}" for k, v in backends.items())
    add(
        f'<text x="{ML}" y="48" font-size="12" fill="{COLORS["muted"]}">'
        f"{html.escape(subtitle)}</text>"
    )

    # 网格与 y 轴
    for i in range(5):
        v = i / 4.0
        y = _y(v)
        add(
            f'<line x1="{ML}" y1="{y:.1f}" x2="{ML + PW}" y2="{y:.1f}" '
            f'stroke="{COLORS["grid"]}" stroke-width="1"/>'
        )
        add(
            f'<text x="{ML - 10}" y="{y + 4:.1f}" font-size="11" text-anchor="end" '
            f'fill="{COLORS["muted"]}">{v:.2f}</text>'
        )
    add(
        f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT + PH}" '
        f'stroke="{COLORS["axis"]}" stroke-width="1"/>'
    )

    # x 轴刻度
    step = 60.0
    t = 0.0
    while t <= duration:
        x = _x(t, duration)
        add(
            f'<line x1="{x:.1f}" y1="{MT + PH}" x2="{x:.1f}" y2="{MT + PH + 4}" '
            f'stroke="{COLORS["axis"]}"/>'
        )
        add(
            f'<text x="{x:.1f}" y="{MT + PH + 18}" font-size="11" text-anchor="middle" '
            f'fill="{COLORS["muted"]}">{int(t)}</text>'
        )
        t += step
    add(
        f'<text x="{ML + PW / 2:.0f}" y="{H - 12}" font-size="11" text-anchor="middle" '
        f'fill="{COLORS["muted"]}">会话时间（秒）</text>'
    )
    add(
        f'<text x="16" y="{MT + PH / 2:.0f}" font-size="11" text-anchor="middle" '
        f'fill="{COLORS["muted"]}" transform="rotate(-90 16 {MT + PH / 2:.0f})">'
        f"紧张度（个体内相对量）</text>"
    )

    # 阈值线 + 静息基线
    for data_key, color_key, label, dash in (
        ("theta_hi", "hi", f"PAUSE 阈值 {data['theta_hi']:.2f}", "6 4"),
        ("theta_mid", "mid", f"WATCH 阈值 {data['theta_mid']:.2f}", "6 4"),
        ("baseline", "base", f"静息基线 {data['baseline']:.2f}", "2 4"),
    ):
        v = data[data_key]
        y = _y(v)
        add(
            f'<line x1="{ML}" y1="{y:.1f}" x2="{ML + PW}" y2="{y:.1f}" '
            f'stroke="{COLORS[color_key]}" stroke-width="1.4" stroke-dasharray="{dash}" '
            f'opacity="0.85"/>'
        )
        add(
            f'<text x="{ML + PW + 8}" y="{y + 4:.1f}" font-size="11" '
            f'fill="{COLORS[color_key]}">{html.escape(label)}</text>'
        )

    # 曲线 + 面积
    pts = [(round(_x(tt, duration), 1), round(_y(ss), 1)) for tt, ss in zip(data["t"], data["s"])]
    if pts:
        line = " ".join(f"{x},{y}" for x, y in pts)
        add(
            f'<polygon points="{ML},{_y(0):.1f} {line} {ML + PW},{_y(0):.1f}" '
            f'fill="{COLORS["curve_fill"]}" opacity="0.55"/>'
        )
        add(
            f'<polyline points="{line}" fill="none" stroke="{COLORS["curve"]}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    # 尖峰事件
    shown = 0
    for ts, magnitude, kind in data["events"]:
        if kind != "spike" or shown >= 8:
            continue
        x = _x(ts, duration)
        y = _y(max(0.0, min(1.0, magnitude)))
        if x < ML or x > ML + PW:
            continue
        add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{COLORS["event"]}"/>')
        shown += 1

    # 干预触发点（顶部三角标记 + 动作缩写）
    add(
        f'<line x1="{ML}" y1="{MT - 12}" x2="{ML + PW}" y2="{MT - 12}" '
        f'stroke="{COLORS["frame"]}" stroke-width="1"/>'
    )
    add(
        f'<text x="{ML - 4}" y="{MT - 15}" font-size="10" text-anchor="end" '
        f'fill="{COLORS["muted"]}">干预</text>'
    )
    short = {"coach": "C", "micro_break": "M", "pause": "P", "continue": "-"}
    for ts, action in data["interventions"]:
        x = _x(ts, duration)
        if x < ML or x > ML + PW:
            continue
        add(
            f'<polygon points="{x:.1f},{MT - 12} {x - 4.5:.1f},{MT - 20} {x + 4.5:.1f},{MT - 20}" '
            f'fill="{COLORS["interv"]}"/>'
        )
        add(
            f'<text x="{x:.1f}" y="{MT - 24}" font-size="9" text-anchor="middle" '
            f'fill="{COLORS["interv"]}">{short.get(action, "?")}</text>'
        )

    # 图例
    ly = MT + 8
    legend = [
        (COLORS["curve"], "紧张度（EWMA）", "line"),
        (COLORS["hi"], "PAUSE 阈值", "dash"),
        (COLORS["mid"], "WATCH 阈值", "dash"),
        (COLORS["base"], "静息基线", "dot"),
        (COLORS["event"], "尖峰事件", "dot"),
        (COLORS["interv"], "干预触发（C/M/P）", "tri"),
    ]
    for color, label, kind in legend:
        y = ly
        if kind == "line":
            add(f'<line x1="{ML + PW + 16}" y1="{y}" x2="{ML + PW + 40}" y2="{y}" stroke="{color}" stroke-width="2.2"/>')
        elif kind == "dash":
            add(f'<line x1="{ML + PW + 16}" y1="{y}" x2="{ML + PW + 40}" y2="{y}" stroke="{color}" stroke-width="1.6" stroke-dasharray="5 3"/>')
        elif kind == "dot":
            add(f'<circle cx="{ML + PW + 28}" cy="{y}" r="3.4" fill="{color}"/>')
        else:
            add(f'<polygon points="{ML + PW + 28},{y - 4} {ML + PW + 24},{y + 3} {ML + PW + 32},{y + 3}" fill="{color}"/>')
        add(f'<text x="{ML + PW + 46}" y="{y + 4}" font-size="11" fill="{COLORS["text"]}">{html.escape(label)}</text>')
        ly += 22

    add(
        f'<text x="{ML + PW + 16}" y="{ly + 14}" font-size="10" fill="{COLORS["muted"]}">'
        f"个体内相对量，不可跨人比较</text>"
    )
    add(
        f'<text x="{ML + PW + 16}" y="{ly + 30}" font-size="10" fill="{COLORS["muted"]}">'
        f"{html.escape(source)}，不构成效果证据</text>"
    )

    add("</svg>")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="导出一张会话曲线 SVG")
    ap.add_argument("--out", default="docs/assets/session-timeline.svg")
    ap.add_argument("--duration", type=float, default=540.0,
                    help="合成模式的会话时长（秒）")
    ap.add_argument("--stride", type=int, default=2, help="下采样步长（越大点越少）")
    ap.add_argument("--episodes", default=None,
                    help="从这里读已落盘的会话（episodes.jsonl），而不是现跑合成会话")
    ap.add_argument("--episode-id", default=None, help="配合 --episodes 指定画哪一条")
    args = ap.parse_args()

    if args.episodes:
        data = load_episode(args.episodes, args.episode_id, max(args.stride, 1))
    else:
        data = collect(args.duration, args.stride)
    svg = render(data)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    print(
        f"已写入 {out}（{len(svg) / 1024:.1f} KB，曲线点 {len(data['t'])} 个，"
        f"来源：{data.get('source')}）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
