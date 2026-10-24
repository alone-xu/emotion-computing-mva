"""命令行入口。

用法::

    python -m emotion_computing.cli demo --duration 540
    python -m emotion_computing.cli episodes
    python -m emotion_computing.cli report <episode_id>
    python -m emotion_computing.cli profile
    python -m emotion_computing.cli index-health
    python -m emotion_computing.cli reindex
    python -m emotion_computing.cli delete <episode_id> --yes
    python -m emotion_computing.cli recover
    python -m emotion_computing.cli clear --yes
    python -m emotion_computing.cli audit data --input dataset.json
    python -m emotion_computing.cli audit run --config <json>
    python -m emotion_computing.cli audit dashboard <audit_id>
    python -m emotion_computing.cli audit log

（若未安装包，请先设置 PYTHONPATH 指向 src/，或执行 pip install -e .）
"""

from __future__ import annotations

import argparse
import json
import sys


def _fix_stdout() -> None:
    """Windows 控制台默认可能是 GBK，强制 UTF-8 以免中文乱码。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emotion-computing",
        description="多模态情感计算 MVA：感知 → 快系统 → 轻量慢系统 → 记忆 → 行动",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="跑一次离线会话（合成数据）")
    demo.add_argument("--duration", type=float, default=540.0, help="会话时长（秒）")
    demo.add_argument("--task-label", default="demo", help="任务类型标签")
    demo.add_argument("--task-outcome", type=float, default=None, help="任务完成度 [0,1]")
    demo.add_argument("--data-dir", default="data")

    eps = sub.add_parser("episodes", help="列出历史会话")
    eps.add_argument("--data-dir", default="data")

    rep = sub.add_parser("report", help="打印指定会话的复盘报告")
    rep.add_argument("episode_id")
    rep.add_argument("--data-dir", default="data")

    clr = sub.add_parser("clear", help="清空全部记忆产物")
    clr.add_argument("--data-dir", default="data")
    clr.add_argument("--yes", action="store_true", help="确认删除")

    prof = sub.add_parser("profile", help="查看个体画像（含阈值学习依据）")
    prof.add_argument("--data-dir", default="data")

    ih = sub.add_parser("index-health", help="索引健康度（缺特征字段的旧行会告警）")
    ih.add_argument("--data-dir", default="data")

    rx = sub.add_parser("reindex", help="从 episodes.jsonl 重建 index.jsonl + index.sqlite")
    rx.add_argument("--data-dir", default="data")

    dele = sub.add_parser("delete", help="删除单条历史会话（不再只能全清）")
    dele.add_argument("episode_id")
    dele.add_argument("--data-dir", default="data")
    dele.add_argument("--yes", action="store_true", help="确认删除")

    rec = sub.add_parser("recover", help="把崩溃残留的 running 会话标为 aborted")
    rec.add_argument("--data-dir", default="data")

    # ----- 审核子命令 -----
    audit = sub.add_parser("audit", help="审核流水线（D/M/F/V/I/L 6 大需求）")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)

    ad = audit_sub.add_parser("data", help="D-# 数据层审核（只读，不跑 MVA）")
    ad.add_argument("--input", required=True, help="数据集 JSON 路径")
    ad.add_argument("--output-dir", default="data/audit")

    ar = audit_sub.add_parser("run", help="审核流水线（输入 MVA 输出 + 金标准）")
    ar.add_argument("--config", required=True, help="审核输入 JSON（含 dataset / outputs / ablation）")
    ar.add_argument("--output-dir", default="data/audit")
    ar.add_argument("--reviewer-id", default="cli")

    adash = audit_sub.add_parser("dashboard", help="打开指定 audit_id 的 dashboard.html")
    adash.add_argument("audit_id")
    adash.add_argument("--output-dir", default="data/audit")

    alog = audit_sub.add_parser("log", help="查看 _audit_log.jsonl（可按样本/审核者过滤）")
    alog.add_argument("--sample-id", default=None)
    alog.add_argument("--reviewer-id", default=None)
    alog.add_argument("--output-dir", default="data/audit")

    return parser


def cmd_demo(args: argparse.Namespace) -> int:
    from .pipeline import SessionPipeline

    pipe = SessionPipeline(data_dir=args.data_dir, task_label=args.task_label)
    print(f"感知后端组合：{pipe.backends()}")
    print(f"基线期：{pipe.config.baseline_seconds:.0f}s；预测视界：{pipe.config.forecast_horizon:.0f}s")
    result = pipe.run_offline(duration=args.duration, task_outcome=args.task_outcome)
    episode = result["episode"]
    print("\n会话摘要：")
    print(json.dumps(episode.summary, ensure_ascii=False, indent=2))
    print(f"\n复盘报告：{result['report_path']}")
    print(f"data/ 内容：{result['outputs']}")
    return 0


def cmd_episodes(args: argparse.Namespace) -> int:
    from .memory import EpisodeStore

    rows = EpisodeStore(args.data_dir).load_index()
    if not rows:
        print("（暂无历史会话）")
        return 0
    print(f"{'episode_id':<14}{'任务':<10}{'均值':>8}{'峰值':>8}{'尖峰':>6}{'完成度':>8}")
    for r in rows:
        print(
            f"{r.get('episode_id', ''):<14}{str(r.get('task_label', '')):<10}"
            f"{r.get('mean', 0.0):>8.3f}{r.get('max', 0.0):>8.3f}"
            f"{r.get('n_spikes', 0):>6}{str(r.get('task_outcome', '-')):>8}"
        )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .memory import EpisodeStore

    store = EpisodeStore(args.data_dir)
    path = store.reports_dir / f"{args.episode_id}.md"
    if not path.exists():
        print(f"未找到报告：{path}")
        return 1
    print(path.read_text("utf-8"))
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    from .memory import EpisodeStore

    if not args.yes:
        print("这会删除全部 episodes/index/profile 与报告。确认请加 --yes")
        return 1
    store = EpisodeStore(args.data_dir)
    store.clear()
    store.close()
    print("已清空记忆产物。")
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    from .memory import EpisodeStore

    store = EpisodeStore(args.data_dir)
    profile = store.load_profile()
    store.close()
    print(json.dumps(profile.as_dict(), ensure_ascii=False, indent=2))
    print("\n—— 阈值依据 ——")
    evidence = profile.threshold_evidence or {}
    if not evidence:
        print("（尚无学习记录；跑过至少 1 次会话后才会写入）")
    for key in ("theta_mid", "theta_hi", "tau_rev"):
        item = evidence.get(key)
        if item:
            print(f"  {key}: {item.get('value')}  ← {item.get('rule')} (n={item.get('n')})")
    if evidence.get("learned_at"):
        print(f"  学习时间: {evidence['learned_at']}")
    return 0


def cmd_index_health(args: argparse.Namespace) -> int:
    from .memory import EpisodeStore

    store = EpisodeStore(args.data_dir)
    health = store.index_health()
    store.close()
    print(json.dumps(health, ensure_ascii=False, indent=2))
    if health["skipped_rows"]:
        print(
            f"\n⚠ 有 {health['skipped_rows']} 行缺少特征字段，检索时不可见。"
            "运行 `reindex` 重建。"
        )
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    from .memory import EpisodeStore

    store = EpisodeStore(args.data_dir)
    stats = store.reindex()
    store.close()
    print(f"索引已重建：{json.dumps(stats, ensure_ascii=False)}")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    from .memory import EpisodeStore

    if not args.yes:
        print(f"这会删除会话 {args.episode_id} 及其索引与报告。确认请加 --yes")
        return 1
    store = EpisodeStore(args.data_dir)
    ok = store.delete_episode(args.episode_id)
    store.close()
    print("已删除。" if ok else "未找到该会话。")
    return 0 if ok else 1


def cmd_recover(args: argparse.Namespace) -> int:
    from .memory import EpisodeStore

    store = EpisodeStore(args.data_dir, auto_recover=False)
    n = store.recover()
    store.close()
    print(f"把 {n} 个未正常结束的会话标记为 aborted（样本保留）。" if n else "没有残留的 running 会话。")
    return 0


# --------------------------------------------------------------------------
# 审核子命令
# --------------------------------------------------------------------------


def cmd_audit_data(args: argparse.Namespace) -> int:
    from .audit.data_audit import audit_dataset
    from .contracts import GoldDataset, GoldLabel

    raw = json.loads(open(args.input, encoding="utf-8").read())
    samples = tuple(GoldLabel(**s) for s in raw.get("samples", []))
    ds = GoldDataset(name=raw.get("name", "cli"), samples=samples)
    rep = audit_dataset(ds)
    out_dir = Path_lib(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"data_qa_{ds.name}.json"
    out.write_text(json.dumps(rep.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"数据层审核完成：verdict={rep.overall_verdict}")
    print(f"  α = {rep.consistency.alpha:.4f}")
    print(f"  抽检覆盖率 = {rep.spot_check.coverage * 100:.1f}%")
    print(f"  写入：{out}")
    return 0


def cmd_audit_run(args: argparse.Namespace) -> int:
    from .audit.runner import AuditRunner, AuditRunnerConfig
    from .audit.data_audit import QASpotCheckEntry
    from .audit.module_evaluator import MVASampleOutput
    from .contracts import GoldDataset, GoldLabel

    raw = json.loads(open(args.config, encoding="utf-8").read())
    samples = tuple(GoldLabel(**s) for s in raw["dataset"]["samples"])
    ds = GoldDataset(name=raw["dataset"].get("name", "cli"), samples=samples)

    outputs = [MVASampleOutput(**o) for o in raw["outputs"]]

    ablation_outputs = None
    if "ablation" in raw:
        ablation_outputs = {k: [MVASampleOutput(**x) for x in v] for k, v in raw["ablation"].items()}

    spot_assignments = None
    spot_sample_ids = ()
    if "spot_check" in raw:
        spot_sample_ids = tuple(raw["spot_check"].get("sample_ids", []))
        spot_assignments = {
            s["sample_id"]: QASpotCheckEntry(**{k: v for k, v in s.items() if k in QASpotCheckEntry.__dataclass_fields__})
            for s in raw["spot_check"].get("assignments", [])
        }

    cfg = AuditRunnerConfig(
        audit_id=raw.get("audit_id", ""),
        output_dir=args.output_dir,
        reviewer_id=args.reviewer_id,
    )
    runner = AuditRunner(cfg)
    rep = runner.run(
        ds, outputs,
        labels_matrix=raw.get("labels_matrix"),
        spot_sample_ids=spot_sample_ids,
        spot_assignments=spot_assignments,
        ablation_outputs=ablation_outputs,
        iterations=raw.get("iterations", []),
    )
    print(f"审核完成：audit_id={rep.audit_id}")
    print(f"  错误样本：{rep.n_errors} / {rep.n_samples}")
    print(f"  报告：{Path_lib(args.output_dir) / rep.audit_id / 'audit_report.md'}")
    print(f"  仪表盘：{Path_lib(args.output_dir) / rep.audit_id / 'dashboard.html'}")
    return 0


def cmd_audit_dashboard(args: argparse.Namespace) -> int:
    """打印 dashboard.html 路径（浏览器打开）。"""
    from pathlib import Path
    p = Path_lib(args.output_dir) / args.audit_id / "dashboard.html"
    if not p.exists():
        print(f"未找到：{p}")
        return 1
    print(p)
    return 0


def cmd_audit_log(args: argparse.Namespace) -> int:
    from .audit.log import AuditLogStore

    store = AuditLogStore(Path_lib(args.output_dir) / "_audit_log.jsonl")
    rows = store.load()
    if args.sample_id:
        rows = [r for r in rows if r.sample_id == args.sample_id]
    if args.reviewer_id:
        rows = [r for r in rows if r.reviewer_id == args.reviewer_id]
    if not rows:
        print("（无匹配日志）")
        return 0
    for r in rows:
        print(json.dumps(r.as_dict(), ensure_ascii=False))
    return 0


def Path_lib(p: str):
    """Local lazy import to keep top imports light."""
    from pathlib import Path
    return Path(p)


def main(argv: list[str] | None = None) -> int:
    _fix_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "demo": cmd_demo,
        "episodes": cmd_episodes,
        "report": cmd_report,
        "clear": cmd_clear,
        "profile": cmd_profile,
        "index-health": cmd_index_health,
        "reindex": cmd_reindex,
        "delete": cmd_delete,
        "recover": cmd_recover,
        "audit": _dispatch_audit,
    }
    return handlers[args.command](args)


def _dispatch_audit(args: argparse.Namespace) -> int:
    table = {
        "data": cmd_audit_data,
        "run": cmd_audit_run,
        "dashboard": cmd_audit_dashboard,
        "log": cmd_audit_log,
    }
    return table[args.audit_command](args)


if __name__ == "__main__":
    raise SystemExit(main())
