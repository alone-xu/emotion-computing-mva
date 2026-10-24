"""端到端：把 RAVDESS 真实数据跑过的 MVA 会话交给 audit 流水线。

输入：
  * data/real_data_runs/episodes.jsonl  ← scripts/validate_real_data.py 写出
  * data/real_data_runs/profile.json

输出：
  * data/audit/audit_ravdess/<audit_id>/audit_report.md
  * data/audit/audit_ravdess/<audit_id>/dashboard.html
  * data/audit/audit_ravdess/<audit_id>/modality_attributions.json
  * data/audit/audit_ravdess/<audit_id>/module_evaluations.json
  * data/audit/audit_ravdess/<audit_id>/ablation_results.json
  * data/audit/audit_ravdess/<audit_id>/failure_cases.jsonl
  * data/audit/audit_ravdess/<audit_id>/failure_distribution.json
  * data/audit/audit_ravdess/<audit_id>/data_qa.json
  * data/audit/_iterations.jsonl
  * data/audit/_audit_log.jsonl
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from emotion_computing.audit.data_audit import QASpotCheckEntry
from emotion_computing.audit.module_evaluator import MVASampleOutput
from emotion_computing.audit.runner import AuditRunner, AuditRunnerConfig
from emotion_computing.contracts import GoldDataset, GoldLabel


RAVDESS_DATA = ROOT / "data" / "real_data_runs"
AUDIT_DIR = ROOT / "data" / "audit" / "audit_ravdess"


def _stress_to_state(stress: float) -> str:
    if stress < 0.45:
        return "OK"
    if stress < 0.70:
        return "WATCH"
    return "PAUSE"


def _load_episodes() -> list[dict[str, Any]]:
    path = RAVDESS_DATA / "episodes.jsonl"
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def episode_to_outputs(ep: dict) -> list[MVASampleOutput]:
    """把 Episode 的 samples 转成 MVASampleOutput 列表。

    注入已知错误以验证归因链路：
      * i % 5 == 0：高估 0.4 → 触发 state 跨级（应归因到 slow_eval / action）
      * i % 5 == 1：构造破坏者（face 高置信反向）→ 应归因到 modality_sabotage
      * i % 5 == 2：单模态强偏离但整体 fused 偏 → 应归因到 perception_failure
      * 其它：完美预测
    """
    out: list[MVASampleOutput] = []
    for i, s in enumerate(ep.get("samples", [])):
        stress = float(s.get("smoothed", 0.0))
        conf = float(s.get("confidence", 0.5))
        contributions = s.get("contributions", {}) or {}

        if i % 5 == 0:
            # 高估型错误：fused 偏离 0.40+ → 状态机跨级
            perturbed = max(0.0, min(1.0, stress + 0.45))
            state = "PAUSE" if perturbed > 0.70 else "WATCH"
            out.append(MVASampleOutput(
                sample_id=f"{ep['episode_id']}_t{s.get('t', 0):.0f}",
                stress=perturbed, state=state, action="continue",
                modality_scores=dict(contributions),
                modality_confidence={"face": conf, "speech": conf, "text": conf},
            ))
        elif i % 5 == 1:
            # 模态破坏者：fused 偏低，face 强说高（高置信）→ face 是破坏者
            perturbed = max(0.0, stress - 0.30)
            # 让 gold 端的 face 模态值明显低于 fused 方向，conf 高
            # attribute_sample 用 sample.modality_labels[face] 作为"face 的金标准值"
            # 我们这里没法改 gold，但让"扰动"在 fused 体现即可；具体归因由 F-2 步骤判定
            out.append(MVASampleOutput(
                sample_id=f"{ep['episode_id']}_t{s.get('t', 0):.0f}",
                stress=perturbed,
                state="OK",
                action="continue",
                modality_scores={"face": stress + 0.4, "speech": 0.5, "text": 0.5},
                modality_confidence={"face": 0.9, "speech": 0.5, "text": 0.5},
            ))
        elif i % 5 == 2:
            # 感知失败：模态值与 gold 差距大
            perturbed = max(0.0, min(1.0, stress + 0.30))
            out.append(MVASampleOutput(
                sample_id=f"{ep['episode_id']}_t{s.get('t', 0):.0f}",
                stress=perturbed,
                state="WATCH",
                action="continue",
                modality_scores={"face": 0.9, "speech": 0.1, "text": 0.5},
                modality_confidence={"face": 0.3, "speech": 0.3, "text": 0.3},
            ))
        else:
            # 完美预测
            out.append(MVASampleOutput(
                sample_id=f"{ep['episode_id']}_t{s.get('t', 0):.0f}",
                stress=stress,
                state=_stress_to_state(stress),
                action="continue",
                modality_scores=dict(contributions),
                modality_confidence={"face": conf, "speech": conf, "text": conf},
            ))
    return out


def episode_to_gold(ep: dict) -> list[GoldLabel]:
    """用样本的 smoothed stress 与 phase 标签做金标准。

    per-modality gold 设为有差异的值（face 高、speech 中、text 低），
    这样 attribute_sample 才能基于模态方向判断"破坏者"。
    """
    out: list[GoldLabel] = []
    for s in ep.get("samples", []):
        stress = float(s.get("smoothed", 0.0))
        # 三个模态稍微错开
        ml = {
            "face": min(1.0, max(0.0, stress + 0.10)),
            "speech": stress,
            "text": min(1.0, max(0.0, stress - 0.05)),
        }
        out.append(GoldLabel(
            sample_id=f"{ep['episode_id']}_t{s.get('t', 0):.0f}",
            modality_labels=ml,
            stress=stress,
            state=_stress_to_state(stress),
            action_appropriateness=3,
            annotator_id="synthetic_ravdess",
        ))
    return out


def main() -> int:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    episodes = _load_episodes()
    if not episodes:
        print(f"未找到 {RAVDESS_DATA / 'episodes.jsonl'}；请先跑 scripts/validate_real_data.py")
        return 1

    # 把多会话合并成一个金标准集（最多 200 样本）
    all_gold: list[GoldLabel] = []
    all_outputs: list[MVASampleOutput] = []
    for ep in episodes:
        all_gold.extend(episode_to_gold(ep))
        all_outputs.extend(episode_to_outputs(ep))
    all_gold = all_gold[:200]
    all_outputs = all_outputs[:200]
    if not all_gold:
        print("无可用样本")
        return 1

    dataset = GoldDataset(name="ravdess", samples=tuple(all_gold))

    # 模拟抽检（D-2）：随机抽 12% 样本
    import random
    random.seed(42)
    spot_count = max(1, int(0.12 * len(all_gold)))
    spot_ids = random.sample([g.sample_id for g in all_gold], spot_count)
    spot_assignments = {
        sid: QASpotCheckEntry(
            sample_id=sid, is_correct=True,
            reviewer_id="auto_checker",
            reviewed_at="2026-10-10T09:00:00",
        )
        for sid in spot_ids
    }

    # 构造消融实验：P_face_only 模式（用 stress 当 face 模态分数）
    face_only = [
        MVASampleOutput(
            sample_id=o.sample_id, stress=o.stress,
            state=o.state, action=o.action,
            modality_scores={"face": o.stress, "speech": 0.5, "text": 0.5},
            modality_confidence={"face": 0.5, "speech": 0.5, "text": 0.5},
        )
        for o in all_outputs
    ]
    # 构造消融：F_off（不平滑）→ 加噪声
    f_off = [
        MVASampleOutput(
            sample_id=o.sample_id,
            stress=max(0.0, min(1.0, o.stress + 0.1 * (1 if i % 2 == 0 else -1))),
            state=o.state, action=o.action,
        )
        for i, o in enumerate(all_outputs)
    ]

    cfg = AuditRunnerConfig(
        audit_id=uuid.uuid4().hex[:10],
        output_dir=str(AUDIT_DIR),  # 让 audit_id 子目录建在 audit_ravdess/ 下
        abort_on_consistency_fail=False,
    )
    runner = AuditRunner(cfg)
    rep = runner.run(
        dataset, all_outputs,
        spot_sample_ids=tuple(spot_ids),
        spot_assignments=spot_assignments,
        ablation_outputs={
            "P_face_only": face_only,
            "F_off": f_off,
        },
    )
    print(f"\n[OK] audit_id={rep.audit_id}")
    print(f"  样本: {rep.n_samples}, 错误: {rep.n_errors}")
    print(f"  失败分布: {json.dumps({k: v for k, v in rep.failure_distribution.counts.items() if v > 0}, ensure_ascii=False)}")
    print(f"  修复方向: {rep.failure_distribution.top_repair}")
    print(f"  报告: {AUDIT_DIR / rep.audit_id / 'audit_report.md'}")
    print(f"  仪表盘: {AUDIT_DIR / rep.audit_id / 'dashboard.html'}")
    print(f"  失败清单: {AUDIT_DIR / rep.audit_id / 'failure_cases.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
