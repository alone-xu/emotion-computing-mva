"""跨层数据契约（全系统唯一的事实来源）。

设计规则（docs/02-architecture.md §4）：
  * 层与层之间只通过本文件的类型通信，禁止裸 dict 传递业务数据。
  * 契约字段只增不改；新增字段必须带默认值。
  * 每个面向"结论"的类型都必须携带可解释信息（features / contributions / reasons）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------


def to_dict(obj: Any) -> dict:
    """把契约对象递归转换成可 JSON 序列化的 dict。"""
    return asdict(obj)


def _frozen_dict(d: Mapping[str, float]) -> dict[str, float]:
    return {str(k): float(v) for k, v in d.items()}


# --------------------------------------------------------------------------
# ① 感知层
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModalityObservation:
    """单个模态在某一时刻的观测结果。

    这是感知层对外的唯一出口。``backend`` 字段必须如实反映
    **本帧真正生效的实现**（mediapipe / haar / librosa / lexicon / synthetic …），
    这样"系统到底跑了什么"永远可追溯（架构文档 §7）。
    """

    modality: str  # "face" | "speech" | "text"
    timestamp: float
    score: float  # 该模态的紧张度分项，已压缩到 [0, 1]
    confidence: float  # 数据质量置信度，[0, 1]
    available: bool
    backend: str
    features: dict[str, float] = field(default_factory=dict)
    note: str = ""  # 人类可读的补充说明（如缺失原因）

    def __post_init__(self) -> None:
        # frozen dataclass 只能用 object.__setattr__ 做裁剪
        object.__setattr__(self, "score", float(min(1.0, max(0.0, self.score))))
        object.__setattr__(
            self, "confidence", float(min(1.0, max(0.0, self.confidence)))
        )

    def as_dict(self) -> dict:
        d = to_dict(self)
        d["features"] = _frozen_dict(self.features)
        return d


@dataclass(frozen=True)
class PerceptionFrame:
    """一帧的多模态融合结果。"""

    timestamp: float
    observations: tuple[ModalityObservation, ...]
    fused_raw: float  # 融合后的原始紧张度 [0, 1]
    normalized: float  # 基线归一化后的紧张度 [0, 1]（对外主值）
    contributions: dict[str, float] = field(default_factory=dict)  # 各模态贡献占比

    def available_modalities(self) -> tuple[str, ...]:
        return tuple(o.modality for o in self.observations if o.available)

    def backend_map(self) -> dict[str, str]:
        return {o.modality: o.backend for o in self.observations}

    def as_dict(self) -> dict:
        d = to_dict(self)
        d["observations"] = [o.as_dict() for o in self.observations]
        d["contributions"] = _frozen_dict(self.contributions)
        return d


# --------------------------------------------------------------------------
# ② 快系统
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StressSample:
    """快系统对外的实时紧张度样本。"""

    timestamp: float
    raw: float  # 融合后的瞬时值
    smoothed: float  # EWMA 平滑值 —— 系统对外的主实时值
    confidence: float
    contributions: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = to_dict(self)
        d["contributions"] = _frozen_dict(self.contributions)
        return d


@dataclass(frozen=True)
class StressEvent:
    """快系统检测到的事件。"""

    timestamp: float
    kind: str  # "spike" | "rise" | "fall" | "sustained_high"
    magnitude: float
    dominant_modality: str
    detail: str = ""

    def as_dict(self) -> dict:
        return to_dict(self)


# --------------------------------------------------------------------------
# ③ 轻量慢系统
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Forecast:
    """未来一段时间紧张度的预测。

    ``validated`` 恒为 False（MVA 阶段未经真实数据回测）——这是需求文档 §7.2
    的诚实性要求在代码层面的落地，不得为了"好看"而改成 True。
    """

    horizon_s: float
    point: float
    lower: float
    upper: float
    slope: float  # 当前趋势斜率（每分钟）
    baseline: float  # 个体基线（均值回复目标）
    method: str
    validated: bool = False

    def as_dict(self) -> dict:
        return to_dict(self)


@dataclass(frozen=True)
class CapacityAssessment:
    """"是否还能继续执行任务"的评估结果。"""

    state: str  # "OK" | "WATCH" | "PAUSE"
    score: float  # [0, 1]，越大越能继续
    forecast: Forecast
    resilience: float  # 恢复力估计 [0, 1]
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        d = to_dict(self)
        d["forecast"] = self.forecast.as_dict()
        d["reasons"] = list(self.reasons)
        return d


# --------------------------------------------------------------------------
# ④ 记忆层
# --------------------------------------------------------------------------


#: Episode 结构版本。加字段必须递增，并在 ``Episode.from_dict`` 里处理旧版本。
EPISODE_SCHEMA_VERSION = 2

#: PersonalProfile 结构版本。
PROFILE_SCHEMA_VERSION = 2

#: Episode 生命周期状态（崩溃恢复用）。
#: ``running`` 是会话进行中写入的检查点；进程异常退出后由 store 启动时改为 ``aborted``。
EPISODE_STATUSES: tuple[str, ...] = ("running", "done", "aborted")


@dataclass
class Episode:
    """一次完整会话的数据单元（记忆层基本单位）。

    隐私红线：本结构**不包含**任何原始视频帧、音频波形或原始文本。
    只保留特征、统计量与结论（需求文档 M-6）。
    ``notes`` 只允许写结构化短标签，不允许写用户原话（见 memory/store.py 的隐私门）。
    """

    episode_id: str
    started_at: str
    ended_at: str | None = None
    task_label: str = "default"
    status: str = "done"  # running | done | aborted
    schema_version: int = EPISODE_SCHEMA_VERSION
    #: 归一化输出尺度上的静息点（结构常数，用于恢复时间判定）
    baseline_mu: float = 0.0
    #: 静息期**融合原始值**的离散度（个体校准记录）
    baseline_sigma: float = 0.0
    #: 静息期**融合原始值**的中心。与 baseline_sigma 同域，
    #: 这一对 (μ, σ) 才是"这个人的静息水平"，会被 EMA 回写进画像。
    baseline_raw_mu: float = 0.0
    #: 静息期融合原始值的离散度（与 baseline_raw_mu 配对；= 融合引擎的 fused_stats[1]）
    baseline_raw_sigma: float = 0.0
    backends: dict[str, str] = field(default_factory=dict)
    samples: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    interventions: list[dict] = field(default_factory=list)
    capacity_timeline: list[dict] = field(default_factory=list)
    task_outcome: float | None = None
    summary: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    #: 未来版本新增、本版本不认识的字段原样保留（不静默丢弃，P2-#11）
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return to_dict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Episode":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        unknown = {k: v for k, v in data.items() if k not in known}
        # 旧版本（schema_version 缺失 = 1）没有 status 字段，默认视为已正常结束
        kwargs.setdefault("schema_version", 1)
        kwargs.setdefault("status", "done")
        if unknown:
            extra = dict(kwargs.get("extra") or {})
            extra.update(unknown)
            kwargs["extra"] = extra
        if kwargs["status"] not in EPISODE_STATUSES:
            kwargs["status"] = "done"
        return cls(**kwargs)


@dataclass
class PersonalProfile:
    """跨会话的个体画像（由记忆层回写，供下一次会话加载）。

    实现需求 M-5 与场景 3："越用越懂这个人"。

    回写规则（memory/pipeline 一致实现）：
      * ``baseline_mu`` / ``baseline_sigma``：**静息期融合原始域的 (μ, σ) 对**，
        按样本量门控的 EMA 平滑更新，绝不用单次会话直接覆盖
        （一次异常会话不会永久拉高基线）。这两个字段必须同域使用 ——
        拿归一化尺度的 μ 配融合尺度的 σ 做标准化是错的。
      * ``theta_mid`` / ``theta_hi``：取最近 N 次**同任务标签**会话均值的分位数，
        并把计算依据写入 ``threshold_evidence``（可解释，不允许"黑箱阈值"）；
      * ``tau_rev``：历史恢复时间中位数 × 1.2；
      * ``intervention_effectiveness``：干预前后 ΔS 的 EMA。
    """

    sessions: int = 0
    baseline_mu: float = 0.0
    baseline_sigma: float = 0.0
    theta_mid: float = 0.55
    theta_hi: float = 0.75
    tau_rev: float = 90.0
    resilience: float = 0.5
    intervention_effectiveness: dict[str, float] = field(default_factory=dict)
    #: 每个自适应阈值的计算依据（输入样本、规则、更新时间）——可解释性要求
    threshold_evidence: dict = field(default_factory=dict)
    schema_version: int = PROFILE_SCHEMA_VERSION
    updated_at: str = ""
    #: 未来版本新增字段原样保留
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return to_dict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PersonalProfile":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        unknown = {k: v for k, v in data.items() if k not in known}
        kwargs.setdefault("schema_version", 1)
        kwargs.setdefault("threshold_evidence", {})
        kwargs.setdefault("updated_at", "")
        if unknown:
            extra = dict(kwargs.get("extra") or {})
            extra.update(unknown)
            kwargs["extra"] = extra
        # 旧画像可能把数值存成整数/字符串，统一成 float，避免下游除零或比较异常
        for key in ("baseline_mu", "baseline_sigma", "theta_mid", "theta_hi", "tau_rev", "resilience"):
            try:
                kwargs[key] = float(kwargs.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                kwargs.pop(key, None)
        try:
            kwargs["sessions"] = int(kwargs.get("sessions", 0) or 0)
        except (TypeError, ValueError):
            kwargs["sessions"] = 0
        return cls(**kwargs)


# --------------------------------------------------------------------------
# ⑤ 行动层
# --------------------------------------------------------------------------


#: 允许的动作集合，按"干预强度"从低到高排列。
ACTION_ORDER: tuple[str, ...] = ("continue", "coach", "micro_break", "pause")


@dataclass(frozen=True)
class ActionDecision:
    """行动层的输出。"""

    timestamp: float
    action: str  # "continue" | "coach" | "micro_break" | "pause"
    intervention: str | None
    message: str
    rationale: str
    urgent: bool = False

    def as_dict(self) -> dict:
        return to_dict(self)


@dataclass(frozen=True)
class InterventionRecord:
    """一次干预的执行记录（写进 Episode，用于后续评估有效性）。"""

    timestamp: float
    action: str
    intervention: str
    stress_before: float
    stress_after: float | None = None

    def as_dict(self) -> dict:
        return to_dict(self)


# --------------------------------------------------------------------------
# 复盘
# --------------------------------------------------------------------------


@dataclass
class ReviewReport:
    """会话复盘报告（记忆层产出）。"""

    episode_id: str
    generated_at: str
    timeline_rows: list[dict] = field(default_factory=list)
    peaks: list[dict] = field(default_factory=list)
    recovery: dict = field(default_factory=dict)
    outcome_link: dict = field(default_factory=dict)
    cross_session: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    backends: dict[str, str] = field(default_factory=dict)
    disclaimer: str = ""

    def as_dict(self) -> dict:
        return to_dict(self)

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append(f"# 会话复盘报告 · {self.episode_id}")
        lines.append("")
        lines.append(f"- 生成时间：{self.generated_at}")
        lines.append(f"- 感知后端：{self.backends}")
        lines.append("")

        lines.append("## 1. 紧张度演化时间线")
        lines.append("")
        lines.append("| 时刻(s) | 紧张度 | 顶部模态 | 事件 |")
        lines.append("|---:|---:|---|---|")
        for row in self.timeline_rows:
            lines.append(
                f"| {row.get('t', 0):.0f} | {row.get('stress', 0):.3f} | "
                f"{row.get('top_modality', '-')} | {row.get('event', '')} |"
            )
        lines.append("")

        lines.append("## 2. 峰值归因")
        lines.append("")
        if self.peaks:
            lines.append("| 时刻(s) | 峰值 | 主导模态 | 各模态贡献 |")
            lines.append("|---:|---:|---|---|")
            for p in self.peaks:
                lines.append(
                    f"| {p.get('t', 0):.0f} | {p.get('stress', 0):.3f} | "
                    f"{p.get('dominant', '-')} | {p.get('contributions', {})} |"
                )
        else:
            lines.append("_本次会话未检测到显著峰值。_")
        lines.append("")

        lines.append("## 3. 恢复画像")
        lines.append("")
        for k, v in self.recovery.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

        lines.append("## 4. 与任务完成度的关联")
        lines.append("")
        if self.outcome_link:
            for k, v in self.outcome_link.items():
                lines.append(f"- {k}: {v}")
        else:
            lines.append("_本次会话未注入 task_outcome，跳过关联分析。_")
        lines.append("")

        lines.append("## 5. 跨会话对比（同类任务）")
        lines.append("")
        if self.cross_session:
            lines.append("| 会话 | 平均 | 峰值 | 尖峰数 | 完成度 |")
            lines.append("|---|---:|---:|---:|---:|")
            for r in self.cross_session:
                lines.append(
                    f"| {r.get('episode_id')} | {r.get('mean', 0):.3f} | "
                    f"{r.get('max', 0):.3f} | {r.get('n_spikes', 0)} | "
                    f"{r.get('task_outcome', '-')} |"
                )
        else:
            lines.append("_暂无同类任务的历史会话。_")
        lines.append("")

        lines.append("## 6. 建议")
        lines.append("")
        for s in self.suggestions:
            lines.append(f"- {s}")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append(self.disclaimer)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# 审核层契约（追加于 03-audit-requirements.md / 04-audit-architecture.md）
# 既不替换也不修改既有 dataclass；MVA 业务代码不受影响。
# --------------------------------------------------------------------------

#: 失败类型枚举（03 §F-1）
FAILURE_TYPES: tuple[str, ...] = (
    "perception_failure",
    "modality_sabotage",
    "slow_eval_failure",
    "memory_interference",
    "action_failure",
    "data_problem",
    "combined_failure",
)

#: 失败类型的人类可读中文标签（03 §F-3 报告用）
FAILURE_LABELS: dict[str, str] = {
    "perception_failure":   "感知失败",
    "modality_sabotage":   "模态破坏",
    "slow_eval_failure":   "评估失败",
    "memory_interference": "记忆干扰",
    "action_failure":      "策略失败",
    "data_problem":        "数据问题",
    "combined_failure":    "组合失败",
}

#: 失败类型 → 推荐修复方向（03 §F-1）
FAILURE_REPAIR: dict[str, str] = {
    "perception_failure":   "提升面部/语音特征提取",
    "modality_sabotage":   "修改融合策略，增加置信度校准",
    "slow_eval_failure":   "优化慢系统的预测/状态机提示",
    "memory_interference": "改进检索相关性阈值",
    "action_failure":      "调整干预策略生成逻辑",
    "data_problem":        "修正标注或剔除模糊样本",
    "combined_failure":    "重新设计融合逻辑",
}


@dataclass(frozen=True)
class GoldLabel:
    """单个样本的金标准标注（审核输入）。"""

    sample_id: str
    modality_labels: dict[str, float]  # {"face":0.3,"speech":0.7,"text":0.5}
    stress: float                      # 端到端金标准紧张度 ∈ [0,1]
    state: str = "OK"                  # "OK" | "WATCH" | "PAUSE"
    action_appropriateness: int = 3    # 1-5，1=最不恰当，5=最恰当
    confidence: float = 1.0            # 标注者置信度 ∈ [0,1]
    annotator_id: str = "anon"
    notes: str = ""

    def as_dict(self) -> dict:
        d = to_dict(self)
        d["modality_labels"] = _frozen_dict(self.modality_labels)
        return d


@dataclass
class GoldDataset:
    """金标准测试集（审核输入）。"""

    name: str
    samples: tuple[GoldLabel, ...] = ()
    modality_availability: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> dict:
        return to_dict(self)


@dataclass(frozen=True)
class ModuleMetric:
    """单个模块的单个指标。"""

    module: str
    metric_name: str
    value: float
    n: int
    threshold: float | None
    status: str  # "ok" | "warn" | "fail" | "unverified"

    def as_dict(self) -> dict:
        return to_dict(self)


@dataclass
class ModuleEvaluation:
    """一个模块的完整评估。"""

    module: str
    metrics: tuple[ModuleMetric, ...] = ()
    sample_failures: tuple[str, ...] = ()  # 失败 sample_id 列表

    def as_dict(self) -> dict:
        return to_dict(self)

    def worst_status(self) -> str:
        order = {"ok": 0, "warn": 1, "unverified": 2, "fail": 3}
        if not self.metrics:
            return "unverified"
        return max((m.status for m in self.metrics), key=lambda s: order.get(s, 0))


@dataclass(frozen=True)
class ModalityAttribution:
    """M-3：单样本的模态级归因。"""

    sample_id: str
    face: float
    speech: float
    text: float
    fused: float
    gold: float
    dominant_modality: str
    saboteur: str | None
    reason: str

    def as_dict(self) -> dict:
        return to_dict(self)


@dataclass
class FailureCase:
    """F-1：单样本失败归因。"""

    sample_id: str
    error: float
    system_attribution: str
    human_attribution: str | None = None
    correction_reason: str | None = None
    modality_attribution: ModalityAttribution | None = None
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = to_dict(self)
        if self.modality_attribution is not None:
            d["modality_attribution"] = self.modality_attribution.as_dict()
        return d

    @property
    def final_attribution(self) -> str:
        return self.human_attribution or self.system_attribution


@dataclass
class FailureDistribution:
    """F-3：失败类型分布。"""

    counts: dict[str, int] = field(default_factory=dict)
    percentages: dict[str, float] = field(default_factory=dict)
    top_repair: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return to_dict(self)

    def total(self) -> int:
        return sum(self.counts.values())


@dataclass(frozen=True)
class AuditLogEntry:
    """L-1：单条审核操作日志。"""

    log_id: str
    audit_id: str
    reviewer_id: str
    sample_id: str
    system_attribution: str
    human_attribution: str | None
    correction_reason: str | None
    follow_up: str  # "none" | "bug_filed" | "config_tuned" | "data_relabeled" | "ignored"
    timestamp: str

    def as_dict(self) -> dict:
        return to_dict(self)


@dataclass
class AuditReport:
    """一次审核会话的完整产物。"""

    audit_id: str
    generated_at: str
    dataset_name: str
    n_samples: int
    n_errors: int
    data_qa: dict = field(default_factory=dict)
    module_evaluations: tuple[ModuleEvaluation, ...] = ()
    ablation: dict[str, float] = field(default_factory=dict)
    failure_distribution: FailureDistribution = field(default_factory=FailureDistribution)
    failure_cases: tuple[FailureCase, ...] = ()
    iterations: tuple[dict, ...] = ()  # 历史迭代对比
    disclaimer: str = ""
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return to_dict(self)
