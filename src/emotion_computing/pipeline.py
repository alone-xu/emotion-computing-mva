"""五层编排：一次在线会话的主循环。

数据流（严格按需求文档规定的顺序）：

    感知 → 快系统 → 轻量慢系统 → 记忆 → 行动

节奏（架构文档 §6）：
  * 每一步（默认 1Hz）：感知 → 融合 → 快系统 → 事件检测
  * 每 1s：样本入记忆缓冲
  * 每 60s：会话检查点（``status="running"`` 快照落盘，崩溃不丢数据）
  * 每 5s：慢系统（趋势/预测/可继续性）→ 记忆检索 → 行动层决策
  * 会话末：写 Episode → 生成复盘报告 → 回写个体画像

基线期（前 baseline_seconds 秒）只做标定，**其样本不进入慢系统与记忆**，
因为标定前后的尺度不同（标定前是原始融合值，标定后是个体内相对量）。

记忆回流（评审 P1-#7）
----------------------
记忆不再只进报告：``_memory_hint()`` 每个慢系统周期都用**当前会话的部分聚合特征**
去历史里找相似情境，把结果交给行动层；行动层据此提前升级干预级别，
并把"因为哪条历史"写进 rationale —— 决策可追溯，不是玄学。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from .action import ActionPolicy
from .config import Config
from .contracts import Episode, InterventionRecord, StressEvent, StressSample
from .fast_system import EventDetector, FastEstimator
from .memory import (
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_MAX_Z_DISTANCE,
    FEATURE_KEYS,
    EpisodeStore,
    build_report,
    find_similar,
    feature_vector,
    summarize,
)
from .perception.base import build_sensor
from .perception.fusion import FusionEngine, safe_sigmoid
from .perception.synthetic import SyntheticStressDriver
from .slow_system import CapacityAssessor, StressPredictor

#: 干预后观察多久来评估其效果（秒）
INTERVENTION_WINDOW_S = 60.0

#: 部分情境检索时用的更严阈值：只有"真的像"的历史才允许影响当前决策
MEMORY_HINT_MAX_Z_DISTANCE = 1.5


def _percentile(values: list[float], q: float) -> float:
    """线性插值分位数（不引 numpy，保持零依赖）。"""
    if not values:
        return 0.0
    vals = sorted(values)
    n = len(vals)
    if n == 1:
        return float(vals[0])
    pos = (n - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    return float(vals[lo] + (vals[hi] - vals[lo]) * (pos - lo))


def _clamp(value: float, low: float, high: float) -> float:
    return float(min(high, max(low, value)))


@dataclass
class StepResult:
    t: float
    calibrated: bool
    stress: float | None
    state: str | None
    action: str | None
    events: list[StressEvent]

    def as_dict(self) -> dict:
        return {
            "t": self.t,
            "calibrated": self.calibrated,
            "stress": self.stress,
            "state": self.state,
            "action": self.action,
            "events": [e.as_dict() for e in self.events],
        }


class SessionPipeline:
    def __init__(
        self,
        config: Config | None = None,
        data_dir: str = "data",
        task_label: str = "default",
        driver: SyntheticStressDriver | None = None,
        sensors: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or Config()
        self.store = EpisodeStore(data_dir)
        self.profile = self.store.load_profile()

        # 记忆回写：用历史画像覆盖个体化参数（需求 M-5 / 场景 3）
        if self.profile.sessions > 0:
            self.config.theta_mid = self.profile.theta_mid
            self.config.theta_hi = self.profile.theta_hi
            self.config.tau_rev = self.profile.tau_rev

        self.driver = driver or SyntheticStressDriver()
        self.sensors: dict[str, Any] = (
            sensors
            if sensors is not None
            else {
                name: build_sensor(name, self.config, self.driver)
                for name in ("face", "speech", "text")
            }
        )

        self.fusion = FusionEngine(self.config)
        self.estimator = FastEstimator(self.config)
        self.detector = EventDetector(self.config)
        self.predictor = StressPredictor(self.config)
        self.assessor = CapacityAssessor(self.config)

        self.episode: Episode = self.store.new_episode(task_label)
        self.episode.task_label = task_label

        # ---- 记忆检索素材（会话开始时载入一次；索引缺字段会在此告警）----
        self.index_rows: list[dict] = self.store.load_index()
        self.history_prior: dict = self._build_history_prior()

        self.policy = ActionPolicy(self.config, self.profile, history_prior=self.history_prior)

        # 标定后的基线水平（normalized 尺度上的 z≈0 对应值）
        self.baseline_stress: float = safe_sigmoid(
            -self.config.z_target / self.config.z_tau
        )
        self.resilience: float = float(self.profile.resilience)

        self._stress_history: list[tuple[float, float]] = []
        self._last_record_t: float | None = None
        self._last_slow_t: float | None = None
        self._last_checkpoint_t: float | None = None
        self._pending_interventions: list[InterventionRecord] = []
        self._events: list[StressEvent] = []
        #: 最近一次影响决策的记忆命中（可追溯）
        self.last_memory_hint: dict | None = None

    # ------------------------------------------------------------------
    def backends(self) -> dict[str, str]:
        return {name: getattr(s, "backend", "unknown") for name, s in self.sensors.items()}

    # ------------------------------------------------------------------
    def _build_history_prior(self) -> dict:
        """把同类任务的历史压成一个"先验"，给行动层用（P1-#7）。"""
        rows = [
            r
            for r in self.index_rows
            if r.get("task_label") == self.episode.task_label and "max" in r
        ]
        if not rows:
            return {"sessions": 0, "task_label": self.episode.task_label}

        theta_hi = float(self.config.theta_hi)
        peaks = [float(r.get("max", 0.0) or 0.0) for r in rows]
        outcomes = [
            float(r["task_outcome"])
            for r in rows
            if r.get("task_outcome") is not None
        ]
        return {
            "sessions": len(rows),
            "task_label": self.episode.task_label,
            # 历史里有多大比例的同类会话曾冲进 PAUSE 区间
            "pause_rate": sum(1 for p in peaks if p >= theta_hi) / len(peaks),
            "mean_peak": sum(peaks) / len(peaks),
            "mean_outcome": (sum(outcomes) / len(outcomes)) if outcomes else None,
        }

    # ------------------------------------------------------------------
    def _memory_hint(self, t: float) -> dict | None:
        """用当前会话的**部分**聚合特征检索历史，返回最相似的一条（无则 None）。"""
        if not self.index_rows or not self.episode.samples:
            return None
        partial = summarize(self.episode)
        hits = find_similar(
            self.index_rows,
            feature_vector(partial),
            task_label=self.episode.task_label,
            top_k=1,
            exclude_id=self.episode.episode_id,
            max_z_distance=min(
                MEMORY_HINT_MAX_Z_DISTANCE,
                float(getattr(self.config, "memory_max_z_distance", DEFAULT_MAX_Z_DISTANCE)),
            ),
            half_life_days=float(
                getattr(self.config, "memory_half_life_days", DEFAULT_HALF_LIFE_DAYS)
            ),
        )
        return hits[0] if hits else None

    # ------------------------------------------------------------------
    def step(self, payload: dict[str, Any]) -> StepResult:
        cfg = self.config
        t = float(payload.get("t", 0.0))

        observations = tuple(sensor.sense(payload) for sensor in self.sensors.values())

        if not self.fusion.calibrated:
            self.fusion.observe_baseline(observations)
            if t >= cfg.baseline_seconds:
                self.fusion.finalize_baseline()

        frame = self.fusion.fuse(observations, t)
        sample = self.estimator.update(frame)
        events = self.detector.update(sample)
        self._events.extend(events)

        if not self.fusion.calibrated:
            return StepResult(t=t, calibrated=False, stress=None, state=None, action=None,
                              events=events)

        # ---- 标定后：进入记忆与慢系统 ----
        self._stress_history.append((sample.timestamp, sample.smoothed))

        if self._last_record_t is None or (t - self._last_record_t) >= cfg.memory_interval:
            self._record_sample(sample, t)
            self._last_record_t = t

        # ---- 会话检查点：崩溃最多丢一个 checkpoint_interval（P0-#4）----
        if (
            self.episode.samples
            and (
                self._last_checkpoint_t is None
                or (t - self._last_checkpoint_t) >= cfg.checkpoint_interval
            )
        ):
            self.store.save_checkpoint(self.episode)
            self._last_checkpoint_t = t

        decision = None
        state = None
        if self._last_slow_t is None or (t - self._last_slow_t) >= cfg.slow_interval:
            state, decision = self._run_slow(sample, t)
            self._last_slow_t = t

        return StepResult(
            t=t,
            calibrated=True,
            stress=sample.smoothed,
            state=state,
            action=(decision.action if decision else None),
            events=events,
        )

    # ------------------------------------------------------------------
    def _record_sample(self, sample: StressSample, t: float) -> None:
        self.episode.samples.append(
            {
                "t": t,
                "raw": round(sample.raw, 5),
                "smoothed": round(sample.smoothed, 5),
                "confidence": round(sample.confidence, 4),
                "contributions": {k: round(v, 4) for k, v in sample.contributions.items()},
            }
        )

    def _run_slow(self, sample: StressSample, t: float):
        times = [x for x, _ in self._stress_history]
        values = [v for _, v in self._stress_history]

        forecast = self.predictor.forecast(times, values, self.baseline_stress)
        assessment = self.assessor.assess(
            times, values, forecast, self.resilience, contributions=sample.contributions
        )

        hint = self._memory_hint(t)
        self.last_memory_hint = hint

        self.episode.capacity_timeline.append(
            {
                "t": t,
                "state": assessment.state,
                "score": round(assessment.score, 4),
                "forecast": round(forecast.point, 4),
                "slope": round(forecast.slope, 4),
                "memory_hint": (hint or {}).get("episode_id"),
            }
        )

        decision = self.policy.decide(sample, assessment, now=t, memory_hint=hint)

        if decision.action != "continue" and decision.intervention:
            record = InterventionRecord(
                timestamp=t,
                action=decision.action,
                intervention=decision.intervention,
                stress_before=sample.smoothed,
            )
            self._pending_interventions.append(record)
            self.episode.interventions.append(record.as_dict())

        return assessment.state, decision

    # ------------------------------------------------------------------
    def _settle_interventions(self) -> None:
        """用干预后 60s 的紧张度更新每条干预的效果（需求 A-5）。"""
        samples = self.episode.samples
        if not samples:
            return
        for rec_dict in self.episode.interventions:
            ts = float(rec_dict["timestamp"])
            window = [
                float(s["smoothed"])
                for s in samples
                if ts < float(s["t"]) <= ts + INTERVENTION_WINDOW_S
            ]
            rec_dict["stress_after"] = round(sum(window) / len(window), 5) if window else None

    # ------------------------------------------------------------------
    def finalize(self, task_outcome: float | None = None) -> dict:
        cfg = self.config
        self.episode.task_outcome = task_outcome
        self.episode.baseline_mu = self.baseline_stress
        fused_stats = self.fusion.fused_stats
        self.episode.baseline_sigma = fused_stats[1] if fused_stats else 0.0
        # 原始域的一对 (μ, σ)：这才是"这个人静息时什么样"，跨会话可比较
        self.episode.baseline_raw_mu = fused_stats[0] if fused_stats else 0.0
        self.episode.baseline_raw_sigma = fused_stats[1] if fused_stats else 0.0
        self.episode.backends = self.backends()
        self.episode.events = [e.as_dict() for e in self._events]

        self._settle_interventions()
        self.episode.summary = summarize(self.episode)

        # ---- 跨会话检索（必须在 append 之前，避免把自己检索进来）----
        index_rows = self.store.load_index()
        cross: list[dict] = []
        if index_rows:
            cross = find_similar(
                index_rows,
                feature_vector(self.episode.summary),
                task_label=self.episode.task_label,
                top_k=3,
                exclude_id=self.episode.episode_id,
                max_z_distance=float(
                    getattr(cfg, "memory_max_z_distance", DEFAULT_MAX_Z_DISTANCE)
                ),
                half_life_days=float(
                    getattr(cfg, "memory_half_life_days", DEFAULT_HALF_LIFE_DAYS)
                ),
            )

        # ---- 落盘 ----
        self.store.append_episode(self.episode)

        report = build_report(self.episode, cross_session=cross, backends=self.backends())
        report_path = self.store.write_report(self.episode.episode_id, report.to_markdown())

        # ---- 个性化回写（需求 M-5）----
        self._update_profile()
        self.store.save_profile(self.profile)

        return {
            "episode": self.episode,
            "report": report,
            "report_path": report_path,
            "outputs": sorted(p.name for p in self.store.data_dir.iterdir()),
        }

    # ------------------------------------------------------------------
    def _update_profile(self) -> None:
        """把这次会话学到的东西回写给下一次会话（需求 M-5）。

        评审 P0-#1/#2/#3/#5 的修复点：
          * 空会话不计数、不更新任何统计（#5）；
          * baseline_mu / baseline_sigma 都用**样本量门控 EMA**，不再直接覆盖（#3）；
          * baseline_sigma 真正被写回（#2）；
          * theta_mid / theta_hi / tau_rev 真正被学习，并把依据写进 threshold_evidence（#1）。
        """
        cfg = self.config
        samples = self.episode.samples or []
        if not samples:
            return  # 空会话：不计数、不污染画像

        prev_sessions = int(self.profile.sessions)
        self.profile.sessions = prev_sessions + 1
        # α = 1/min(sessions+1, cap)：第一次会话直接采用，之后逐步变"迟钝"
        alpha = 1.0 / min(prev_sessions + 1, max(int(cfg.ema_cap), 1))

        new_mu = float(
            self.episode.baseline_raw_mu or 0.0
        ) or float(self.baseline_stress)
        new_sigma = float(
            self.episode.baseline_raw_sigma or 0.0
        ) or float(self.episode.baseline_sigma or 0.0)
        prev_mu = float(self.profile.baseline_mu)
        prev_sigma = float(self.profile.baseline_sigma)
        if prev_sessions == 0:
            self.profile.baseline_mu = new_mu
            if new_sigma > 0:
                self.profile.baseline_sigma = new_sigma
        else:
            self.profile.baseline_mu = (1.0 - alpha) * prev_mu + alpha * new_mu
            if new_sigma > 0:
                self.profile.baseline_sigma = (1.0 - alpha) * prev_sigma + alpha * new_sigma

        # ---- 恢复力（原本就是 EMA，保留）----
        recovery = (self.episode.summary or {}).get("recovery_time")
        current_resilience = (
            max(0.0, min(1.0, 1.0 - float(recovery) / 300.0))
            if recovery is not None
            else 0.5
        )
        self.profile.resilience = 0.7 * self.profile.resilience + 0.3 * current_resilience

        # ---- 阈值自适应（P0-#1）----
        self._learn_thresholds()

        # 把基线的更新过程也记成证据：下次有人问"这个 0.11 哪来的"能答上
        evidence = dict(self.profile.threshold_evidence or {})
        evidence["baseline"] = {
            "value": {
                "mu": round(float(self.profile.baseline_mu), 5),
                "sigma": round(float(self.profile.baseline_sigma), 5),
            },
            "rule": "EMA，α = 1/min(已有会话数+1, ema_cap)；μ 与 σ 同为静息期融合原始域",
            "n": prev_sessions,
            "alpha": round(alpha, 4),
            "this_session": {"mu": round(new_mu, 5), "sigma": round(new_sigma, 5)},
        }
        self.profile.threshold_evidence = evidence

        # ---- 干预有效性：ΔS 越负越有效 ----
        for rec in self.episode.interventions:
            after = rec.get("stress_after")
            if after is None:
                continue
            delta = after - float(rec.get("stress_before", 0.0))
            effect = -delta  # 下降为正收益
            iv_id = rec.get("intervention", "")
            prev = self.profile.intervention_effectiveness.get(iv_id, 0.0)
            self.profile.intervention_effectiveness[iv_id] = 0.7 * prev + 0.3 * effect

    # ------------------------------------------------------------------
    def _learn_thresholds(self) -> None:
        """从同类任务的历史会话里学阈值，并把"依据"写进画像。

        规则（可解释，允许人工推翻）：
          * ``theta_mid`` = 最近 N 次同任务会话 ``mean`` 的 p60；
          * ``theta_hi``  = ``theta_mid + 0.20``（保持迟滞带宽）；
          * ``tau_rev``   = 最近 N 次同任务会话恢复时间中位数 × 1.2。
        样本不足 2 次时保持原值并如实记录 "insufficient_data"。
        """
        cfg = self.config
        window = max(int(cfg.profile_window), 2)
        rows = [
            r
            for r in self.store.load_index()
            if r.get("task_label") == self.episode.task_label and "mean" in r
        ][-window:]

        means = [float(r.get("mean", 0.0) or 0.0) for r in rows]
        recoveries = [
            float(r["recovery_time"])
            for r in rows
            if r.get("recovery_time") is not None
        ]

        evidence: dict = {
            "task_label": self.episode.task_label,
            "window_n": len(rows),
        }

        if len(means) >= 2:
            theta_mid = _clamp(_percentile(means, 60.0), 0.35, 0.85)
            theta_hi = _clamp(theta_mid + 0.20, theta_mid + 0.05, 0.98)
            self.profile.theta_mid = theta_mid
            self.profile.theta_hi = theta_hi
            evidence["theta_mid"] = {
                "value": round(theta_mid, 4),
                "rule": "p60(最近 N 次同任务会话的 mean)",
                "n": len(means),
                "inputs": [round(m, 4) for m in means],
            }
            evidence["theta_hi"] = {
                "value": round(theta_hi, 4),
                "rule": "theta_mid + 0.20（迟滞带宽）",
                "theta_mid": round(theta_mid, 4),
            }
        else:
            evidence["theta_mid"] = {
                "value": round(float(self.profile.theta_mid), 4),
                "rule": "insufficient_data（同任务会话 < 2，沿用旧值）",
                "n": len(means),
            }

        if len(recoveries) >= 2:
            median_recovery = _percentile(recoveries, 50.0)
            tau_rev = _clamp(median_recovery * 1.2, 30.0, 600.0)
            self.profile.tau_rev = tau_rev
            evidence["tau_rev"] = {
                "value": round(tau_rev, 2),
                "rule": "median(recovery_time) × 1.2",
                "n": len(recoveries),
                "inputs": [round(r, 1) for r in recoveries],
            }
        else:
            evidence["tau_rev"] = {
                "value": round(float(self.profile.tau_rev), 2),
                "rule": "insufficient_data（有效恢复时间 < 2，沿用旧值）",
                "n": len(recoveries),
            }

        evidence["learned_at"] = datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        )
        self.profile.threshold_evidence = evidence

    # ------------------------------------------------------------------
    def run_offline(
        self,
        duration: float = 540.0,
        dt: float | None = None,
        task_outcome: float | None = None,
        on_step: Any = None,
    ) -> dict:
        """离线跑完整会话（使用合成/注入的数据源）。"""
        step_dt = float(dt if dt is not None else 1.0 / max(self.config.fast_hz, 1e-6))
        t = 0.0
        while t <= duration + 1e-9:
            result = self.step({"t": t})
            if on_step is not None:
                on_step(result)
            t += step_dt
        return self.finalize(task_outcome)

    def run_stream(
        self, payloads: Iterable[dict[str, Any]], task_outcome: float | None = None
    ) -> dict:
        """跑一个已被外部提供的 payload 流（真实摄像头/麦克风场景）。"""
        for payload in payloads:
            self.step(payload)
        return self.finalize(task_outcome)
