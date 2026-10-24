"""全局配置与个体画像的持久化。

所有阈值均为**启发式初值**（需求文档 §8 / 架构文档 §8），需要在真实使用中
按个体校准；不要把它们当成心理学上的通用常数。

持久化可靠性（评审 P0-#4）：
  * 画像一律 **原子写**（同目录临时文件 + ``os.replace``），断电不会留下半个 JSON；
  * 读到损坏画像时**不静默吞掉**：先把坏文件备份成 ``profile.json.corrupt-<ts>``，
    再发一条 warning 日志，然后才回落到默认画像。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .contracts import PersonalProfile

LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    # ---- 时序 ----
    baseline_seconds: float = 30.0  # 静息基线期
    fast_hz: float = 1.0  # 合成/离线模式下的快系统采样率
    slow_interval: float = 5.0  # 慢系统评估周期（秒）
    memory_interval: float = 1.0  # 记忆缓冲写入间隔（秒）
    #: 会话检查点间隔：每这么久把一次 running 草稿追加进 episodes.jsonl（P0-#4）
    checkpoint_interval: float = 60.0

    # ---- 快系统 ----
    tau_fast: float = 1.5  # EWMA 时间常数（秒）
    theta_spike: float = 0.15  # 尖峰幅度阈值
    spike_sigma_k: float = 3.0  # 尖峰显著性倍数
    t_hi: float = 20.0  # "高位持续"判据时长（秒）

    # ---- 慢系统 ----
    trend_window: float = 60.0  # 趋势估计窗口（秒）
    forecast_horizon: float = 60.0  # 预测视界（秒）
    tau_rev: float = 90.0  # 均值回复时间常数初值（秒）
    theta_mid: float = 0.55  # WATCH 阈值
    theta_hi: float = 0.75  # PAUSE 阈值
    theta_hysteresis: float = 0.05  # 上下阈值分离量，抑制抖动
    t_watch: float = 60.0  # WATCH 持续多久升级为 micro_break

    # ---- 行动层冷却 ----
    cooldown_coach: float = 60.0
    cooldown_micro_break: float = 120.0
    cooldown_pause: float = 90.0

    # ---- 融合权重（可被个体画像覆盖）----
    w_face: float = 0.40
    w_speech: float = 0.35
    w_text: float = 0.25

    # ---- 归一化 ----
    z_target: float = 2.5  # 认为"已明显紧张"的 z 值（静息期 z≈0）
    z_tau: float = 1.2  # sigmoid 的软度；越大越迟钝
    sigma_prior: float = 0.12  # σ 收缩先验：静息期 σ 天然极小，需与之混合
    sigma_floor: float = 0.02  # σ 绝对下界，避免除零
    audio_sr: int = 16000  # 语音采样率

    # ---- 检索 ----
    #: z 距离阈值：超过它就不算"相似情境"（P1-#8：不再无条件返回 top-k）
    memory_max_z_distance: float = 2.0
    #: 时间衰减半衰期（天）：两年前的会话不会和昨天等权（P2-#13）
    memory_half_life_days: float = 180.0

    # ---- 个性化学习 ----
    #: 阈值自适应取最近多少次同任务会话
    profile_window: int = 10
    #: EMA 的样本量上限（α = 1 / min(sessions+1, ema_cap)）
    ema_cap: int = 10

    # ---- 路径 ----
    data_dir: str = "data"

    # ------------------------------------------------------------------
    def modality_weights(self) -> dict[str, float]:
        return {"face": self.w_face, "speech": self.w_speech, "text": self.w_text}

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        write_text_atomic(Path(path), json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text("utf-8"))
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


# --------------------------------------------------------------------------
# 原子写入
# --------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_text_atomic(path: Path, text: str) -> None:
    """原子写：同目录临时文件 + fsync + ``os.replace``。

    同目录是必须的——``os.replace`` 跨卷不保证原子。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------
# 个体画像
# --------------------------------------------------------------------------


def load_profile(path: str | Path) -> PersonalProfile:
    """加载个体画像；不存在则返回默认画像。

    损坏时不静默返回默认值：备份坏文件 + 记 warning，让"历史画像消失"变成
    可发现的事件（评审 P0-#4）。
    """
    p = Path(path)
    if not p.exists():
        return PersonalProfile()
    try:
        return PersonalProfile.from_dict(json.loads(p.read_text("utf-8")))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        backup = p.with_name(f"{p.name}.corrupt-{datetime.now().strftime('%Y%m%dT%H%M%S')}")
        try:
            os.replace(p, backup)
        except OSError:
            backup = None  # type: ignore[assignment]
        LOGGER.warning(
            "个体画像损坏（%s）：%s；已备份到 %s，本次回落到默认画像。",
            type(exc).__name__,
            exc,
            backup,
        )
        return PersonalProfile()


def save_profile(path: str | Path, profile: PersonalProfile) -> None:
    profile.updated_at = _now_iso()
    write_text_atomic(
        Path(path), json.dumps(profile.as_dict(), indent=2, ensure_ascii=False)
    )
