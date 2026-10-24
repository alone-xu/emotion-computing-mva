"""④ 记忆层 · 相似情境检索。

实现需求 M-3。

MVA 阶段刻意只用**统计相似度**（加权标准化欧氏距离 + 任务标签约束），
不引入向量数据库或嵌入模型 —— 会话数量在这个阶段很小，
可解释性与零依赖比检索精度更重要。

三处关键修正（评审 P1-#8）：
  1. **绝对距离**：标准化用**固定物理尺度**（``FEATURE_SCALE``），而不是 top-k 内的
     最大距离。这样同一对会话的 distance 在 top_k=3 / top_k=5 / 不同查询下都一样，
     跨查询可比；similarity = exp(-d²/2) 也是绝对量。
  2. **最低阈值**：只有 ``distance <= max_z_distance`` 才算"相似情境"；历史里全是
     不相干的会话时返回空列表，而不是硬凑 3 条。
  3. **特征加权 + 时间衰减**：紧张度/斜率这类特征权重高于时长；越老的会话排名越靠后
     （半衰期 ``memory_half_life_days``）。
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Sequence

from ..contracts import Episode
from .store import FEATURE_KEYS, FEATURE_SCALE, FEATURE_WEIGHTS, summarize

__all__ = [
    "FEATURE_KEYS",
    "FEATURE_SCALE",
    "FEATURE_WEIGHTS",
    "DEFAULT_MAX_Z_DISTANCE",
    "DEFAULT_HALF_LIFE_DAYS",
    "episode_feature_vector",
    "find_similar",
    "feature_vector",
]

#: 默认"算相似"的加权 z 距离上限
DEFAULT_MAX_Z_DISTANCE: float = 2.0

#: 默认时间衰减半衰期（天）
DEFAULT_HALF_LIFE_DAYS: float = 180.0


def episode_feature_vector(episode: Episode) -> list[float]:
    summary = episode.summary or summarize(episode)
    return feature_vector(summary)


def feature_vector(summary: dict) -> list[float]:
    """从聚合统计里取出特征向量（顺序与 ``FEATURE_KEYS`` 一致）。"""
    return [float(summary.get(k, 0.0) or 0.0) for k in FEATURE_KEYS]


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _age_days(started_at: str | None, now: datetime) -> float | None:
    dt = _parse_iso(started_at)
    if dt is None:
        return None
    return max(0.0, (now - dt).total_seconds() / 86400.0)


def _decay(age_days: float | None, half_life_days: float) -> float:
    if age_days is None or half_life_days <= 0:
        return 1.0
    return float(0.5 ** (age_days / half_life_days))


def weighted_distance(
    query: Sequence[float],
    row: Sequence[float],
    weights: Sequence[float] | None = None,
    scales: Sequence[float] | None = None,
) -> float:
    """加权标准化欧氏距离（每维除以固定尺度，量纲无关、跨查询可比）。"""
    w = list(weights) if weights is not None else [FEATURE_WEIGHTS[k] for k in FEATURE_KEYS]
    s = list(scales) if scales is not None else [FEATURE_SCALE[k] for k in FEATURE_KEYS]
    total_w = sum(w) or 1.0
    acc = 0.0
    for q, r, wi, si in zip(query, row, w, s):
        si = si if si > 1e-9 else 1.0
        delta = (float(q) - float(r)) / si
        acc += wi * delta * delta
    return math.sqrt(acc / total_w)


def find_similar(
    index_rows: Sequence[dict],
    query_features: Sequence[float],
    task_label: str | None = None,
    top_k: int = 3,
    exclude_id: str | None = None,
    max_z_distance: float = DEFAULT_MAX_Z_DISTANCE,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    now: datetime | None = None,
) -> list[dict]:
    """在历史索引中找最相似的情境。

    返回按（相似度 × 时间衰减）降序排列的记录，每条附加：

    * ``distance``：加权标准化欧氏距离（**绝对量**，与 top_k 无关）
    * ``similarity``：``exp(-distance²/2)`` ∈ (0, 1]，跨查询可比
    * ``age_days`` / ``decay``：会话年龄与时间衰减因子
    * ``score``：用于排序的最终分（similarity × decay）

    ``distance > max_z_distance`` 的历史会话**不会**返回；没有合格历史时返回 ``[]``，
    由调用方如实说明"没有相似情境"，而不是硬凑数。
    """
    query = [float(v) for v in query_features]
    if len(query) != len(FEATURE_KEYS):
        raise ValueError(
            f"查询特征维度 {len(query)} 与 FEATURE_KEYS {len(FEATURE_KEYS)} 不一致"
        )

    now_dt = now or datetime.now(timezone.utc)
    scored: list[dict] = []
    for row in index_rows:
        if task_label is not None and row.get("task_label") != task_label:
            continue
        if exclude_id is not None and row.get("episode_id") == exclude_id:
            continue
        if not all(k in row for k in FEATURE_KEYS):
            continue  # 缺字段的旧行由 store.load_index() 计数告警，这里静默跳过

        values = [float(row.get(k, 0.0) or 0.0) for k in FEATURE_KEYS]
        distance = weighted_distance(query, values)
        if distance > max_z_distance:
            continue  # 硬阈值：不相干的历史不许标成"相似"

        age = _age_days(row.get("started_at"), now_dt)
        decay = _decay(age, half_life_days)
        similarity = math.exp(-0.5 * distance * distance)
        out = dict(row)
        out["distance"] = float(distance)
        out["similarity"] = float(similarity)
        out["age_days"] = None if age is None else round(age, 2)
        out["decay"] = float(decay)
        out["score"] = float(similarity * decay)
        scored.append(out)

    scored.sort(key=lambda r: (-r["score"], r.get("episode_id", "")))
    return scored[: max(int(top_k), 1)]
