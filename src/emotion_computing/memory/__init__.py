"""④ 记忆层：沉淀、检索、复盘、个性化回写。

实现需求 M-1 ~ M-6。

隐私红线（M-6）：本层只存储**特征、统计量与结论**，
不存储原始视频帧、音频波形或完整原始文本。
``assert_privacy_safe()`` 在每次落盘前强制执行这条红线，
``PrivacyViolation`` 是唯一允许的失败方式（拒绝写入，而不是"先存后删"）。
"""

from __future__ import annotations

from .index_db import SqliteIndex
from .retrieval import (
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_MAX_Z_DISTANCE,
    FEATURE_KEYS,
    FEATURE_SCALE,
    FEATURE_WEIGHTS,
    episode_feature_vector,
    feature_vector,
    find_similar,
)
from .review import build_report
from .store import (
    PRIVACY_ALLOWED_SAMPLE_KEYS,
    PRIVACY_MAX_NOTE_CHARS,
    EpisodeStore,
    PrivacyViolation,
    assert_privacy_safe,
    summarize,
)

__all__ = [
    "EpisodeStore",
    "SqliteIndex",
    "summarize",
    "build_report",
    "episode_feature_vector",
    "feature_vector",
    "find_similar",
    "FEATURE_KEYS",
    "FEATURE_WEIGHTS",
    "FEATURE_SCALE",
    "DEFAULT_MAX_Z_DISTANCE",
    "DEFAULT_HALF_LIFE_DAYS",
    "assert_privacy_safe",
    "PrivacyViolation",
    "PRIVACY_ALLOWED_SAMPLE_KEYS",
    "PRIVACY_MAX_NOTE_CHARS",
]
