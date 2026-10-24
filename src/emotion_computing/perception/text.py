"""① 感知层 · 语义文本。

后端优先级：
  1. ``TransformerTextSensor`` —— transformers 情感模型。**必须显式开启**
     （环境变量 ``EC_ENABLE_TRANSFORMERS=1``），因为首次运行需要下载权重，
     不应在隐式路径上阻塞。默认模型可用 ``EC_TEXT_MODEL`` 覆盖。
  2. ``LexiconTextSensor`` —— 内置中英焦虑词典 + 否定/程度规则，零依赖、永远可用。

诚实性说明 [V]：
    词典权重是人工设定的启发式，未在标注语料上校准。输出是「文本中的
    焦虑表达强度」的代理量，不是经过验证的情感分类器输出。
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np

from ..contracts import ModalityObservation
from .base import ModalitySensor, clip01

#: 焦虑/紧张词 → 强度（正数表示加剧，负数表示缓解）
ANXIETY_LEXICON: dict[str, float] = {
    # 中文
    "紧张": 0.6, "焦虑": 0.7, "慌": 0.8, "害怕": 0.7, "担心": 0.5,
    "压力": 0.6, "来不及": 0.8, "做不完": 0.8, "不会": 0.5, "错了": 0.6,
    "糟糕": 0.7, "完了": 0.9, "崩溃": 0.9, "喘不过气": 0.9, "手抖": 0.8,
    "心跳": 0.5, "不知道": 0.4, "怎么办": 0.7, "好难": 0.5, "太难": 0.6,
    "不行": 0.7, "撑不住": 0.9, "累": 0.4, "烦": 0.6, "急": 0.7,
    "快点": 0.5, "来不及了": 0.9, "空白": 0.6, "忘记": 0.5,
    # 中文（缓解）
    "冷静": -0.3, "没事": -0.4, "放松": -0.4, "还好": -0.3, "稳": -0.3,
    "可以": -0.2, "搞定": -0.3, "顺利": -0.4,
    # 英文
    "anxious": 0.7, "nervous": 0.7, "scared": 0.7, "worried": 0.5,
    "stressed": 0.7, "panic": 0.9, "overwhelmed": 0.9, "tense": 0.6,
    "fail": 0.7, "deadline": 0.5, "hurry": 0.6, "can't": 0.5,
    "calm": -0.4, "fine": -0.3, "okay": -0.2,
}

#: 否定词：紧邻焦虑词前缀时削弱/反转其强度。
#: `[F]` 不能退回"子串包含"写法——「非常紧张」里的「非」会命中「非」，
#: 把高分直接翻转成 0（本项目实测踩过）。一律用「前缀末尾是否以否定词结尾」。
NEGATIONS: tuple[str, ...] = (
    "没有", "不是", "不太", "不要", "不能", "不会",
    "不", "没", "别", "无", "非", "未",
    "not", "no", "never",
)

#: 程度副词 → 放大倍数
INTENSIFIERS: dict[str, float] = {
    "很": 1.5, "非常": 1.8, "特别": 1.7, "太": 1.6, "超": 1.7,
    "极其": 2.0, "有点": 1.2, "稍微": 1.1, "好": 1.4, "真是": 1.5,
    "very": 1.6, "so": 1.4, "really": 1.5, "extremely": 2.0,
}


def lexicon_anxiety_score(text: str) -> tuple[float, dict[str, float]]:
    """词典法：返回 (紧张度 ∈ [0,1], 命中词贡献表)。"""
    if not text or not text.strip():
        return 0.0, {}

    total = 0.0
    hits: dict[str, float] = {}
    for word, weight in ANXIETY_LEXICON.items():
        for match in re.finditer(re.escape(word), text):
            contribution = float(weight)
            # 只看紧邻前缀，并去掉尾部空白（英文场景下否定词后常有空格）
            prefix = text[max(0, match.start() - 5) : match.start()].rstrip()

            # 1) 程度副词放大
            stripped = prefix
            for adv, multiplier in INTENSIFIERS.items():
                if prefix.endswith(adv):
                    contribution *= multiplier
                    stripped = prefix[: -len(adv)]
                    break

            # 2) 否定削弱/反号：剥离程度副词**之后**再看末尾是否是否定词，
            #    否则「非常」会被「非」误判成否定。
            if any(stripped.endswith(neg) for neg in NEGATIONS):
                contribution *= -0.5

            total += contribution
            hits[word] = hits.get(word, 0.0) + contribution

    positive = max(total, 0.0)
    # 单调饱和：命中越强越高，但永远不超过 1
    score = clip01(1.0 - float(np.exp(-positive / 1.5)))
    return score, hits


class LexiconTextSensor(ModalitySensor):
    """零依赖的词典法语义传感器（降级后端，永远可用）。"""

    name = "text"

    def __init__(self, config: Any, driver: Any = None) -> None:
        super().__init__()
        from .synthetic import SyntheticStressDriver

        self.driver = driver if isinstance(driver, SyntheticStressDriver) else None
        self.config = config
        self.backend = "lexicon"

    def sense(self, payload: dict[str, Any]) -> ModalityObservation:
        t = float(payload.get("t", 0.0))
        text = payload.get("text")
        if text is None:
            return self._synthetic_fallback(t, "no_text")
        text = str(text)
        if not text.strip():
            return self._missing(t, "empty_text")

        score, hits = lexicon_anxiety_score(text)
        return ModalityObservation(
            modality=self.name,
            timestamp=t,
            score=score,
            confidence=0.6,
            available=True,
            backend="lexicon",
            features={k: float(v) for k, v in list(hits.items())[:12]},
            note="词典法代理量，权重未经语料校准 [V]",
        )


class TransformerTextSensor(ModalitySensor):
    """transformers 情感模型后端（需显式开启）。

    开启方式::

        set EC_ENABLE_TRANSFORMERS=1
        set EC_TEXT_MODEL=uer/roberta-base-finetuned-jd-binary-chinese

    未开启时构造即失败，由 ``build_sensor`` 自动降级到词典法 ——
    这是刻意的设计：隐式下载几亿参数权重会毁掉可预测性。
    """

    name = "text"

    def __init__(self, config: Any, driver: Any = None) -> None:
        super().__init__()
        from .synthetic import SyntheticStressDriver

        self.driver = driver if isinstance(driver, SyntheticStressDriver) else None
        self.config = config

        if os.environ.get("EC_ENABLE_TRANSFORMERS", "") != "1":
            raise RuntimeError(
                "未启用 transformers 后端（需设 EC_ENABLE_TRANSFORMERS=1）"
            )

        from transformers import pipeline

        model = os.environ.get(
            "EC_TEXT_MODEL", "uer/roberta-base-finetuned-jd-binary-chinese"
        )
        self._pipe = pipeline("sentiment-analysis", model=model)
        self._negative_labels = {
            "negative", "neg", "label_0", "negative(消极)", "消极",
        }
        self.backend = "transformers"

    def sense(self, payload: dict[str, Any]) -> ModalityObservation:
        t = float(payload.get("t", 0.0))
        text = payload.get("text")
        if text is None or not str(text).strip():
            return self._missing(t, "no_text")

        try:
            out = self._pipe(str(text)[:512])[0]
            label = str(out.get("label", "")).lower()
            prob = float(out.get("score", 0.0))
            is_negative = label in self._negative_labels or label.endswith("_0")
            negative_prob = prob if is_negative else 1.0 - prob
            return ModalityObservation(
                modality=self.name,
                timestamp=t,
                score=clip01(negative_prob),
                confidence=0.75,
                available=True,
                backend="transformers",
                features={"negative_prob": negative_prob, "raw_prob": prob},
                note=f"模型输出：{out.get('label')}（负向概率作为紧张度代理）",
            )
        except Exception as exc:  # noqa: BLE001
            return self._synthetic_fallback(t, f"transformers_error:{exc!r}")
