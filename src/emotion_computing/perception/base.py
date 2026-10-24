"""感知层基础设施：模态传感器接口 + 降级装配。

降级规范见 docs/02-architecture.md §7：
    首选后端 → 降级后端 → 合成兜底。
    每次降级只告警一次，且必须体现在 ModalityObservation.backend 上。
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable

from ..contracts import ModalityObservation

# --------------------------------------------------------------------------
# 只告警一次
# --------------------------------------------------------------------------

_WARNED: set[str] = set()


def warn_once(message: str) -> None:
    """同一条告警在整个进程生命周期内只输出一次。"""
    if message in _WARNED:
        return
    _WARNED.add(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def reset_warnings() -> None:
    """测试用：清空告警去重表。"""
    _WARNED.clear()


# --------------------------------------------------------------------------
# 模态传感器
# --------------------------------------------------------------------------


class ModalitySensor(ABC):
    """所有模态传感器的基类。

    ``sense()`` 接收一个 payload 字典，必须至少包含 ``t``（时间戳，秒）。
    真实后端会额外读取 ``frame`` / ``audio`` / ``text`` 等字段；
    若缺失则退化为该后端的"无输入"行为（返回 available=False 或合成值）。
    """

    name: str = "unknown"

    def __init__(self) -> None:
        self.backend: str = "uninitialized"
        self._fallback_reason: str = ""

    # -- 子类实现 ---------------------------------------------------------
    @abstractmethod
    def sense(self, payload: dict[str, Any]) -> ModalityObservation:
        raise NotImplementedError

    # -- 通用 -------------------------------------------------------------
    def is_available(self) -> bool:
        return self.backend != "unavailable"

    def describe(self) -> dict[str, str]:
        return {"modality": self.name, "backend": self.backend, "note": self._fallback_reason}

    def _missing(self, t: float, reason: str) -> ModalityObservation:
        """当本帧确实没有可用数据时返回的观测。"""
        return ModalityObservation(
            modality=self.name,
            timestamp=t,
            score=0.0,
            confidence=0.0,
            available=False,
            backend=self.backend,
            note=reason,
        )

    def _synthetic_fallback(self, t: float, reason: str) -> ModalityObservation:
        """真实输入缺失时的退化出口。

        若持有合成驱动，则用合成值填充并把 backend 标记为 ``<backend>+synthetic``，
        保证「系统到底跑了什么」在输出里始终可见（架构文档 §7）。
        """
        driver = getattr(self, "driver", None)
        if driver is None:
            return self._missing(t, reason)
        if hasattr(driver, "is_missing") and driver.is_missing(self.name, t):
            return self._missing(t, "synthetic_dropout")
        return ModalityObservation(
            modality=self.name,
            timestamp=t,
            score=float(driver.modality_score(self.name, t)),
            confidence=0.3,
            available=True,
            backend=f"{self.backend}+synthetic",
            features={"truth": float(driver.truth(t))},
            note=f"真实输入不可用（{reason}），退回合成值",
        )


# --------------------------------------------------------------------------
# 降级装配
# --------------------------------------------------------------------------

BackendFactory = Callable[[Any, Any], ModalitySensor]


def _candidates(name: str) -> list[tuple[str, str]]:
    """返回 (模块路径, 类名) 的有序候选列表。延迟 import，避免循环依赖。"""
    table = {
        "face": [
            ("emotion_computing.perception.face", "MediaPipeFaceSensor"),
            ("emotion_computing.perception.face", "YuNetFaceSensor"),
            ("emotion_computing.perception.face", "HaarFaceSensor"),
        ],
        "speech": [
            ("emotion_computing.perception.speech", "LibrosaSpeechSensor"),
        ],
        "text": [
            ("emotion_computing.perception.text", "TransformerTextSensor"),
            ("emotion_computing.perception.text", "LexiconTextSensor"),
        ],
    }
    return table.get(name, [])


def build_sensor(name: str, config: Any, driver: Any = None) -> ModalitySensor:
    """按"首选 → 降级 → 合成"顺序装配一个模态传感器。永不抛异常。"""
    import importlib

    for module_path, class_name in _candidates(name):
        try:
            module = importlib.import_module(module_path)
            factory: BackendFactory = getattr(module, class_name)
            sensor = factory(config, driver)
            if sensor.is_available():
                return sensor
            warn_once(
                f"[{name}] 后端 {class_name} 构造成功但不可用，继续降级："
                f"{sensor._fallback_reason}"
            )
        except Exception as exc:  # noqa: BLE001 —— 降级链必须吞掉一切异常
            warn_once(f"[{name}] 后端 {class_name} 不可用，降级：{exc!r}")

    # 兜底：合成信号源，永远不会失败
    from .synthetic import SyntheticSensor

    warn_once(f"[{name}] 所有真实后端不可用，回退到合成信号源（synthetic）。")
    return SyntheticSensor(name, driver)


# --------------------------------------------------------------------------
# 特征缩放工具（感知层共享）
# --------------------------------------------------------------------------
#
# 说明：下面这些 ramp 阈值都是**启发式初值**（需求文档 §8），
# 用在一个"把原始特征压到 [0,1] 的紧张度代理量"的粗略映射上。
# 它们没有经过标注数据校准，属于 [V] 待验证内容。


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def ramp_up(x: float, lo: float, hi: float) -> float:
    """x ≤ lo → 0；x ≥ hi → 1；中间线性。hi 应大于 lo。"""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return clip01((x - lo) / (hi - lo))


def ramp_down(x: float, hi: float, lo: float) -> float:
    """x ≥ hi → 0；x ≤ lo → 1；中间线性（值越小越紧张）。hi 应大于 lo。"""
    if hi <= lo:
        return 1.0 if x <= lo else 0.0
    return clip01((hi - x) / (hi - lo))

