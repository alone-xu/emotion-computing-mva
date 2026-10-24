"""① 感知层 · 面部表情。

首选后端：MediaPipe FaceLandmarker（468 关键点）→ 几何推导 AU 代理特征。
降级后端：OpenCV Haar 人脸检测 → 面积/运动启发式。

实测到的两个真实版本差异（`[F]`，本机 Python 3.13.5 实测）：
  * ``mediapipe 1.1.0`` **没有** ``mp.solutions``（legacy API 已移除），
    只有 ``mp.tasks``；需要 ``.task`` 模型文件，默认放在 ``models/`` 下。
  * ``opencv-python 5.0.0`` 的 ``cv2`` **没有** ``CascadeClassifier``
    （5.x 移除了旧 objdetect API），因此 Haar 降级路径在 5.x 上不可用。
  本模块对上述两种情况都如实处理：不假装自己在做人脸检测，而是逐级降级。

诚实性说明 [V]（重要）：
    关键点索引来自 FaceMesh 公开拓扑，几何阈值来自 FACS 的常识映射，
    但**未在本项目中用带标注的表情数据验证过**。
    因此本模块输出的是「紧张相关的面部活动」的启发式代理量，
    **不是 AU 分类结果，也不等价于任何经过验证的表情识别模型**。
"""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import ModalityObservation
from .base import ModalitySensor, ramp_down, ramp_up

#: 默认模型目录（项目根 / models）
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3] / "models"

# --------------------------------------------------------------------------
# FaceMesh 关键点索引（公开拓扑）
# --------------------------------------------------------------------------

_LM: dict[str, int] = {
    "left_brow_inner": 107,
    "right_brow_inner": 336,
    "left_brow_top": 105,
    "right_brow_top": 334,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_eye_top": 159,
    "left_eye_bottom": 145,
    "right_eye_top": 386,
    "right_eye_bottom": 374,
    "mouth_left": 61,
    "mouth_right": 291,
    "mouth_top": 13,
    "mouth_bottom": 14,
    "nose_tip": 1,
}

#: 各面部特征的紧张度权重（启发式初值 [A]，可校准）
FACE_FEATURE_WEIGHTS: dict[str, float] = {
    "brow_furrow": 0.30,  # AU4 皱眉
    "eye_openness": 0.15,  # AU5/AU7 眼部紧张
    "mouth_tension": 0.20,  # AU23/24 唇部紧张
    "mouth_corner_down": 0.15,  # AU15 嘴角下拉
    "head_motion": 0.10,  # 不安微动
    "blink_rate": 0.10,  # 焦虑性眨眼
}


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def face_features_from_landmarks(pts: np.ndarray) -> dict[str, float]:
    """从 (N, 3) 归一化关键点中提取面部紧张相关几何特征。"""
    xy = np.asarray(pts, dtype=float)[:, :2]

    def p(name: str) -> np.ndarray:
        return xy[_LM[name]]

    interocular = _dist(p("left_eye_inner"), p("right_eye_inner")) + 1e-9
    eye_width_l = _dist(p("left_eye_outer"), p("left_eye_inner")) + 1e-9
    eye_width_r = _dist(p("right_eye_inner"), p("right_eye_outer")) + 1e-9

    # Eye Aspect Ratio：值越小眼睛越闭合
    ear = 0.5 * (
        _dist(p("left_eye_top"), p("left_eye_bottom")) / eye_width_l
        + _dist(p("right_eye_top"), p("right_eye_bottom")) / eye_width_r
    )

    # 眉内端间距（皱眉时变小）
    brow_furrow = _dist(p("left_brow_inner"), p("right_brow_inner")) / interocular
    # 眉-眼垂直距离（眉上扬时变大）
    brow_raise = (
        _dist(p("left_brow_top"), p("left_eye_top"))
        + _dist(p("right_brow_top"), p("right_eye_top"))
    ) / (2.0 * interocular)

    mouth_width = _dist(p("mouth_left"), p("mouth_right")) + 1e-9
    mouth_height = _dist(p("mouth_top"), p("mouth_bottom"))
    mouth_ratio = mouth_height / mouth_width  # 抿嘴时变小

    # 嘴角相对唇中线的垂直偏移：坐标 y 向下，正值 = 嘴角下垂
    mouth_mid_y = 0.5 * (p("mouth_top")[1] + p("mouth_bottom")[1])
    corner_y = 0.5 * (p("mouth_left")[1] + p("mouth_right")[1])
    mouth_corner_down = (corner_y - mouth_mid_y) / interocular

    return {
        "brow_furrow": float(brow_furrow),
        "brow_raise": float(brow_raise),
        "eye_openness": float(ear),
        "mouth_tension": float(mouth_ratio),
        "mouth_corner_down": float(mouth_corner_down),
        "interocular": float(interocular),
    }


def face_score(
    features: dict[str, float], blink_rate: float = 0.0, head_motion: float = 0.0
) -> float:
    """把几何特征压缩成 [0,1] 的面部紧张度分项（启发式 [V]）。"""
    comps = {
        # 皱眉：比值越小越紧张（静息约 0.40，明显皱眉约 0.26）
        "brow_furrow": ramp_down(features.get("brow_furrow", 0.40), hi=0.42, lo=0.26),
        # 眼部睁大 / 眼睑紧张
        "eye_openness": ramp_up(features.get("eye_openness", 0.30), lo=0.24, hi=0.42),
        # 抿嘴（嘴宽高比变小）
        "mouth_tension": ramp_down(features.get("mouth_tension", 0.30), hi=0.32, lo=0.06),
        # 嘴角下垂
        "mouth_corner_down": ramp_up(
            features.get("mouth_corner_down", 0.0), lo=0.0, hi=0.06
        ),
        # 头部微动（归一化位移/秒）
        "head_motion": ramp_up(head_motion, lo=0.005, hi=0.060),
        # 眨眼率（次/分）：静息约 15-20，焦虑常 > 25
        "blink_rate": ramp_up(blink_rate, lo=20.0, hi=45.0),
    }
    total_w = sum(FACE_FEATURE_WEIGHTS.values())
    score = sum(FACE_FEATURE_WEIGHTS[k] * comps[k] for k in FACE_FEATURE_WEIGHTS)
    return float(np.clip(score / total_w, 0.0, 1.0))


class _FrameTracker:
    """跨帧状态：头部运动、眨眼率。两个后端共用。"""

    def __init__(self) -> None:
        self._prev_nose: np.ndarray | None = None
        self._prev_t: float | None = None
        self._eye_open = True
        self._blinks: deque[float] = deque(maxlen=128)

    def update(self, nose_xy: np.ndarray, ear: float, t: float) -> tuple[float, float]:
        """返回 (head_motion, blink_rate)。"""
        head_motion = 0.0
        if self._prev_nose is not None and self._prev_t is not None:
            dt = max(t - self._prev_t, 1e-6)
            head_motion = float(np.linalg.norm(nose_xy - self._prev_nose) / dt)
        self._prev_nose, self._prev_t = np.asarray(nose_xy, dtype=float), t

        opened = ear > 0.19
        if self._eye_open and not opened:
            self._blinks.append(t)
        self._eye_open = opened

        window = 30.0
        recent = [b for b in self._blinks if t - b <= window]
        span = min(window, max(t, 1.0))
        blink_rate = float(len(recent) * 60.0 / span)
        return head_motion, blink_rate


# --------------------------------------------------------------------------
# 首选后端：MediaPipe FaceLandmarker
# --------------------------------------------------------------------------


class MediaPipeFaceSensor(ModalitySensor):
    name = "face"

    def __init__(self, config: Any, driver: Any = None) -> None:
        super().__init__()
        from .synthetic import SyntheticStressDriver

        self.driver = driver if isinstance(driver, SyntheticStressDriver) else None
        self.config = config
        self._tracker = _FrameTracker()
        self._last_ts_ms = -1

        import mediapipe as mp

        self._mp = mp
        self._cv2 = __import__("cv2")

        # A) 新版 Tasks API（mediapipe >= 1.0：legacy solutions 已移除）
        if hasattr(mp, "tasks"):
            self._landmarker = self._build_tasks_landmarker(mp)
            self._mode = "tasks"
            self.backend = "mediapipe-task"

        # B) 旧版 solutions API（mediapipe < 1.0）
        elif hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._mode = "solutions"
            self.backend = "mediapipe"

        else:
            raise RuntimeError(
                f"mediapipe {getattr(mp, '__version__', '?')} 既不含 tasks 也不含 solutions"
            )

    # ------------------------------------------------------------------
    def _build_tasks_landmarker(self, mp: Any) -> Any:
        from mediapipe.tasks.python import vision

        model_dir = Path(os.environ.get("EC_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
        model_path = Path(
            os.environ.get("EC_FACE_LANDMARKER", str(model_dir / "face_landmarker.task"))
        )
        if not model_path.exists():
            raise RuntimeError(
                f"缺少 FaceLandmarker 模型文件：{model_path}；"
                "请运行 python scripts/download_models.py 下载"
            )
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return vision.FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    def _landmarks_from_frame(self, frame: np.ndarray, t: float) -> np.ndarray | None:
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)

        if self._mode == "tasks":
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB, data=rgb
            )
            ts_ms = int(round(t * 1000))
            if ts_ms <= self._last_ts_ms:  # Tasks VIDEO 模式要求严格递增
                ts_ms = self._last_ts_ms + 1
            self._last_ts_ms = ts_ms
            result = self._landmarker.detect_for_video(mp_image, ts_ms)
            faces = getattr(result, "face_landmarks", None)
            if not faces:
                return None
            return np.array([[lm.x, lm.y, lm.z] for lm in faces[0]], dtype=float)

        # solutions 路径
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in result.multi_face_landmarks[0].landmark],
            dtype=float,
        )

    # ------------------------------------------------------------------
    def sense(self, payload: dict[str, Any]) -> ModalityObservation:
        t = float(payload.get("t", 0.0))
        frame = payload.get("frame")
        if frame is None:
            return self._synthetic_fallback(t, "no_frame")

        try:
            pts = self._landmarks_from_frame(frame, t)
            if pts is None or len(pts) <= max(_LM.values()):
                return self._missing(t, "no_face_detected")

            feats = face_features_from_landmarks(pts)
            nose = pts[_LM["nose_tip"], :2]
            head_motion, blink_rate = self._tracker.update(
                nose, feats["eye_openness"], t
            )

            score = face_score(feats, blink_rate=blink_rate, head_motion=head_motion)
            features = {**feats, "blink_rate": blink_rate, "head_motion": head_motion}
            return ModalityObservation(
                modality=self.name,
                timestamp=t,
                score=score,
                confidence=0.8,
                available=True,
                backend=self.backend,
                features={k: float(v) for k, v in features.items()},
                note="几何代理特征，未经表情标注校准 [V]",
            )
        except Exception as exc:  # noqa: BLE001
            return self._synthetic_fallback(t, f"mediapipe_error:{exc!r}")


# --------------------------------------------------------------------------
# 降级后端 1：OpenCV DNN（YuNet）
# --------------------------------------------------------------------------


class YuNetFaceSensor(ModalitySensor):
    """OpenCV DNN 人脸检测器（YuNet）后端。

    `[F]` 本机实测：``cv2 5.0.0`` 移除了 Haar 的 ``CascadeClassifier``，
    但**保留了** ``cv2.FaceDetectorYN`` 与 ``cv2.dnn.readNetFromONNX``。
    因此这是 OpenCV 5.x 上真实可用的人脸检测路径，只需一个 ~230KB 的 onnx 模型。

    输入：人脸框 + 5 个关键点（右眼、左眼、鼻尖、右嘴角、左嘴角）。
    可推导：瞳距、眼-嘴垂直距离、嘴部宽度、人脸尺度变化、头部运动。

    局限（重要，[V]）：5 个点远少于 FaceMesh 的 468 点，**无法**得到
    眉毛/眼睑等 AU 代理特征，因此特征丰富度与置信度都低于 MediaPipe 后端。
    这里给出的是「人脸几何活动」的粗糙代理，不是表情识别结果。
    """

    name = "face"

    def __init__(self, config: Any, driver: Any = None) -> None:
        super().__init__()
        from .synthetic import SyntheticStressDriver

        self.driver = driver if isinstance(driver, SyntheticStressDriver) else None
        self.config = config

        import cv2

        self._cv2 = cv2
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError(f"当前 OpenCV {cv2.__version__} 不提供 FaceDetectorYN")

        model_dir = Path(os.environ.get("EC_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
        model_path = Path(
            os.environ.get(
                "EC_YUNET_MODEL", str(model_dir / "face_detection_yunet_2023mar.onnx")
            )
        )
        if not model_path.exists():
            raise RuntimeError(
                f"缺少 YuNet 模型文件：{model_path}；"
                "请运行 python scripts/download_models.py 下载"
            )

        self._detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), 0.5, 0.3, 50
        )
        self.backend = "yunet"

        self._prev_nose: np.ndarray | None = None
        self._prev_t: float | None = None
        self._sizes: deque[float] = deque(maxlen=60)

    def sense(self, payload: dict[str, Any]) -> ModalityObservation:
        t = float(payload.get("t", 0.0))
        frame = payload.get("frame")
        if frame is None:
            return self._synthetic_fallback(t, "no_frame")

        try:
            height, width = frame.shape[:2]
            self._detector.setInputSize((int(width), int(height)))
            _, faces = self._detector.detect(frame)
            if faces is None or len(faces) == 0:
                return self._missing(t, "no_face_detected")

            face = max(faces, key=lambda f: float(f[2]) * float(f[3]))
            fw, fh = float(face[2]), float(face[3])
            right_eye = np.array([float(face[4]), float(face[5])])
            left_eye = np.array([float(face[6]), float(face[7])])
            nose = np.array([float(face[8]), float(face[9])])
            right_mouth = np.array([float(face[10]), float(face[11])])
            left_mouth = np.array([float(face[12]), float(face[13])])

            interocular = float(np.linalg.norm(left_eye - right_eye)) + 1e-9
            eye_mid = (left_eye + right_eye) / 2.0
            mouth_mid = (left_mouth + right_mouth) / 2.0
            mouth_width = float(np.linalg.norm(left_mouth - right_mouth)) + 1e-9
            eye_mouth = float(np.linalg.norm(mouth_mid - eye_mid)) / interocular
            face_ratio = fw / (fh + 1e-9)
            face_scale = (fw * fh) / (width * height + 1e-9)

            head_motion = 0.0
            if self._prev_nose is not None and self._prev_t is not None:
                dt = max(t - self._prev_t, 1e-6)
                head_motion = float(np.linalg.norm(nose - self._prev_nose) / dt)
            self._prev_nose, self._prev_t = nose, t

            self._sizes.append(face_scale)
            scale_change = 0.0
            if len(self._sizes) > 5:
                ref = float(np.median(self._sizes)) + 1e-9
                scale_change = float(abs(self._sizes[-1] - ref) / ref)

            det_conf = float(face[14]) if len(face) > 14 else 0.6

            # 启发式组合（阈值均为 [V] 待校准）
            score = float(
                np.clip(
                    0.45 * ramp_up(head_motion, lo=0.005, hi=0.050)
                    + 0.25 * ramp_up(abs(eye_mouth - 1.6), lo=0.15, hi=0.50)
                    + 0.20 * ramp_up(scale_change, lo=0.05, hi=0.25)
                    + 0.10 * ramp_up(abs(face_ratio - 0.78), lo=0.08, hi=0.25),
                    0.0,
                    1.0,
                )
            )
            return ModalityObservation(
                modality=self.name,
                timestamp=t,
                score=score,
                confidence=float(np.clip(0.55 * det_conf + 0.20, 0.0, 1.0)),
                available=True,
                backend="yunet",
                features={
                    "interocular": interocular,
                    "eye_mouth": eye_mouth,
                    "mouth_width": mouth_width,
                    "face_ratio": face_ratio,
                    "face_scale": face_scale,
                    "scale_change": scale_change,
                    "head_motion": head_motion,
                    "det_confidence": det_conf,
                },
                note="YuNet 5 点几何代理，特征丰富度低于 FaceMesh [V]",
            )
        except Exception as exc:  # noqa: BLE001
            return self._synthetic_fallback(t, f"yunet_error:{exc!r}")


# --------------------------------------------------------------------------
# 降级后端 2：OpenCV Haar
# --------------------------------------------------------------------------


def _load_haar_cascade(cv2: Any) -> Any:
    """在不同 OpenCV 版本中定位 Haar 级联分类器。

    `[F]` 实测事实：本机 ``opencv-python 5.0.0.93`` 的 ``cv2`` 中**不存在**
    ``CascadeClassifier``（顶层与 ``cv2.objdetect`` 都没有），OpenCV 5.x
    移除了这套旧 objdetect API。此时本降级后端不可用，应如实抛错、
    交给下一级降级（synthetic），而不是假装自己在做人脸检测。
    """
    cls = getattr(cv2, "CascadeClassifier", None)
    if cls is None:
        objdetect = getattr(cv2, "objdetect", None)
        cls = getattr(objdetect, "CascadeClassifier", None) if objdetect else None
    if cls is None:
        raise RuntimeError(
            f"当前 OpenCV {getattr(cv2, '__version__', '?')} 不提供 CascadeClassifier"
            "（5.x 已移除该 API）；请安装 mediapipe 作为面部首选后端"
        )

    cascade_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if not cascade_dir:
        raise RuntimeError("cv2.data.haarcascades 不可用")
    path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
    if not os.path.exists(path):
        raise RuntimeError(f"找不到级联文件：{path}")

    cascade = cls(path)
    if cascade.empty():
        raise RuntimeError(f"级联文件加载失败：{path}")
    return cascade


class HaarFaceSensor(ModalitySensor):
    """降级实现：只能拿到人脸框，特征退化为面积变化与运动能量。"""

    name = "face"

    def __init__(self, config: Any, driver: Any = None) -> None:
        super().__init__()
        from .synthetic import SyntheticStressDriver

        self.driver = driver if isinstance(driver, SyntheticStressDriver) else None
        self.config = config

        import cv2

        self._cv2 = cv2
        self._cascade = _load_haar_cascade(cv2)
        self.backend = "haar"

        self._prev_center: np.ndarray | None = None
        self._prev_t: float | None = None
        self._sizes: deque[float] = deque(maxlen=60)

    def sense(self, payload: dict[str, Any]) -> ModalityObservation:
        t = float(payload.get("t", 0.0))
        frame = payload.get("frame")
        if frame is None:
            return self._synthetic_fallback(t, "no_frame")

        try:
            gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
            if len(faces) == 0:
                return self._missing(t, "no_face_detected")

            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            center = np.array([x + w / 2.0, y + h / 2.0])
            area = float(w * h)

            motion = 0.0
            if self._prev_center is not None and self._prev_t is not None:
                dt = max(t - self._prev_t, 1e-6)
                motion = float(np.linalg.norm(center - self._prev_center) / dt)
            self._prev_center, self._prev_t = center, t
            self._sizes.append(area)

            area_change = 0.0
            if len(self._sizes) > 5:
                ref = float(np.median(self._sizes)) + 1e-9
                area_change = float(abs(self._sizes[-1] - ref) / ref)

            score = float(
                np.clip(
                    0.6 * ramp_up(motion, lo=1.0, hi=25.0)
                    + 0.4 * ramp_up(area_change, lo=0.05, hi=0.35),
                    0.0,
                    1.0,
                )
            )
            return ModalityObservation(
                modality=self.name,
                timestamp=t,
                score=score,
                confidence=0.4,  # 降级后端，置信度打折
                available=True,
                backend="haar",
                features={
                    "face_area": area,
                    "motion": motion,
                    "area_change": area_change,
                },
                note="Haar 降级后端，仅使用面积/运动启发式",
            )
        except Exception as exc:  # noqa: BLE001
            return self._synthetic_fallback(t, f"haar_error:{exc!r}")
