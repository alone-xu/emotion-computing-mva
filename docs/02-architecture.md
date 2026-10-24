# 多模态情感计算系统 —— 架构设计文档（MVA v0.1）

> 上游依据：`01-requirements.md`。本文定义**怎么实现**：分层、契约、算法、时序、降级。
> 五层顺序固定：**感知 → 快系统 → 轻量慢系统 → 记忆 → 行动**（这也是数据流动的顺序）。

---

## 1. 架构总览

```
                    ┌──────────────────────────────────────────────────────────┐
   视频帧 ──┐       │                      五层数据流                            │
   音频帧 ──┼──► ① 感知层 ──► ② 快系统 ──► ③ 轻量慢系统 ──► ⑤ 行动层 ──► 干预输出
   文本   ──┘       │  Perception   Fast System   Lite Slow       Action      │
                    │       │            │            │              ▲        │
                    │       └────────────┴────────────┘              │        │
                    │                    │                           │        │
                    │                    ▼                           │        │
                    │            ④ 记忆层 Memory ────────────────────┘        │
                    │        (存储 / 检索 / 复盘 / 个性化回写)                  │
                    └──────────────────────────────────────────────────────────┘
```

**分层职责与节奏**

| 层 | 角色 | 节奏 | 时间尺度 | 类比 |
|---|---|---|---|---|
| ① 感知 Perception | 把原始信号变成统一模态观测 | 每帧（~30fps） | 33ms | 感官 |
| ② 快系统 Fast System | 即时紧张度 + 事件 | 每帧 | 33ms | System 1（直觉） |
| ③ 轻量慢系统 Lite Slow | 趋势 / 预测 / 可继续性 | 每 5s | 秒~分钟 | System 2（审慎，轻量版） |
| ④ 记忆 Memory | 沉淀、检索、复盘、回写 | 每 1s 缓冲 / 会话末 | 会话~跨会话 | 海马体 |
| ⑤ 行动 Action | 分级干预决策 | 每 5s 或事件触发 | 秒~分钟 | 运动皮层 |

**关键设计决定**

- `[A]` **层间只通过冻结契约（dataclass）通信**，不传原始信号对象。层可独立替换与单测。
- `[A]` **快系统与慢系统物理分离**：快系统绝不阻塞在慢系统上；慢系统在独立缓冲上异步评估。
- `[A]` **记忆层是"旁路 + 回环"**：实时路径不依赖记忆（保证低延迟），但行动层与慢系统可读记忆（个性化）；
- `[A]` **闭环**：行动 → 影响使用者 → 下一轮感知，使系统进入动态调节回路。

---

## 2. 设计原则

1. **契约先行**：任何跨层数据必须是有类型的 dataclass，禁止 `dict` 裸传。
2. **真值可退化**：每个真实实现都有降级路径；缺依赖/缺设备时自动回退并**只告警一次**。
3. **纯函数内核**：融合、归一化、平滑、预测、决策一律是无状态纯函数，便于单测与复现。
4. **可解释即一等公民**：每个数值输出都携带 `contributions` / `reasons`。
5. **隐私即架构**：原始音视频只在内存中存活当帧生命周期，永不落盘。
6. **诚实标注**：所有未经真实数据校准的指标，代码里用 `UNVALIDATED` 常量与字段名显式标记。

---

## 3. 目录结构

```
emotion-computing/
├── README.md
├── pyproject.toml
├── requirements.txt
├── docs/
│   ├── 01-requirements.md
│   └── 02-architecture.md
├── src/emotion_computing/
│   ├── __init__.py
│   ├── config.py                 # 配置（阈值/权重/时间常数）
│   ├── contracts.py              # ★ 全部跨层数据契约（本文第 4 节）
│   ├── perception/               # ① 感知层
│   │   ├── __init__.py
│   │   ├── base.py               # ModalitySensor 抽象基类 + 降级装配
│   │   ├── face.py               # MediaPipe FaceMesh → AU 代理特征
│   │   ├── speech.py             # librosa 声学特征
│   │   ├── text.py               # transformers 或有词典
│   │   ├── fusion.py             # 多模态融合 + 基线归一化
│   │   └── synthetic.py          # 合成信号源（演示/测试用）
│   ├── fast_system/              # ② 快系统
│   │   ├── __init__.py
│   │   ├── estimator.py          # EWMA 平滑 + 样本产出
│   │   └── events.py             # 尖峰/持续高位事件检测
│   ├── slow_system/              # ③ 轻量慢系统
│   │   ├── __init__.py
│   │   ├── trend.py              # 稳健斜率 + CUSUM 变点
│   │   ├── predictor.py          # OU + 趋势外推的短期预测
│   │   └── capacity.py           # 可继续性状态机
│   ├── memory/                   # ④ 记忆层
│   │   ├── __init__.py
│   │   ├── store.py              # Episode JSONL 存储
│   │   ├── retrieval.py          # 相似情境检索
│   │   └── review.py             # 复盘报告生成
│   ├── action/                   # ⑤ 行动层
│   │   ├── __init__.py
│   │   ├── policy.py             # 分级决策 + 迟滞
│   │   └── interventions.py      # 干预库
│   ├── pipeline.py               # 五层编排（在线会话主循环）
│   └── cli.py                    # 命令行入口
├── scripts/
│   ├── demo_end_to_end.py        # 一条命令跑通的端到端演示
│   └── check_environment.py      # 探测本机可用的真实后端
├── tests/
│   ├── test_fusion.py
│   ├── test_fast_system.py
│   ├── test_slow_system.py
│   ├── test_memory.py
│   └── test_action.py
└── data/                         # 运行时产物（episodes.jsonl / reports/）
```

---

## 4. 跨层数据契约（`contracts.py`）

这是全系统最重要的文件。所有层只认这些类型。

```python
# ---------- ① 感知层输出 ----------
@dataclass(frozen=True)
class ModalityObservation:
    modality: str            # "face" | "speech" | "text"
    timestamp: float         # 秒（单调时钟）
    score: float             # 该模态的紧张度分项 ∈ [0,1]
    confidence: float        # ∈ [0,1]，数据质量
    available: bool          # False = 本帧该模态缺失
    backend: str             # 实际生效的实现："mediapipe" | "haar" | "lexicon" | "synthetic"
    features: dict[str, float]  # 可解释的中间特征

@dataclass(frozen=True)
class PerceptionFrame:
    timestamp: float
    observations: tuple[ModalityObservation, ...]
    fused_raw: float             # 融合后的原始紧张度 ∈ [0,1]
    normalized: float            # 基线归一化后 ∈ [0,1]
    contributions: dict[str, float]  # 各模态贡献占比，和为 1（可用模态）

# ---------- ② 快系统输出 ----------
@dataclass(frozen=True)
class StressSample:
    timestamp: float
    raw: float
    smoothed: float              # EWMA 结果，系统对外的主实时值
    confidence: float
    contributions: dict[str, float]

@dataclass(frozen=True)
class StressEvent:
    timestamp: float
    kind: str                    # "spike" | "rise" | "fall" | "sustained_high"
    magnitude: float
    dominant_modality: str

# ---------- ③ 轻量慢系统输出 ----------
@dataclass(frozen=True)
class Forecast:
    horizon_s: float
    point: float
    lower: float
    upper: float
    slope: float                 # 当前趋势斜率（每分钟）
    baseline: float              # 个体基线（均值回复目标）
    method: str                  # 例如 "ou_trend_v1"
    validated: bool = False      # ★ 诚实标记：未经真实数据校准

@dataclass(frozen=True)
class CapacityAssessment:
    state: str                   # "OK" | "WATCH" | "PAUSE"
    score: float                 # ∈ [0,1]，越大越能继续
    forecast: Forecast
    resilience: float            # 恢复力估计
    reasons: tuple[str, ...]     # 可解释理由

# ---------- ④ 记忆层 ----------
@dataclass
class Episode:
    episode_id: str
    started_at: str              # ISO8601
    ended_at: str | None
    task_label: str              # 任务类型（用于相似检索）
    baseline_mu: float
    baseline_sigma: float
    samples: list[dict]          # 精简时序（ts, raw, smoothed, contributions）
    events: list[dict]
    interventions: list[dict]
    task_outcome: float | None   # ∈ [0,1]，外部注入（正确率/进度）
    capacity_timeline: list[dict]

# ---------- ⑤ 行动层输出 ----------
@dataclass(frozen=True)
class ActionDecision:
    timestamp: float
    action: str                  # "continue" | "coach" | "micro_break" | "pause"
    intervention: str | None     # 干预库中的 id
    message: str                 # 面向使用者的话术
    rationale: str               # 为什么这么做（可解释）
    urgent: bool
```

**契约稳定性规则**：契约字段只增不改。新增字段必须有默认值。

---

## 5. 各层详细设计

### 5.1 ① 感知层（Perception）

#### 5.1.1 抽象接口

```python
class ModalitySensor(ABC):
    name: str
    def is_available(self) -> bool: ...
    def sense(self, payload) -> ModalityObservation: ...
```

每个 sensor 负责：`真实后端 → 降级后端 → 合成后端` 的自动装配（在 `__init__` 中探测定型，只告警一次）。

#### 5.1.2 面部表情 `face.py`

**首选后端：MediaPipe FaceMesh（468 关键点）** `[F]` 可安装（py3 wheel）。

从关键点几何推导 **AU 代理特征**（不训练分类器，纯几何，可解释、零样本）：

| 特征 | 计算 | 紧张含义 |
|---|---|---|
| `brow_furrow` | 眉内端间距 / 瞳距，越小越皱眉 | AU4 皱眉 |
| `brow_raise` | 眉-眼垂直距离 / 瞳距 | AU1+2 眉上扬 |
| `eye_openness` (EAR) | 眼纵横比 | AU7 眼睑紧张 / 惊讶 |
| `mouth_tension` | 嘴宽高比 MAR 的压缩度 | AU23/24 唇部紧张 |
| `mouth_corner_down` | 嘴角相对唇中线的垂直偏移 | AU15 嘴角下拉 |
| `head_motion` | 头部姿态（鼻尖-眼线）变化率 | 不安/微动 |
| `blink_rate` | EAR 穿越阈值的频率（30s 窗） | 焦虑性眨眼加速 |

模块级聚合：

```
face_score = clamp( Σ_w w_i · z_i , 0, 1 )   # z_i 为个体基线归一化后的特征
权重初值（启发式，可校准）[A]：
  brow_furrow 0.30, eye_openness 0.15, mouth_tension 0.20,
  mouth_corner_down 0.15, head_motion 0.10, blink_rate 0.10
```

**降级 1：OpenCV DNN YuNet**（`YuNetFaceSensor`）—— `cv2.FaceDetectorYN`，人脸框 + 5 个关键点
（双眼、鼻尖、双嘴角）。可推导瞳距、眼-嘴距离、嘴宽、人脸尺度变化、头部运动。
**5 点远少于 468 点，无法得到眉毛/眼睑等 AU 代理特征**，故置信度打折。

**降级 2：OpenCV Haar**（`HaarFaceSensor`）—— 只能得到人脸框，特征退化为面积变化与运动能量。
`[F]` 实测：`opencv-python 5.0.0` 已移除 `CascadeClassifier`，该路径在 5.x 上不可用。

**降级 3：合成** —— 见 5.1.5。

> `[F]` 实测补充：`mediapipe 1.1.0` 已移除 legacy `mp.solutions`，本模块改走
> Tasks API（`vision.FaceLandmarker` + `models/face_landmarker.task`）。

#### 5.1.3 语音 `speech.py`

**首选后端：librosa** `[F]` 已装，可真实运行。窗口默认 1.6s，hop 0.5s。

| 特征 | 方法 | 紧张含义 |
|---|---|---|
| `f0_mean` / `f0_std` | `pyin` 基频轨迹 | 音调升高、抖动 |
| `energy_rms` | RMS 能量 | 音量抬高/变弱 |
| `speech_rate_proxy` | 有声段/秒（音节率代理） | 语速加快 |
| `pause_ratio` | 静音占比 | 停顿异常（卡壳） |
| `jitter` | F0 相邻周期差 | 声带不稳 |
| `hnr` | 谐噪比 | 嗓音紧张度（下降） |

```
speech_score = clamp( Σ w·z , 0, 1 )   权重 [A]：f0_std .30, energy .20,
                                          speech_rate .15, pause_ratio .15,
                                          jitter .10, hnr .10(反向)
```

**可选升级**：ASR（faster-whisper）转写 → 喂给语义模块。MVA 中作为可选依赖。

#### 5.1.4 语义 `text.py`

**首选后端：transformers** 情感/焦虑模型（中文优先，如中文 RoBERTa 情感分类）。
**降级 1：词典法** —— 内置中文/英文焦虑词典 + 否定词 + 程度副词规则（无需下载，永远可用）：

```
text_score = clamp( (Σ 焦虑词权 + 程度放大 - 否定翻转) / 归一化基数 , 0, 1 )
```

**降级 2：合成**。

#### 5.1.5 合成信号源 `synthetic.py`

用途：**演示与测试**。生成可控的紧张度曲线（静息 → 上升 → 峰值 → 恢复 → 二次上升），并注入噪声与模态缺失，用于验证融合/平滑/预测/决策的正确性。这是 E-3（降级验收）与单测的基础设施。

#### 5.1.6 融合与基线归一化 `fusion.py`

**Step 1 — 基线标定（会话前 30s 静息期）**

```
对每一维特征 i：μ_i, σ_i  ← 静息期样本
```

**Step 2 — 特征级归一化（零样本，无需训练）**

```
z_i = (x_i - μ_i) / max(σ_i, ε)          # 个体内 z-score
s_i = sigmoid( (z_i - z_target) / τ )     # 映射到 [0,1]；z_target 为"紧张起点"
```

**Step 3 — 模态级融合（加权平均，天然支持缺失重分配）**

```
S_raw = Σ_m ( w_m · c_m · s_m ) / Σ_m ( w_m · c_m )
        其中仅对 available 的模态求和；c_m 为 confidence
```

> 这正是需求 P-7 的实现：某模态缺失时，其权重自动从分母消失，**无需特殊分支**。

**Step 4 — 融合后再次基线归一化**（消除个体在"整体敏感度"上的差异）。

**贡献占比（可解释性）**：`contributions` 按 **w·c·s** 归一化，而不是只按 w·c。
若只按权重，占比是常数，"主导模态"永远等于权重最大的模态，归因失效；
按 w·c·s 归一化后，占比才反映「哪个模态真正把紧张度推了上去」。

#### 5.1.7 隐私约束的实现方式

- `face.py` 只把**关键点坐标**转成 `float` 特征返回，**帧图像在处理后立即丢弃引用**。
- `speech.py` 只返回统计量，音频数组不进入任何持久化路径。
- 代码层面：`Episode.samples` 结构体中**不存在**任何原始数据字段。

---

### 5.2 ② 快系统（Fast System）

#### 5.2.1 EWMA 平滑 `estimator.py`

时间常数 τ_fast 默认 1.5s，用真实 Δt 计算系数，保证不同帧率下行为一致：

```
α = 1 - exp( -Δt / τ_fast )
S_t = α · S_raw + (1 - α) · S_{t-1}
```

#### 5.2.2 事件检测 `events.py`

- **spike**：`S_t - S_{t-k} > θ_spike` 且该变化 > `k_spike × σ_recent`（k=3）。
- **sustained_high**：`S_t > θ_hi` 持续 `T_hi`（默认 20s）。
- **rise / fall**：窗口斜率方向显著但幅度未达 spike。
- 事件携带 `dominant_modality` = 贡献占比最大的模态（可解释归因）。

#### 5.2.3 性能预算

单帧预算 33ms：面部关键点 ~15ms、语音（每 hop 一次，摊薄）、融合 <1ms。实际不达标时按需求 N4 不做优化，仅记录耗时。

---

### 5.3 ③ 轻量慢系统（Lite Slow System）

> 「轻量」的硬约束：**不训练神经网络、不引入深度模型**。全部为可解释统计方法。

#### 5.3.1 趋势与变点 `trend.py`

- **稳健斜率**：滑动窗口（默认 60s）做 Theil–Sen 或裁剪最小二乘，抗离群。
- **变点检测**：CUSUM，检测紧张度的持续性偏移（区分"波动"与"真的变糟了"）。

#### 5.3.2 短期预测 `predictor.py`（OU + 趋势外推）

物理直觉：紧张度有**均值回复**（人总会平复）叠加**当前趋势**（正在变糟）。

```
Ŝ(t+H) = S_t + slope·H + (μ_base - S_t)·(1 - exp(-H/τ_rev))
```

- `τ_rev`：均值回复时间常数，在线估计（初始 90s）。
- `slope`：来自 `trend.py` 的稳健斜率（单位：每分钟）。
- `μ_base`：个体基线，来自记忆层回写（无历史则用静息基线）。

**不确定性区间**：
```
σ_resid ← 滑动窗口内预测残差的稳健标准差
[lower, upper] = Ŝ ± 1.96·σ_resid·sqrt(H/H_ref)
```

**在线自适应**：每次评估后用真实观测更新 `τ_rev`、`σ_resid`（指数平滑的简单梯度），**无需重训练**。

> `validated=False` 恒定标记：该方法**未经真实数据回测**，属方法学合理但精度未知 `[V]`。

#### 5.3.3 可继续性状态机 `capacity.py`

```
输入：S_t（当前）、Forecast(H=60s)、resilience（回落的平均速率）

score = clip( 1 - 0.6·Ŝ(t+60) - 0.2·max(slope,0)/slope_ref - 0.2·(1 - resilience) , 0, 1 )

state =
    "PAUSE"  若 Ŝ(t+60) ≥ θ_hi  或 (S_t ≥ θ_hi 持续 ≥ T_hi)
    "WATCH"  若 Ŝ(t+60) ≥ θ_mid 或 slope > 0 且持续上升
    "OK"     否则
```

**resilience**（恢复力）：历史中从峰值回落 50% 所需时间的倒数，经归一化。恢复快 = 还扛得住。

**reasons** 输出示例：`("预测 60s 后紧张度 0.81 超过阈值 0.75", "当前已连续 24s 高于 0.70", "近 10min 恢复速率低于基线")`。

---

### 5.4 ④ 记忆层（Memory）

#### 5.4.1 存储 `store.py`

- **主存储**：`data/episodes.jsonl`，一行一个 `Episode` 快照，**追加写**，人类可读。
  这是**追加式日志**：会话进行中每 `checkpoint_interval`（默认 60s）追加一条
  `status="running"` 快照，会话结束追加 `status="done"` 后压缩日志（每个 episode
  只留最后一行）。进程崩溃后重新构造 `EpisodeStore` 会把残留 `running` 标为
  `aborted` —— 长会话中途挂掉最多丢一个检查点间隔，而不是整场会话。
- **检索加速**：会话结束时把 Episode 的**聚合特征向量**写入 `data/index.jsonl`，
  并同步一份 `data/index.sqlite`（M-2 要求的 SQLite 索引）。
  `index.jsonl` 是事实来源、`index.sqlite` 是纯派生物，可用 `reindex` 重建。
- **原子写**：所有整文件重写（日志压缩 / 索引 / 画像）都是
  "同目录临时文件 + fsync + `os.replace`"；跨进程读写走 `episodes.jsonl.lock` 文件锁。
- **隐私门**：`assert_privacy_safe()` 在落盘前检查样本字段白名单、
  `notes` 只允许结构化短标签；越线抛 `PrivacyViolation`，拒绝写入。

聚合特征（用于相似检索）：
```
[ mean_S, max_S, std_S, slope_early, slope_late,
  time_to_peak_ratio, n_spikes, duration_min ]
```
每个特征带**固定物理尺度**（`FEATURE_SCALE`）与**权重**（`FEATURE_WEIGHTS`），
距离因此是绝对量、跨查询可比；`distance > max_z_distance` 的历史不返回。

#### 5.4.2 相似检索 `retrieval.py`

给定当前会话（可以是部分会话）的特征，用**加权标准化欧氏距离**返回 Top-K 历史片段。

三条约定的规则（评审后固定下来，不要再改回 top-k 内归一化）：

1. 标准化用每特征的**固定物理尺度**，不用样本内 std —— 距离是绝对量，
   同一对会话在 `top_k=1/3/5`、不同查询下算出的 `distance` 相同；
2. 有**硬阈值**：`distance > memory_max_z_distance`（默认 2.0）不算相似，
   宁可比空也不硬凑；
3. 有**特征权重**（紧张度/斜率 > 峰值位置 > 尖峰数 > 时长）与**时间衰减**
   （半衰期 `memory_half_life_days`，默认 180 天）。

记忆回流：pipeline 每个慢系统周期都用当前部分特征检索一次，
命中（`distance ≤ 1.5` 且历史峰值进过高位）则行动层提前一档介入，
并把"因为哪条历史"写进 `ActionDecision.rationale`。

#### 5.4.3 复盘 `review.py`

生成 `ReviewReport`：

1. **时间线**：紧张度演化（含事件标记、干预标记）——文本表格 + 可选 ASCII 曲线。
2. **峰值归因**：Top 峰值时刻，各模态贡献占比 → "这次峰值主要由 语音(F0 抖动) 驱动"。
3. **恢复画像**：平均恢复时间、恢复力趋势（跨会话）。
4. **与完成度关联**：`task_outcome` 与紧张度统计量的相关（若某会话没注入 outcome，则跳过并注明）。
5. **跨会话对比**：同类任务最近 N 次的关键指标对比。
6. **建议**：基于干预有效性记录给出可执行建议。

> 输出中必须包含 `UNVALIDATED` 免责段落（对应需求 7.2）。

#### 5.4.4 个性化回写

会话结束计算并入 `data/profile.json`：
```
baseline_mu / baseline_sigma / theta_hi / theta_mid / tau_rev / resilience / intervention_effectiveness{}
```
下次会话启动时由 pipeline 加载，实现"越用越懂这个人"（需求 M-5、场景 3）。

---

### 5.5 ⑤ 行动层（Action）

#### 5.5.1 决策 `policy.py`

```
输入：StressSample(实时) + CapacityAssessment(预测) + 历史干预记录

候选动作判定（优先级从高到低）：
  PAUSE 且 urgent 条件满足            → action = "pause"
  WATCH 持续 ≥ T_watch                → action = "micro_break"
  WATCH 初次                          → action = "coach"
  OK                                  → action = "continue"
```

**迟滞（hysteresis）实现**：为每个动作维护 `last_emitted` 时间与触发值；同一动作在 `cooldown`（默认 coach: 60s, micro_break: 120s, pause: 90s）内不重复输出。上下阈值分离（θ_up > θ_down）避免边界抖动。

**召回与干扰的平衡** `[A]`：默认偏保守（宁可漏报不要骚扰），因为需求 3 明确指出"干预不能变成另一种压力源"。

#### 5.5.2 干预库 `interventions.py`

| id | 触发场景 | 内容 |
|---|---|---|
| `breathing_478` | WATCH/PAUSE，高唤醒 | 4-7-8 呼吸引导 |
| `pacing` | 语速/音调升高 | 放慢节奏提示 |
| `task_downgrade` | PAUSE，多次触发 | 建议拆小任务/降难度 |
| `reframe` | 负向语义主导 | 认知重构短句 |
| `micro_reset` | micro_break | 20 秒离屏/伸展 |
| `encourage` | 恢复期 | 正向反馈 |

每条干预带 `rationale`，由 `policy.py` 填入具体数值理由。

#### 5.5.3 有效性闭环

干预输出后，系统记录干预前后 60s 的紧张度变化 `ΔS_intervention`。多会话累积后，同类情境优先选择历史上更有效的干预（写入 `profile.json` 的 `intervention_effectiveness`）。

---

## 6. 时序：一次会话的完整生命周期

```
t=0        启动 pipeline，加载 profile.json（记忆回写）
t∈[0,30s]  ① 基线期：采集静息特征 → μ_i, σ_i, μ_base
t=30s      进入任务期，开启快系统主循环
每帧(33ms) ① 感知：face / speech / text → PerceptionFrame
           ② 快系统：融合→EWMA→StressSample；检测事件
每1s       ④ 记忆：把 StressSample 推入会话缓冲
每5s       ③ 慢系统：trend → forecast → capacity
           ⑤ 行动：policy → ActionDecision
           ④ 记忆：记录 intervention
           ⑤ 若 action != continue → 输出 message（默认静默窗口仅记录）
任务结束   外部注入 task_outcome
会话末     ④ 记忆：封装 Episode → 追加 episodes.jsonl（终态）→ 压缩日志
           ④ 复盘：生成 ReviewReport → data/reports/<id>.md
           ④ 回写：更新 profile.json（基线 EMA + 阈值自适应 + 干预有效性）
```

会话进行中每 60s 还有一步：`④ 记忆：追加 status="running" 检查点`（崩溃恢复用）。

---

## 7. 降级策略的实现规范

```python
def build_sensor(name, config) -> ModalitySensor:
    for backend in BACKENDS[name]:          # 有序候选
        try:
            return backend(config)          # 惰性构造，import 在内部
        except Exception as exc:
            warn_once(f"[{name}] 后端 {backend.key} 不可用，降级: {exc}")
    return SyntheticSensor(name)            # 兜底，永不失败
```

强制要求：**任何一次降级都必须体现在 `ModalityObservation.backend` 字段上**，并在复盘报告中列出本次会话的实际后端组合。这样"系统到底跑了什么"永远可追溯。

---

## 8. 配置项（`config.py` 默认值）

| 键 | 默认 | 说明 |
|---|---|---|
| `baseline_seconds` | 30 | 静息基线期 |
| `tau_fast` | 1.5 | 快系统 EWMA 时间常数（s） |
| `theta_mid` | 0.55 | WATCH 阈值 |
| `theta_hi` | 0.75 | PAUSE 阈值 |
| `theta_spike` | 0.15 | 尖峰幅度阈值 |
| `T_hi` | 20 | 高位持续判据（s） |
| `T_watch` | 60 | WATCH 升级为 micro_break 的持续时长 |
| `forecast_horizon` | 60 | 预测视界（s） |
| `tau_rev` | 90 | 均值回复时间常数初值（s） |
| `w_face/w_speech/w_text` | 0.40/0.35/0.25 | 模态基础权重 |
| `cooldown` | 见 5.5.1 | 动作冷却 |

所有阈值均为**启发式初值** `[A]`，需在真实使用中按个体校准。

---

## 9. 测试策略

| 测试 | 目标 |
|---|---|
| `test_fusion.py` | 权重归一化、缺失模态重分配、基线 z-score 正确性 |
| `test_fast_system.py` | EWMA 在不同 Δt 下收敛一致；spike 检测正确触发/不误触发 |
| `test_slow_system.py` | 合成上升序列 → 预测应该也上升；均值回复使长期预测趋近 μ_base；状态机边界 |
| `test_memory.py` | JSONL 追加/读回、相似检索返回合理排序、复盘报告字段完整 |
| `test_action.py` | 迟滞生效（冷却期内不重复输出）、PAUSE 优先级、rationale 非空 |

端到端：`scripts/demo_end_to_end.py` 用合成信号跑完整生命周期，断言产出文件存在且结构合法。

---

## 10. 需求追踪矩阵

| 需求 | 落地位置 |
|---|---|
| P-1~P-2 | `perception/face.py` |
| P-3 | `perception/speech.py` |
| P-4~P-5 | `perception/text.py` |
| P-6 | `contracts.ModalityObservation` |
| P-7~P-8 | `perception/fusion.py` |
| P-9 | `perception/base.py` + 第 7 节 |
| F-1~F-6 | `fast_system/estimator.py`, `events.py` |
| L-1~L-6 | `slow_system/trend.py`, `predictor.py`, `capacity.py` |
| M-1~M-6 | `memory/store.py`, `retrieval.py`, `review.py` |
| A-1~A-6 | `action/policy.py`, `interventions.py` |
| E-1~E-4 | `scripts/demo_end_to_end.py`, `tests/` |

---

## 11. 版本记录

| 版本 | 变更 |
|---|---|
| v0.1 | 首版架构：五层 + 契约 + OU 预测 + 降级规范 + 追踪矩阵 |
