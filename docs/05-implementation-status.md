# 05 · 需求实现状态矩阵

> 这份文件存在的唯一目的：**让人不用读完 6000 行代码，也能一眼看出哪些需求实现了、哪些没有。**
>
> 需求编号来自 [`01-requirements.md`](01-requirements.md)（"必须"项）。
> 状态只有三种：**实现** / **部分实现**（说明缺什么）/ **未实现**（说明为什么）。
> 有争议请直接看"证据"列指到的代码，不必相信我写的这一行字。

最后核对：2026-10-24（对应 `pytest` **157 项全绿**）。

---

## ① 感知层（P）

| 编号 | 需求 | 状态 | 证据 / 说明 |
|---|---|---|---|
| P-1 | 面部检测 + 紧张相关表情特征 | 实现 | `perception/face.py`；三级后端 MediaPipe → YuNet → Haar → synthetic |
| P-2 | 7 项面部特征 | 实现 | `face_features_from_landmarks()` 输出 `brow_furrow`/`brow_raise`/`eye_openness`/`mouth_tension`/`mouth_corner_down` + `blink_rate`/`head_motion` |
| P-3 | 声学特征（F0/能量/语速代理/停顿比/jitter/shimmer/HNR） | 实现（shimmer 为代理） | `acoustic_features()`；shimmer 用帧级峰值幅度的相邻相对差近似，不是逐周期的严格定义，代码注释里已标明 |
| P-4 | 语义负向度 / 焦虑度 | 实现 | `perception/text.py`：transformers 可选，缺依赖降级为词典 + 否定/程度规则 |
| P-5 | ASR 转写（可选） | 未实现 | 语义模态当前吃"外部传入的文本"；接 faster-whisper 在路线图里 |
| P-6 | 统一模态观测契约 | 实现 | `contracts.ModalityObservation`（含 `available` / `confidence` / `backend`） |
| P-7 | 缺失模态自动重分配权重 | 实现 | `fusion.py`：`Σw·c·s / Σw·c`，缺失模态的权重自然从分母消失 |
| P-8 | 个体基线归一化 | 实现 | `fusion.py`：σ 收缩 `σ_eff=√(σ²+σ_prior²)` + sigmoid |
| P-9 | 降级策略 | 实现 | `perception/base.py` 装配链；降级事实写进 `ModalityObservation.backend` |

## ② 快系统（F）

| 编号 | 需求 | 状态 | 证据 |
|---|---|---|---|
| F-1 | 即时紧张度 ≤100ms/帧 | 实现 | `fast_system/estimator.py`（EWMA + 归一化，纯 numpy 无模型推理） |
| F-2 | EWMA 平滑 | 实现 | 时间感知 `α=1-exp(-Δt/τ)` |
| F-3/F-4 | 尖峰事件（spike/rise/fall/sustained_high，含幅度与主导模态） | 实现 | `fast_system/events.py` |
| F-5 | 只依赖短窗口 | 实现 | 滑动窗口 + 常数内存状态 |
| F-6 | 携带模态贡献分解 | 实现 | `StressSample.contributions` |

## ③ 轻量慢系统（L）

| 编号 | 需求 | 状态 | 证据 / 说明 |
|---|---|---|---|
| L-1 | 趋势 + 变点检测 | 实现 | `trend.py`：Theil–Sen 稳健斜率 + CUSUM |
| L-2 | 短期预测 + 区间 | 实现 | `predictor.py`：点估计 + 上下界 |
| L-3 | 可解释、在线自适应 | 实现 | OU 风格均值回复 + 趋势外推；`update_residual()` 在线校正 |
| L-4 | 可继续性状态机 | 实现 | `capacity.py`：OK / WATCH / PAUSE + 得分 + 理由 |
| L-5 | 归因：趋势由哪个模态驱动 | 实现（代理口径） | `capacity.assess(..., contributions=)` 输出"最近样本的紧张度主要由 X 驱动（占比）"；用**当前样本**做代理，不是窗口内逐帧严格分解 |
| L-6 | 预测精度未验证必须标注 | 实现 | `Forecast.validated` 恒为 `False`，`emotion_computing.UNVALIDATED` |

## ④ 记忆层（M）

| 编号 | 需求 | 状态 | 证据 |
|---|---|---|---|
| M-1 | 经验存储（特征/统计/事件/干预/结果） | 实现 | `memory/store.py`，`episodes.jsonl` 追加式日志 + 60s 检查点 |
| M-2 | JSONL + **SQLite 索引** | 实现 | `memory/index_db.py` 的 `index.sqlite`；JSONL 是事实来源，SQLite 可 `reindex` 重建 |
| M-3 | 相似情境检索 | 实现 | `retrieval.py`：加权标准化欧氏距离 + 绝对阈值 + 时间衰减 |
| M-4 | 复盘报告（时间线/峰值归因/恢复耗时/完成度关联） | 实现 | `review.py` → `reports/<episode_id>.md` |
| M-5 | 基线/阈值/干预有效性回写 | 实现 | `pipeline._update_profile()` + `_learn_thresholds()`，依据写入 `profile.threshold_evidence` |
| M-6 | 不存原始音视频 | 实现 | `store.assert_privacy_safe()`：样本字段白名单 + `notes` 只允许结构化短标签，越线抛 `PrivacyViolation` |

## ⑤ 行动层（A）

| 编号 | 需求 | 状态 | 证据 |
|---|---|---|---|
| A-1 | 综合实时值 + 预测 + **记忆类比**分级动作 | 实现 | `action/policy.py`：`history_prior` + `memory_hint` 两路历史证据 |
| A-2 | 四级动作 | 实现 | `continue` / `coach` / `micro_break` / `pause` |
| A-3 | 迟滞（冷却 + 阈值带宽） | 实现 | 冷却表 + `theta_mid/theta_hi` 分离 |
| A-4 | 干预可解释 | 实现 | `ActionDecision.rationale` 写清触发理由与历史依据 |
| A-5 | 干预效果回写 | 实现 | `_settle_interventions()` → 干预前后 ΔS 的 EMA |
| A-6 | 默认静默 | 实现 | 只在状态升级或回落确认时出话 |

## ⑥ 审核层（D/M/F/V/I/L）

| 编号 | 需求 | 状态 | 证据 |
|---|---|---|---|
| D-# | 数据层审核（一致性 α、抽检覆盖、分布） | 实现 | `audit/data_audit.py`；`α < 0.67` 主动中止 |
| M-1/M-2 | 逐模块指标 + 消融 | 实现 | `audit/module_evaluator.py` |
| M-3 | 模态破坏者识别 | 实现 | `audit/modality_diagnostic.py` |
| F-1/F-2/F-3 | 七类失败归因 + 分布 + 修复优先级 | 实现 | `audit/failure_attribution.py` |
| V-1 | 单文件可视化 | 实现 | `audit/viz.py` → `dashboard.html`（无外部依赖） |
| L-1 | 审核日志与追溯 | 实现 | `audit/log.py` → `_audit_log.jsonl` |

---

## 明确不做的事（Scoped out）

这些不是"没写完"，是**主动排除**，写在这里避免误解：

| 不做 | 原因 |
|---|---|
| 临床/医学诊断 | 输出是个体内相对量，没有临床效度数据就宣称诊断是耍流氓 |
| 跨人比较、跨人泛化训练 | MVA 阶段的数据量撑不起任何泛化结论 |
| 深度多模态融合大模型 | 需求里"轻量"二字是硬约束；且本地实时推理优先 |
| 对外给出"准确率/预测精度"数字 | 预测模型未经真实数据回测，`validated=False` |
| 员工考核、绩效评估类用途 | 伦理红线，见需求文档 §3 |

## 已知限制（不是 bug，是当前阶段的边界）

1. **没有金标准压力标注数据**：所有阈值/权重是启发式初值，`UNVALIDATED` 常量就是为此存在；
2. **RAVDESS 是表演语音，不是真实高压任务**：它只能验证"管线能跑通 + 归因链路可用"，
   不能用来论证效果（`examples/README.md` 里写明了这次跑批是**注入已知错误**来验证归因）；
3. **会话数少的时候阈值学习会抖**：`theta_mid = p60(最近 N 次同类会话均值)`，
   N < 2 时沿用旧值并在 `threshold_evidence` 里标 `insufficient_data`；
4. **Windows 优先**：跨进程文件锁在 Windows 用 `msvcrt`、POSIX 用 `flock`，两侧都有 CI。

## 怎么自己 verify（别只看文档）

```bash
python -m pytest tests -q                      # 157 项
python scripts/check_environment.py            # 本机实际生效的后端
python scripts/demo_end_to_end.py              # 端到端 + 跨会话记忆 + 崩溃恢复
python scripts/audit_real_data.py              # 用 RAVDESS 跑一遍审核流水线
python -m emotion_computing.cli profile        # 看阈值到底学到了什么、依据是什么
```

想验证"记忆真的影响决策"，看 `data/reports/<id>.md` 的跨会话表，
以及 `episodes.jsonl` 里 `capacity_timeline[*].memory_hint` —— 非空就说明这一拍是被历史影响的。
