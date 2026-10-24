# 真实音频跑批样例（RAVDESS）

这里放的是**用真实音频驱动整条管线**跑出来的一次会话复盘，不是合成信号。

| 项 | 值 |
|---|---|
| 数据源 | RAVDESS（Ryerson Audio-Visual Database of Emotional Speech），只取 `Actor_01..03` 的语音片段 |
| 驱动方式 | `python scripts/validate_real_data.py`，把真实 wav 逐帧喂进 `SessionPipeline` |
| **实际生效的感知后端** | `face=yunet`（OpenCV DNN 真实推理）、`speech=librosa`（真实声学特征）、`text=lexicon` |
| 这次跑批规模 | **26 个会话**，`task_label=ravdess_real` |
| 紧张度范围 | 均值 0.059 ~ 0.276，峰值 0.115 ~ 0.683 |
| 本文件这条 | `ef18721606fb`：91 个标定后样本、峰值 0.683、触发 2 次干预（这条是 26 条里信号最丰富的一条） |

## 这条报告能说明什么

- **真实音频链路是通的**：曲线不是合成驱动画出来的，语音贡献占比在 0.37~0.38 之间真实参与融合；
- **自下而上的归因可用**：峰值归因表里能看到同一段话里 `face`/`speech` 谁在主导（0.419 vs 0.383）；
- **行动层真的会出手**：`干预次数: 2`，说明慢系统进了 WATCH/PAUSE 并给出了干预；
- **跨会话记忆在跑**：第 5 节对比了此前的 3 条同类会话。

## 这条报告**不能**说明什么（别拿它当效果证据）

- RAVDESS 是**演员按剧本表演的语音**，不是真实高压任务，更不是真人受试者数据；
- 因此它只能证明"真实信号 → 特征 → 融合 → 状态机 → 干预 → 复盘"这条链子在真实输入上不崩，
  **不能**证明紧张度测得准，也不能用来算准确率；
- `validate_real_data.py` 里还有 4 条断言（情绪梯度、模态贡献上升、状态机介入、记忆复盘），
  那些是脚本内断言，跑不通会直接中断；但当时没有保留 stdout，所以这里只把**能复核的产物**当证据。

## 时间戳说明

README 顶部那张曲线图就是从这条会话（`ef18721606fb`）导出的，复现命令：

```bash
python scripts/export_demo_chart.py \
  --episodes data/real_data_runs/episodes.jsonl \
  --episode-id ef18721606fb
```

这批产物生成于 **2026-10-10**，早于记忆层那轮修复（阈值自适应 / SQLite 索引 / 崩溃检查点，
见 [`docs/05-implementation-status.md`](../../docs/05-implementation-status.md)）。
报告格式不受影响 —— 阈值学习只改变"下次会话用哪个阈值"，不改变报告长什么样；
所以图里画的是那次会话实际使用的阈值（默认 0.55 / 0.75），不是学出来的值。

## 想自己复现

```bash
# 需要先按 scripts/download_models.py 拿到模型，并自备 RAVDESS 音频（CC BY-NC-SA，勿入库）
python scripts/validate_real_data.py
```

产物会落在 `data/real_data_runs/`（已 gitignore）。
