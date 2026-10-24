# examples/

这里放**真实的产物样例**，用来展示系统实际输出长什么样——不需要先跑一遍就能看懂。

两个样例回答的是不同的问题：

| 目录 | 是什么 | 先看哪个文件 |
|---|---|---|
| [`real-data-run/`](real-data-run/) | **真实音频驱动**跑出的一次会话复盘（RAVDESS，`face=yunet`/`speech=librosa` 真跑） | [`report.md`](real-data-run/report.md) —— 曲线、峰值归因、干预次数、跨会话对比 |
| [`ravdess-audit/`](ravdess-audit/) | 对上面这类跑批做**审核**：哪一层出错、错在哪、先修什么 | [`audit_report.md`](ravdess-audit/audit_report.md) + [`dashboard.html`](ravdess-audit/dashboard.html) |

## `ravdess-audit/` — 一次真实数据审核会话的输出

来源：`python scripts/audit_real_data.py`
（用 RAVDESS 公开数据集驱动 MVA，再对其做审核）

| 文件 | 说明 |
|---|---|
| `audit_report.md` | **先看这个**——人工可读的总报告，含数据层审核、模块指标、消融、失败分布、修复优先级、错误样本清单 |
| `dashboard.html` | 单文件可视化仪表盘，**双击用浏览器打开**即可（无外部依赖）：失败分布饼图 / 修复优先级柱图 / 模块指标雷达 / 失败样本表 |
| `failure_distribution.json` | F-3 失败类型分布 + 推荐修复方向（机器可读） |
| `data_qa.json` | D-# 数据层审核结果（标注一致性、抽检覆盖率、类别/强度/模态分布） |

### 这次会话的结果概览

| 指标 | 数值 |
|---|---:|
| 样本数 | 200 |
| 错误样本 | 87（43.5%） |
| `modality_sabotage` | 6（6.9%） |
| `combined_failure` | 81（93.1%） |
| 优先修复方向 | `modality_sabotage` → 修改融合策略，增加置信度校准 |

> 说明：为验证归因链路，这次跑批在 MVA 输出上**注入了已知类型的错误**
> （见 `scripts/audit_real_data.py` 的 `episode_to_outputs`）。
> 因此失败分布不代表模型真实性能，只用于演示"每个错误都能被定位到具体模块"。

### 完整产物清单

一次正式审核会在 `data/audit/<dataset>/<audit_id>/` 下落盘 9 个文件：

```
audit_report.md              # 人工可读总报告
audit_report.json            # 机器可读副本
dashboard.html               # 单文件可视化
data_qa.json                 # D-# 数据层审核
module_evaluations.json      # M-1 各模块指标
ablation_results.json        # M-2 消融实验
modality_attributions.json   # M-3 每样本模态归因
failure_cases.jsonl          # F-1 每个错误样本的归因（一行一条）
failure_distribution.json    # F-3 失败分布 + 修复优先级
```

外加跨迭代 `data/audit/_iterations.jsonl` 与审核日志 `data/audit/_audit_log.jsonl`。
