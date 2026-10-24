"""审核层评价指标工具集（纯函数）。

覆盖 03-audit-requirements.md 提到的：
  * CCC（Lin's Concordance Correlation）—— 03 §M-1 快系统
  * RMSE / MAE —— 03 §M-1
  * F1 / 准确率 / 召回 —— 03 §M-1 慢系统状态机
  * Krippendorff α（多标注者，标称/区间/序数）—— 03 §D-1
  * Cohen κ（两标注者）—— 03 §D-1
  * Precision@K —— 03 §M-1 记忆层
  * 修复优先级矩阵 —— 03 §I-1

所有函数都接受可迭代输入、纯 numpy 实现；不含 I/O。
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

# --------------------------------------------------------------------------
# 通用回归指标
# --------------------------------------------------------------------------


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """均方根误差。长度不一致抛 ValueError。"""
    if len(y_true) != len(y_pred):
        raise ValueError("rmse: 长度不一致")
    if not y_true:
        return 0.0
    n = len(y_true)
    s = sum((float(a) - float(b)) ** 2 for a, b in zip(y_true, y_pred))
    return math.sqrt(s / n)


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """平均绝对误差。"""
    if len(y_true) != len(y_pred):
        raise ValueError("mae: 长度不一致")
    if not y_true:
        return 0.0
    n = len(y_true)
    return sum(abs(float(a) - float(b)) for a, b in zip(y_true, y_pred)) / n


def ccc(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Lin's Concordance Correlation Coefficient。

    03 §M-1 要求的连续值一致性指标，取值 [-1, 1]，越接近 1 越好。
    特殊情况：
      * 长度 < 2 → 0
      * 两数组完全相同（即便零方差）→ 1.0
      * 一方零方差且不相等 → 0
    """
    if len(y_true) != len(y_pred):
        raise ValueError("ccc: 长度不一致")
    n = len(y_true)
    if n < 2:
        return 0.0
    a = [float(x) for x in y_true]
    b = [float(x) for x in y_pred]
    if a == b:
        return 1.0
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    var_a = sum((x - mean_a) ** 2 for x in a) / n
    var_b = sum((x - mean_b) ** 2 for x in b) / n
    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n)) / n
    denom = var_a + var_b + (mean_a - mean_b) ** 2
    if denom <= 0:
        return 0.0
    return 2.0 * cov / denom


def bias(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """系统偏差 mean(y_pred) - mean(y_true)，正表示系统偏高估。"""
    if len(y_true) != len(y_pred) or not y_true:
        return 0.0
    return sum(float(p) for p in y_pred) / len(y_pred) - sum(float(t) for t in y_true) / len(y_true)


# --------------------------------------------------------------------------
# 分类指标（03 §M-1 慢系统）
# --------------------------------------------------------------------------


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> list[list[int]]:
    """标签集合由 labels 给出，返回 matrix[row=true][col=pred]。"""
    idx = {lab: i for i, lab in enumerate(labels)}
    mat = [[0 for _ in labels] for _ in labels]
    for t, p in zip(y_true, y_pred):
        i = idx.get(t)
        j = idx.get(p)
        if i is None or j is None:
            continue
        mat[i][j] += 1
    return mat


def precision_recall_f1(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> dict[str, dict[str, float]]:
    """逐类 precision / recall / f1（macro 与 micro 由调用方汇总）。"""
    mat = confusion_matrix(y_true, y_pred, labels)
    out: dict[str, dict[str, float]] = {}
    n_labels = len(labels)
    for i, lab in enumerate(labels):
        tp = mat[i][i]
        fp = sum(mat[r][i] for r in range(n_labels) if r != i)
        fn = sum(mat[i][r] for r in range(n_labels) if r != i)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return out


def macro_f1(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> float:
    pr = precision_recall_f1(y_true, y_pred, labels)
    if not pr:
        return 0.0
    return sum(d["f1"] for d in pr.values()) / len(pr)


def miss_rate(
    y_true: Sequence[str], y_pred: Sequence[str], pos_labels: Sequence[str]
) -> float:
    """漏报率 = 1 - recall_on_positive（03 §M-1 失败信号用）。

    positive = {OK} 的反集（即应该报警但没报）。
    """
    if not y_true:
        return 0.0
    pos = set(pos_labels)
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        is_pos_t = t in pos
        is_pos_p = p in pos
        if is_pos_t and is_pos_p:
            tp += 1
        elif not is_pos_t and is_pos_p:
            fp += 1
        elif is_pos_t and not is_pos_p:
            fn += 1
    if (tp + fn) == 0:
        return 0.0
    return 1.0 - tp / (tp + fn)


# --------------------------------------------------------------------------
# 检索指标（03 §M-1 记忆层）
# --------------------------------------------------------------------------


def precision_at_k(
    retrieved_relevant: Sequence[Sequence[int]], k: int
) -> float:
    """`retrieved_relevant[i]` 是第 i 个 query 的 top-K 命中标记（0/1）。

    返回 Precision@K 平均值。
    """
    if k <= 0 or not retrieved_relevant:
        return 0.0
    s = 0.0
    n = 0
    for row in retrieved_relevant:
        top = list(row[:k])
        if not top:
            continue
        s += sum(top) / len(top)
        n += 1
    return s / n if n else 0.0


# --------------------------------------------------------------------------
# 标注一致性（03 §D-1）
# --------------------------------------------------------------------------


def cohens_kappa(
    labels_a: Sequence[str], labels_b: Sequence[str]
) -> float:
    """Cohen's κ（两标注者，标称）。

    解释：≥ 0.81 几乎完美，0.61-0.80 实质一致，0.41-0.60 中等，≤ 0.40 差。
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("cohens_kappa: 长度不一致")
    n = len(labels_a)
    if n == 0:
        return 0.0
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    ca = Counter(labels_a)
    cb = Counter(labels_b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def krippendorff_alpha_nominal(
    reliability: Sequence[Sequence[str | None]],
) -> float:
    """Krippendorff α（标称，03 §D-1 推荐）。

    `reliability[i]` 是第 i 个样本的所有标注者标签（None 表示缺失）。
    """
    # 重新组织为 (unit, coder, value) 形式
    flat: list[tuple[int, int, str]] = []
    n_coders = 0
    for unit, labels in enumerate(reliability):
        for coder, lab in enumerate(labels):
            if lab is None:
                continue
            flat.append((unit, coder, lab))
            n_coders = max(n_coders, coder + 1)
    if n_coders < 2:
        return 0.0
    return _kripp_alpha(flat, level="nominal")


def krippendorff_alpha_interval(
    reliability: Sequence[Sequence[float | None]],
) -> float:
    """Krippendorff α（区间，03 §D-1 连续值用）。"""
    flat: list[tuple[int, int, float]] = []
    for unit, vals in enumerate(reliability):
        for coder, v in enumerate(vals):
            if v is None:
                continue
            flat.append((unit, coder, float(v)))
    if not flat:
        return 0.0
    return _kripp_alpha(flat, level="interval")


def _kripp_alpha(
    flat: Sequence[tuple[int, int, object]], level: str
) -> float:
    """Krippendorff α 通用实现。"""
    if not flat:
        return 0.0
    # 简单实现：计算 observed disagreement 与 expected disagreement
    # 标称：δ = 0/1（不匹配）
    # 区间：δ = (x - y)²
    n = len(flat)
    if n < 2:
        return 0.0

    # 按 unit 分组
    by_unit: dict[int, list[object]] = {}
    for unit, _, val in flat:
        by_unit.setdefault(unit, []).append(val)

    # Do = observed disagreement（每对同一 unit 内不同标注）
    Do_num = 0.0
    Do_den = 0
    for vals in by_unit.values():
        m = len(vals)
        if m < 2:
            continue
        pair_sum = 0.0
        for i in range(m):
            for j in range(i + 1, m):
                pair_sum += _delta(vals[i], vals[j], level)
        Do_num += pair_sum
        Do_den += m * (m - 1) / 2  # C(m,2)

    if Do_den == 0:
        return 0.0
    Do = Do_num / Do_den

    # De = expected disagreement（全标注池随机配对）
    # Krippendorff 公式：De = sum over unordered pairs (c,c') of nc*nc'*δ(c,c') / C(n,2)
    all_vals = [v for _, _, v in flat]
    cnt = Counter(all_vals)
    n_total = len(all_vals)
    if n_total < 2:
        return 0.0
    De_num = 0.0
    De_den = 0
    items = list(cnt.items())
    n_unique = len(items)
    for i in range(n_unique):
        vi, ci = items[i]
        for j in range(i, n_unique):
            vj, cj = items[j]
            if i == j:
                mult = ci * (ci - 1) // 2       # 同一类内 unordered pairs = C(ci,2)
            else:
                mult = ci * cj                  # 异类 unordered pairs
            if mult == 0:
                continue
            d = _delta(vi, vj, level)
            De_num += mult * d
            De_den += mult

    if De_den == 0:
        return 0.0
    De = De_num / De_den
    if De <= 0:
        return 1.0
    return 1.0 - Do / De


def _delta(a: object, b: object, level: str) -> float:
    if level == "nominal":
        return 0.0 if a == b else 1.0
    if level == "interval":
        return (float(a) - float(b)) ** 2
    raise ValueError(f"unknown level: {level}")


# --------------------------------------------------------------------------
# 修复优先级（03 §I-1）
# --------------------------------------------------------------------------


def repair_priority(
    distribution: dict[str, float],
    feasibility: dict[str, str] | None = None,
) -> list[tuple[str, float, str]]:
    """把失败分布 + 修复可行性映射成"高占比+易修复优先"。

    `feasibility[失败类型] ∈ {"easy", "hard"}`，缺省视为 hard。
    返回按优先级从高到低排序的 [(failure_type, percent, feasibility), ...]。
    """
    feas = feasibility or {}
    items: list[tuple[str, float, str]] = []
    for k, pct in distribution.items():
        items.append((k, float(pct), feas.get(k, "hard")))
    # 排序键：easy > hard（同档内按占比降序）
    feas_order = {"easy": 0, "hard": 1}
    items.sort(key=lambda x: (feas_order.get(x[2], 1), -x[1]))
    return items


# --------------------------------------------------------------------------
# 错误样本判定（03 §0 业务问题）
# --------------------------------------------------------------------------


def error_threshold(y_true: Sequence[float], y_pred: Sequence[float], theta: float) -> list[bool]:
    """|y_pred - y_true| > theta 视为错误样本。"""
    if len(y_true) != len(y_pred):
        raise ValueError("error_threshold: 长度不一致")
    return [abs(float(a) - float(b)) > float(theta) for a, b in zip(y_true, y_pred)]


# --------------------------------------------------------------------------
# 跨迭代对比（03 §I-2）
# --------------------------------------------------------------------------


def metric_delta(prev: float, curr: float) -> dict[str, float]:
    """计算 Δ 与相对变化（绝对差 / 相对差）。"""
    diff = curr - prev
    rel = diff / abs(prev) if abs(prev) > 1e-9 else 0.0
    return {"prev": prev, "curr": curr, "delta": diff, "relative": rel}
