"""audit.metrics 单元测试。"""

from __future__ import annotations

import math

import pytest

from emotion_computing.audit import metrics as M


# ---------- CCC ----------
def test_ccc_perfect_agreement():
    y = [0.1, 0.5, 0.9, 0.3, 0.7]
    assert M.ccc(y, y) == pytest.approx(1.0, abs=1e-9)


def test_ccc_inverse_agreement():
    y = [0.1, 0.5, 0.9, 0.3, 0.7]
    inv = [1.0 - v for v in y]
    assert M.ccc(y, inv) == pytest.approx(-1.0, abs=1e-9)


def test_ccc_zero_variance():
    # 全相等但不相同（fused 偏置 0.05）→ 退化情况；返回 0
    assert M.ccc([0.5, 0.5, 0.5], [0.55, 0.55, 0.55]) == pytest.approx(0.0)
    # 全相等且完全相同 → 视为完美一致
    assert M.ccc([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) == pytest.approx(1.0)


def test_ccc_random_high():
    y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    pred = [v + 0.05 for v in y]
    c = M.ccc(y, pred)
    assert 0.95 < c <= 1.0


# ---------- RMSE / MAE ----------
def test_rmse_mae_basic():
    y_true = [0.0, 1.0, 0.0, 1.0]
    y_pred = [0.2, 0.8, 0.2, 0.8]
    assert M.rmse(y_true, y_pred) == pytest.approx(0.2, abs=1e-6)
    assert M.mae(y_true, y_pred) == pytest.approx(0.2, abs=1e-6)


def test_rmse_length_mismatch():
    with pytest.raises(ValueError):
        M.rmse([0.1, 0.2], [0.1])


# ---------- F1 ----------
def test_macro_f1_balanced():
    y_true = ["OK", "WATCH", "PAUSE", "OK", "WATCH", "PAUSE"]
    y_pred = ["OK", "WATCH", "PAUSE", "OK", "WATCH", "PAUSE"]
    assert M.macro_f1(y_true, y_pred, ["OK", "WATCH", "PAUSE"]) == pytest.approx(1.0)


def test_miss_rate():
    y_true = ["WATCH", "PAUSE", "OK"]
    y_pred = ["OK", "OK", "OK"]
    # 两个 positive 都漏报
    assert M.miss_rate(y_true, y_pred, ["WATCH", "PAUSE"]) == pytest.approx(1.0)


# ---------- Krippendorff α ----------
def test_kripp_alpha_nominal_perfect():
    rel = [["A", "A", "A"], ["B", "B", "B"], ["A", "A", "A"]]
    assert M.krippendorff_alpha_nominal(rel) == pytest.approx(1.0)


def test_kripp_alpha_nominal_partial():
    rel = [["A", "A", "A"], ["A", "B", "C"], ["A", "A", "A"]]
    a = M.krippendorff_alpha_nominal(rel)
    assert 0.0 < a < 1.0


def test_kripp_alpha_interval():
    rel = [[0.1, 0.2, 0.15], [0.8, 0.7, 0.75], [0.1, 0.1, 0.1]]
    a = M.krippendorff_alpha_interval(rel)
    assert a > 0.0


def test_kripp_alpha_handles_missing():
    rel = [["A", "A", None], ["A", "B", "C"], [None, "A", "A"]]
    a = M.krippendorff_alpha_nominal(rel)
    assert -1.0 <= a <= 1.0


# ---------- Cohen κ ----------
def test_cohens_kappa_perfect():
    a = ["OK", "OK", "WATCH", "PAUSE"]
    b = ["OK", "OK", "WATCH", "PAUSE"]
    assert M.cohens_kappa(a, b) == pytest.approx(1.0)


def test_cohens_kappa_disagreement():
    a = ["OK", "OK", "WATCH", "PAUSE"]
    b = ["WATCH", "WATCH", "PAUSE", "OK"]
    k = M.cohens_kappa(a, b)
    # 完全错乱
    assert k < 0.0


# ---------- Precision@K ----------
def test_precision_at_k():
    rows = [[1, 1, 0], [1, 0, 0], [0, 0, 0]]
    # P@K = average of (relevant in top-K / K)
    # = (2/3 + 1/3 + 0/3) / 3 = 1/3
    assert M.precision_at_k(rows, 3) == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_precision_at_k_truncates():
    rows = [[1, 1, 1, 0]]  # k=2 → top2=1,1 → 2/2=1
    assert M.precision_at_k(rows, 2) == pytest.approx(1.0)
    rows = [[1, 1, 0, 0]]   # k=4 → (1+1+0+0)/4=0.5
    assert M.precision_at_k(rows, 4) == pytest.approx(0.5)


# ---------- error_threshold ----------
def test_error_threshold():
    yt = [0.1, 0.5, 0.9]
    yp = [0.2, 0.5, 0.7]
    mask = M.error_threshold(yt, yp, theta=0.15)
    # |0.1-0.2|=0.1 < 0.15 ok
    # |0.5-0.5|=0   ok
    # |0.9-0.7|=0.2 > 0.15 err
    assert mask == [False, False, True]


# ---------- repair_priority ----------
def test_repair_priority_easy_first():
    dist = {"modality_sabotage": 30.0, "perception_failure": 30.0,
            "data_problem": 5.0, "combined_failure": 10.0}
    feas = {"modality_sabotage": "easy", "data_problem": "easy",
            "perception_failure": "hard", "combined_failure": "hard"}
    prio = M.repair_priority(dist, feas)
    # 第一个一定是 easy
    assert prio[0][2] == "easy"
    # modality_sabotage (easy, 30) 应排在 data_problem (easy, 5) 之前
    assert prio[0][0] == "modality_sabotage"
    assert prio[1][0] == "data_problem"
    # hard 中应按占比降序
    hard_block = [r for r in prio if r[2] == "hard"]
    assert hard_block[0][0] == "perception_failure"


# ---------- metric_delta ----------
def test_metric_delta_basic():
    d = M.metric_delta(0.5, 0.6)
    assert d["delta"] == pytest.approx(0.1)
    assert d["relative"] == pytest.approx(0.2)


def test_metric_delta_zero_prev():
    d = M.metric_delta(0.0, 0.5)
    assert d["relative"] == 0.0
