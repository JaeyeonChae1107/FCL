"""Phase 1 — toy 예시로 지표 계산 로직을 컴포넌트 구현 전에 먼저 검증한다."""

import numpy as np
import pytest

from testbed.common.metrics import (
    f1_score,
    precision_score,
    recall_score,
    pr_auc,
    bwt,
    build_r_matrix,
)


def test_precision_recall_f1_hand_computed():
    # TP=2(idx2,3), FP=1(idx1), FN=0
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    assert precision_score(y_true, y_pred) == pytest.approx(2 / 3)
    assert recall_score(y_true, y_pred) == pytest.approx(1.0)
    assert f1_score(y_true, y_pred) == pytest.approx(0.8)


def test_precision_zero_when_no_positive_predictions():
    y_true = [0, 1, 1]
    y_pred = [0, 0, 0]
    assert precision_score(y_true, y_pred) == 0.0
    assert recall_score(y_true, y_pred) == 0.0
    assert f1_score(y_true, y_pred) == 0.0


def test_f1_perfect_prediction():
    y_true = [0, 1, 0, 1, 1]
    y_pred = [0, 1, 0, 1, 1]
    assert f1_score(y_true, y_pred) == pytest.approx(1.0)


def test_pr_auc_perfect_separation():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    assert pr_auc(y_true, y_score) == pytest.approx(1.0)


def test_pr_auc_single_class_returns_zero():
    y_true = [0, 0, 0]
    y_score = [0.1, 0.5, 0.9]
    assert pr_auc(y_true, y_score) == 0.0


def test_bwt_hand_computed():
    # T=3, 마지막 태스크(index 2) 제외. diagonal[0:2] = [0.9, 0.85],
    # last row[0:2] = [0.6, 0.55]
    # sum = (0.6-0.9) + (0.55-0.85) = -0.6 ; denom = T-1 = 2 ; bwt = -0.3
    # (CND-IDS 원본 AutonomousDCN/ADCNmainloop.py:418 공식)
    R = [
        [0.9, 0.0, 0.0],
        [0.0, 0.85, 0.0],
        [0.6, 0.55, 0.8],
    ]
    assert bwt(R) == pytest.approx(-0.3)


def test_bwt_zero_forgetting_when_no_degradation():
    # 마지막 행이 대각과 동일하면 BWT=0 (망각 없음)
    R = [
        [0.9, 0.0, 0.0],
        [0.0, 0.85, 0.0],
        [0.9, 0.85, 0.8],
    ]
    assert bwt(R) == pytest.approx(0.0)


def test_bwt_undefined_for_single_experience():
    R = [[0.9]]
    assert bwt(R) == 0.0


def test_build_r_matrix_rejects_non_square():
    with pytest.raises(ValueError):
        build_r_matrix([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])


def test_build_r_matrix_accepts_square():
    R = build_r_matrix([[0.1, 0.2], [0.3, 0.4]])
    assert R.shape == (2, 2)


def test_per_category_counts_and_recall():
    from testbed.common.metrics import per_category_counts, per_category_recall, per_category_fpr

    # 3행: DoS 공격 탐지 성공, Probe 공격 놓침, normal 오탐 1건
    y_true = [1, 1, 0]
    y_pred = [1, 0, 1]
    category = ["DoS", "Probe", "normal"]
    counts = per_category_counts(y_true, y_pred, category)
    assert counts["DoS"] == {"n": 1, "n_attack": 1, "tp": 1, "fp": 0}
    assert counts["Probe"] == {"n": 1, "n_attack": 1, "tp": 0, "fp": 0}
    assert counts["normal"] == {"n": 1, "n_attack": 0, "tp": 0, "fp": 1}

    recall = per_category_recall(counts)
    assert recall == {"DoS": pytest.approx(1.0), "Probe": pytest.approx(0.0)}
    assert "normal" not in recall  # 공격 행이 없는 category는 recall 대상 아님

    fpr = per_category_fpr(counts)
    assert fpr == {"normal": pytest.approx(1.0)}
    assert "DoS" not in fpr  # 정상 행이 없는 category는 FPR 대상 아님


def test_per_category_recall_zero_when_no_true_positives():
    from testbed.common.metrics import per_category_counts, per_category_recall

    y_true = [1, 1]
    y_pred = [0, 0]
    category = ["U2R", "U2R"]
    counts = per_category_counts(y_true, y_pred, category)
    assert per_category_recall(counts) == {"U2R": 0.0}


def test_per_category_counts_length_mismatch_raises():
    from testbed.common.metrics import per_category_counts

    with pytest.raises(ValueError):
        per_category_counts([1, 0], [1], ["a", "b"])
