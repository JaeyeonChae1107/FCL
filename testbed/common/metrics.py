"""평가 지표 — PRD 3절.

주지표 4개(F1/Precision/Recall/PR-AUC)만 리더보드 정렬에 쓴다. F1/Precision/
Recall은 TP/FP/FN을 직접 계산하는 방식으로 구현하고(3.1절), PR-AUC만
`sklearn.metrics.average_precision_score`를 그대로 사용한다(재구현 금지).

BWT는 CND-IDS 원 논문 공식 그대로 내부 로그·회귀 테스트용으로 계산한다(3.3절).
"""

from typing import List, Sequence

import numpy as np
from sklearn.metrics import average_precision_score


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, fn


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Precision = TP / (TP + FP). (PRD 3.1절, SSF Table I / CADE Table 3 근거)"""
    tp, fp, _ = _confusion_counts(y_true, y_pred)
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall = TP / (TP + FN). (PRD 3.1절, SSF Table I / CADE Table 3 근거)"""
    tp, _, fn = _confusion_counts(y_true, y_pred)
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 = 2*P*R / (P+R). (PRD 3.1절, SSF Table I / CADE Table 3 근거)"""
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    if p + r == 0.0:
        return 0.0
    return float(2 * p * r / (p + r))


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """PR-AUC = sklearn.metrics.average_precision_score (재구현 금지, PRD 3.1절).

    근거: CND-IDS "Pre-threshold Evaluation" 절, SPIDER Table IV/V/VII.
    """
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_score = np.asarray(y_score).reshape(-1).astype(float)
    if len(np.unique(y_true)) < 2:
        # 한 클래스만 존재하면 average_precision_score가 정의되지 않는다.
        return 0.0
    return float(average_precision_score(y_true, y_score))


def bwt(r_matrix: Sequence[Sequence[float]]) -> float:
    """Backward Transfer — CND-IDS 원 논문 공식 그대로 (PRD 3.3절).

    BwdTrans = sum_{i=1}^{m} (R_{mi} - R_{ii}) / (m(m-1)/2)

    r_matrix는 (T, T) 크기의 F1 성능 행렬 (PRD 3.4절 정의). 1-indexed 논문
    공식의 R_{mi}는 마지막 행(0-indexed 로 T-1행)에 대응한다.

    Args:
        r_matrix: T x T performance matrix (R_ij = experience i까지 학습한
                  모델을 experience j에서 평가한 F1).

    Returns:
        BWT 스칼라 값. T < 2 이면 정의되지 않으므로 0.0을 반환한다.
    """
    R = np.asarray(r_matrix, dtype=float)
    T = R.shape[0]
    if T < 2:
        return 0.0
    m = T - 1  # 0-indexed 마지막 행
    total = 0.0
    for i in range(T):
        total += R[m, i] - R[i, i]
    denom = T * (T - 1) / 2
    return float(total / denom)


def build_r_matrix(f1_grid: List[List[float]]) -> np.ndarray:
    """T x T 성능 행렬을 numpy 배열로 정규화한다 (PRD 3.4절 R 정의).

    Args:
        f1_grid: 리스트의 리스트. f1_grid[i][j] = experience i까지 학습한
                 모델을 experience j의 test split에서 평가한 F1.

    Returns:
        (T, T) numpy 배열.

    Raises:
        ValueError: 정사각 행렬이 아닌 경우.
    """
    R = np.asarray(f1_grid, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"R matrix must be square (T, T), got shape {R.shape}")
    return R
