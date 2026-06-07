"""Evaluation metrics for continual anomaly detection experiments.

지표 선택 근거 (IDS 문맥):
  F1               : 공격/정상 클래스 불균형 시 accuracy보다 균형 잡힌 탐지 성능 요약
  Precision        : 알람 정확도 — 낮으면 SOC 알람 피로도(alert fatigue) 유발
  Detection Rate   : 공격 탐지율(Recall/TPR) — 낮으면 실제 침입을 놓침
  False Alarm Rate : 오탐률(FPR) — 높으면 정상 트래픽 차단 오류 발생
  Balanced Accuracy: (TPR+TNR)/2 — 극단적 클래스 불균형에서도 의미 있는 정확도
  BWT              : 연속 학습에서 이전 공격 탐지 능력 보존 여부 (망각 방지 핵심 지표)
  FWT              : 이전 지식이 새 공격 유형 탐지에 얼마나 도움이 되는지
  Label Efficiency : 레이블링(보안 전문가 분석) 비용 절감률
  Avg Inference ms : 실시간 IDS 요구사항 — 탐지 지연 최소화
"""

import time
from typing import List
import numpy as np


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary F1 score (pure numpy, no sklearn dependency).

    IDS 문맥: Precision과 Recall의 조화평균. 공격 트래픽이 전체의 수%에 불과한
    불균형 데이터에서 단순 accuracy(항상 정상 예측으로도 99%+)보다 신뢰할 수 있는
    단일 지표. F1이 높으면 오탐과 미탐이 모두 적음을 의미한다.

    수식: 2 * TP / (2*TP + FP + FN)

    Args:
        y_true: Ground-truth labels (0/1). Shape (N,).
        y_pred: Predicted labels (0/1). Shape (N,).

    Returns:
        F1 score in [0, 1].
    """
    y_true = np.asarray(y_true).flatten().astype(int)
    y_pred = np.asarray(y_pred).flatten().astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Precision — 양성 예측 중 실제 양성 비율 (알람 정확도).

    IDS 문맥: 시스템이 '공격'이라 경보를 발령했을 때 실제로 공격인 비율.
    Precision이 낮으면 오탐(FP) 알람이 많아 보안 운영자의 알람 피로도
    (alert fatigue)가 증가하고, 실제 위협을 무시하게 될 위험이 높아진다.

    수식: TP / (TP + FP)

    Args:
        y_true: Ground-truth labels (0=normal, 1=attack). Shape (N,).
        y_pred: Predicted labels (0/1). Shape (N,).

    Returns:
        Precision in [0, 1]. 0.0 if no positive predictions.
    """
    y_true = np.asarray(y_true).flatten().astype(int)
    y_pred = np.asarray(y_pred).flatten().astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced Accuracy — 클래스 불균형을 보정한 정확도.

    IDS 문맥: NSL-KDD/UNSW-NB15는 정상 트래픽 비율이 압도적으로 높아 항상
    정상으로 예측해도 일반 accuracy가 99%+가 될 수 있다. Balanced Accuracy는
    (TPR + TNR) / 2로 클래스 불균형 편향을 제거한다. 두 클래스 모두에서
    균형 잡힌 성능을 요구하는 IDS 평가에 적합하다.

    수식: (TPR + TNR) / 2  where TNR = TN / (TN + FP)

    Args:
        y_true: Ground-truth labels (0/1). Shape (N,).
        y_pred: Predicted labels (0/1). Shape (N,).

    Returns:
        Balanced accuracy in [0, 1].
    """
    y_true = np.asarray(y_true).flatten().astype(int)
    y_pred = np.asarray(y_pred).flatten().astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return (tpr + tnr) / 2.0


def detection_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True Positive Rate (Recall / Sensitivity) — 공격 탐지율.

    IDS 문맥: 실제 공격 중 탐지한 비율. 보안 관점에서 가장 중요한 지표.
    Detection Rate(= Recall)가 낮으면 실제 침입을 놓치는 미탐지(FN)가 많아
    보안 사고로 이어진다. IDS 평가에서 흔히 'DR'로 약칭한다.

    수식: TP / (TP + FN)

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        TPR in [0, 1].
    """
    y_true = np.asarray(y_true).flatten().astype(int)
    y_pred = np.asarray(y_pred).flatten().astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def false_alarm_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """False Positive Rate (FPR) — 오탐률.

    IDS 문맥: 정상 트래픽을 공격으로 잘못 분류하는 비율.
    FPR이 높으면 ① 정상 서비스 차단 오류 발생, ② 보안 운영자의 알람 피로도 증가,
    ③ 실제 위협 알람이 오탐에 묻혀 무시될 위험. Precision의 보완 지표이며
    ROC 곡선의 x축에 해당한다.

    수식: FP / (FP + TN)

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        FPR in [0, 1].
    """
    y_true = np.asarray(y_true).flatten().astype(int)
    y_pred = np.asarray(y_pred).flatten().astype(int)
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0


def backward_transfer(perf_matrix: List[List[float]]) -> float:
    """Backward Transfer (BWT) — 연속 학습에서 이전 태스크 성능 변화 측정.

    IDS 문맥: 새로운 공격 유형을 학습한 후 이전에 알고 있던 공격 탐지 능력이
    얼마나 유지되는지 정량화한다. 연속 학습 IDS의 핵심 안전성 지표.
    - BWT < 0 : 이전 공격 유형 탐지 성능 저하 (catastrophic forgetting)
    - BWT ≈ 0 : 이전 성능 유지 (이상적)
    - BWT > 0 : 새 학습이 오히려 이전 태스크도 개선 (backward plasticity)

    수식: BWT = (1 / (T-1)) * Σ_{j=1}^{T-1} (R[T][j] - R[j][j])
    R[i][j] = i번째 태스크까지 학습 후 j번째 태스크에서의 F1 점수

    Args:
        perf_matrix: perf_matrix[i][j] = performance on task j after training
                     on task i. 0-indexed.

    Returns:
        BWT scalar. Negative = forgetting, 0 = no forgetting, positive = transfer.
    """
    T = len(perf_matrix)
    if T <= 1:
        return 0.0
    bwt = 0.0
    for j in range(T - 1):
        bwt += perf_matrix[T - 1][j] - perf_matrix[j][j]
    return bwt / (T - 1)


def forward_transfer(perf_matrix: List[List[float]],
                     random_baseline: float = 0.5) -> float:
    """Forward Transfer (FWT) — 이전 지식이 새 태스크에 얼마나 전이되는지 측정.

    IDS 문맥: 새로운 공격 유형을 처음 마주쳤을 때 이전에 학습한 공격 패턴 지식이
    얼마나 도움이 되는지 측정한다. FWT > 0이면 이전 경험이 새 공격 탐지에
    유용함을 의미한다 (제로샷 일반화 능력).

    수식: FWT = (1 / (T-1)) * Σ_{i=1}^{T-1} (R[i-1][i] - R_random)
    R[i-1][i] = i-1번째 태스크까지만 학습한 상태에서 i번째 태스크 F1

    Args:
        perf_matrix: Same as backward_transfer.
        random_baseline: Performance of a random classifier (default 0.5).

    Returns:
        FWT scalar.
    """
    T = len(perf_matrix)
    if T <= 1:
        return 0.0
    fwt = 0.0
    for i in range(1, T):
        fwt += perf_matrix[i - 1][i] - random_baseline
    return fwt / (T - 1)


def label_efficiency(total_samples: int, labeled_samples: int) -> float:
    """Fraction of samples that did NOT require manual labels.

    IDS 문맥: 실제 환경에서 레이블링은 보안 전문가의 수작업 분석을 요구한다.
    Label Efficiency가 높을수록 전문가 분석 비용이 절감된다.
    e.g. 0.99 → 전체 트래픽의 1%만 수동 레이블링으로 탐지 모델 유지 가능.

    수식: 1.0 - (labeled_samples / total_samples)

    Args:
        total_samples: Total samples encountered.
        labeled_samples: Samples that were labeled (used label budget).

    Returns:
        Efficiency in [0, 1]. Higher = fewer labels used.
    """
    if total_samples == 0:
        return 1.0
    return 1.0 - labeled_samples / total_samples


def avg_inference_time_ms(model_fn, data, n_runs: int = 100) -> float:
    """Average per-sample inference latency in milliseconds.

    IDS 문맥: 실시간 네트워크 침입 탐지는 트래픽 처리 속도를 따라가야 한다.
    탐지 지연이 너무 길면 공격이 탐지 이전에 피해를 입힌다.
    일반적으로 실시간 IDS는 패킷당 1ms 이내 처리가 요구된다.

    Args:
        model_fn: Callable that takes data and returns predictions.
        data: Input tensor or array.
        n_runs: Number of timing runs (default 100).

    Returns:
        Mean inference time per call in ms.
    """
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model_fn(data)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times) * 1000)
