"""Evaluation metrics for continual anomaly detection experiments."""

import time
from typing import List
import numpy as np


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary F1 score (pure numpy, no sklearn dependency).

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


def detection_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True Positive Rate (Recall / Sensitivity).

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
    """False Positive Rate.

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
    """Backward Transfer (BWT) — measures forgetting.

    BWT = (1 / (T-1)) * Σ_{j=1}^{T-1} (R[T][j] - R[j][j])

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
    """Forward Transfer (FWT) — measures knowledge transfer to new tasks.

    FWT = (1 / (T-1)) * Σ_{i=1}^{T-1} (R[i-1][i] - R_random[i])

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
    """Fraction of samples that required manual labels.

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
