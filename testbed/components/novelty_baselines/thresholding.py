"""Best-F thresholding — Track B 공통 (PRD 3.5절, CND-IDS 원 논문 Algorithm 1 step 9).

sklearn.metrics.precision_recall_curve로 도달 가능한 각 threshold 후보에서
F1을 계산해 최댓값을 취하는 threshold를 선택한다. 재구현하지 않는다.
"""

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve


def best_f_threshold(scores: torch.Tensor, labels: torch.Tensor) -> float:
    y = labels.detach().cpu().numpy().astype(int)
    s = scores.detach().cpu().numpy().astype(float)

    if len(np.unique(y)) < 2:
        return float(s.max()) if len(s) else 0.0

    precision, recall, thresholds = precision_recall_curve(y, s)
    if len(thresholds) == 0:
        return float(s.max()) if len(s) else 0.0

    denom = precision + recall
    f1 = np.zeros_like(precision)
    nonzero = denom > 0
    f1[nonzero] = 2 * precision[nonzero] * recall[nonzero] / denom[nonzero]

    # precision/recall은 thresholds보다 원소가 1개 더 많다(마지막 항은 threshold=inf에 대응).
    best_idx = int(np.argmax(f1[:-1]))
    return float(thresholds[best_idx])
