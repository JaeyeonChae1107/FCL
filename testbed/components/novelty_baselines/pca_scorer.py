"""PCAScorer — CND-IDS 자체 novelty scorer (PRD 4절/부록A, 12.6절).

CND-IDS 원 논문 근거: AnomolyDetectors/PCA.py — pca_dim='auto'(기본)면 누적
설명분산 95% 이상을 만족하는 최소 성분 수를 선택하고, score는 원공간에서의
재구성오차 절대값 평균이다(`np.abs(x - reconstructed).mean(axis=1)`, PCA.py:34).
Threshold는 Best-F(PRD 3.5절).
"""

from typing import Optional

import numpy as np
import torch

from testbed.base.anomaly_scorer import BaseAnomalyScorer
from testbed.components.novelty_baselines.thresholding import best_f_threshold


class PCAScorer(BaseAnomalyScorer):
    required_backbone = "autoencoder"

    def __init__(self, variance_threshold: float = 0.95):
        self.variance_threshold = variance_threshold
        self._pca = None

    def fit(self, normal_data: torch.Tensor) -> None:
        from sklearn.decomposition import PCA

        X = normal_data.detach().cpu().numpy()
        n_components = min(self.variance_threshold, X.shape[0], X.shape[1])
        self._pca = PCA(n_components=n_components, svd_solver="full")
        self._pca.fit(X)

    def score(self, data: torch.Tensor) -> torch.Tensor:
        if self._pca is None:
            return torch.zeros(len(data), device=data.device)
        X = data.detach().cpu().numpy()
        recon = self._pca.inverse_transform(self._pca.transform(X))
        err = np.abs(X - recon).mean(axis=1)
        return torch.from_numpy(err.astype(np.float32))

    def compute_threshold(self, eval_scores: torch.Tensor,
                           eval_labels: Optional[torch.Tensor]) -> float:
        return best_f_threshold(eval_scores, eval_labels)
