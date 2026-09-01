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
    # compute_threshold()가 Best-F(라벨 필수)를 쓴다 — base/anomaly_scorer.py
    # "2026-09-01" 절 참고.
    threshold_needs_labels = True

    def __init__(self, variance_threshold: float = 0.95):
        self.variance_threshold = variance_threshold
        self._pca = None

    def fit(self, normal_data: torch.Tensor) -> None:
        if len(normal_data) == 0:
            return
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
        # GPU 이식성: sklearn 경로를 거치느라 CPU numpy로 왕복했지만, 반환
        # 텐서는 다른 scorer(예: CADEMADScorer.score())와 동일하게 입력
        # data와 같은 device여야 한다는 BaseAnomalyScorer 암묵 계약을 따라야
        # 한다. torch.from_numpy()는 항상 CPU 텐서를 만들어서 이 계약을
        # 어기고 있었다 — 지금까지는 호출부(cl_client.py)가 매번 즉시
        # .cpu()를 부르거나 결과를 버려서 크래시로 이어지지 않았을 뿐이다.
        return torch.from_numpy(err.astype(np.float32)).to(data.device)

    def compute_threshold(self, eval_scores: torch.Tensor,
                           eval_labels: Optional[torch.Tensor]) -> float:
        return best_f_threshold(eval_scores, eval_labels)
