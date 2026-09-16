"""PCAScorer — CND-IDS 자체 novelty scorer (PRD 4절/부록A, 12.6절).

CND-IDS 원 논문 근거: AnomolyDetectors/PCA.py — pca_dim='auto'(기본)면 누적
설명분산 95% 이상을 만족하는 최소 성분 수를 선택하고, score는 원공간에서의
재구성오차 절대값 평균이다(`np.abs(x - reconstructed).mean(axis=1)`, PCA.py:34).
Threshold는 Best-F(PRD 3.5절).

**구현 단순화(2026-09-14 재검토로 명시, 수학적으로 동등)**: 원문
`PCA.py:12-24`는 "전체 성분으로 PCA 적합 → 누적분산 95% 지점의 성분 수를
직접 계산 → 그 수로 재적합"을 수동 2단계로 한다. 이 구현은 sklearn의
`PCA(n_components=0.95, svd_solver="full")`(0<n_components<1이면 sklearn이
내부적으로 동일한 누적분산 기준 성분 수를 자동 계산)로 한 번에 처리한다 —
결과는 동일하고 sklearn 표준 기능을 쓴 것뿐이지만, 원문과 코드 구조가
1:1 대응하지 않는다는 점을 명시해둔다.
"""

from typing import Optional

import numpy as np
import torch

from testbed.base.anomaly_scorer import BaseAnomalyScorer
from testbed.components.novelty_baselines.thresholding import best_f_threshold


class PCAScorer(BaseAnomalyScorer):
    required_backbone = "autoencoder"
    # compute_threshold()가 Best-F(라벨 필수)를 쓴다 — base/anomaly_scorer.py 참고.
    threshold_needs_labels = True
    # Item 5 — CND-IDS 원문의 PCA fit()이 실제로 보는 참조는 스트림 시작
    # 전 고정된 `init_normal`이다(이번 라운드 라벨-예산 정상 서브셋이
    # 아니다). `CLClient` Step 6가 이 속성을 보고 `held_out_normal_
    # reference`가 있으면 그걸로 refit_on_update()를 호출한다.
    consumes_held_out_reference = True

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
