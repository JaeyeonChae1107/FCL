"""BaseAnomalyScorer — PRD 12.6절.

재보정(refit)과 threshold 결정 요구사항:
- refit_on_update()는 encoder가 갱신될 때마다(매 experience 종료 시) 호출되어
  scorer의 내부 파라미터(PCA 기저, centroid 등)를 최신 encoder 기준으로
  갱신한다. threshold 결정과는 별개 절차다.
- compute_threshold()는 Track별로 완전히 다른 원 논문 방식을 그대로 구현한다
  (PRD 3.4/3.5절):
    Track B(pca/lof/dif): eval_scores=test 데이터 전체의 score, eval_labels
        필수 (Best-F, CND-IDS Algorithm 1).
    Track A(cade_mad): eval_scores=정상 참조 데이터의 score(s_ref), eval_labels
        불필요 (median + T_MAD*MAD, CADE 원 논문).
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch


class BaseAnomalyScorer(ABC):
    required_backbone: str

    @abstractmethod
    def fit(self, normal_data: torch.Tensor) -> None:
        ...

    @abstractmethod
    def score(self, data: torch.Tensor) -> torch.Tensor:
        ...

    def predict(self, data: torch.Tensor, threshold: float) -> torch.Tensor:
        return (self.score(data) > threshold).long()

    def refit_on_update(self, normal_data: torch.Tensor) -> None:
        """encoder가 갱신될 때마다 재계산. 기본 구현은 fit() 재호출."""
        self.fit(normal_data)

    @abstractmethod
    def compute_threshold(self, eval_scores: torch.Tensor,
                           eval_labels: Optional[torch.Tensor]) -> float:
        ...
