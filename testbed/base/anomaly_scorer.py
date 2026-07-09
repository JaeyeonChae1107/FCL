from abc import ABC, abstractmethod
import torch


class BaseAnomalyScorer(ABC):
    """파이프라인 Anomaly Scoring 단계 — 이상 탐지 점수 계산기 추상 기반 클래스.

    update() 파이프라인과 분리된 독립 단계.
    fit()은 정상(label=0) 데이터로만 호출되어 정상 분포의 참조를 학습한다.
    이후 score()/predict()로 새 샘플의 이상 정도를 수치화한다.

    임계값(threshold):
    - CLClient.fit_anomaly_scorer()가 정상 데이터의 95th percentile 점수를
      임계값으로 자동 설정 → predict()에서 (score > threshold) → 이상 판정
    - set_anomaly_threshold()로 수동 조정 가능

    등록 키: 'pca' | 'cade_mad' | 'lof' | 'isoforest' | 'deep_svdd'
    """

    @abstractmethod
    def fit(self, normal_data: torch.Tensor) -> None:
        """Learn a reference distribution from normal (inlier) data.

        Args:
            normal_data: Tensor of normal samples. Shape (N, D).

        Returns:
            None

        Raises:
            ValueError: If normal_data is empty.
        """

    @abstractmethod
    def score(self, data: torch.Tensor) -> torch.Tensor:
        """Compute an anomaly score for each sample.

        Args:
            data: Input tensor. Shape (N, D).

        Returns:
            1-D float tensor of shape (N,). Higher values indicate greater
            likelihood of being anomalous.
        """

    def predict(self, data: torch.Tensor,
                threshold: float) -> torch.Tensor:
        """Classify samples as normal (0) or anomalous (1).

        Args:
            data: Input tensor. Shape (N, D).
            threshold: Decision boundary on the anomaly score.

        Returns:
            Long tensor of shape (N,) with values in {0, 1}.
        """
        return (self.score(data) > threshold).long()
