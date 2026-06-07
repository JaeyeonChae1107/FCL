from abc import ABC, abstractmethod
from typing import List
import torch


class BaseSampleSelector(ABC):
    """파이프라인 Stage 2 — 샘플 선택기 추상 기반 클래스 (능동 학습).

    label_budget 내에서 모델 업데이트에 가장 유익한 샘플을 선택한다.
    drift_score를 활용하여 분포 변화 크기에 따라 선택 전략을 조정할 수 있다.

    선택된 샘플은 Stage 3(메모리 업데이트)과 Stage 5(손실 계산)에 사용된다.
    반환 리스트가 비어있으면 CLClient가 range(label_budget)으로 자동 대체한다.

    등록 키: 'random' | 'ssf'
    """

    @abstractmethod
    def select(self, new_data: torch.Tensor,
               new_labels: torch.Tensor,
               label_budget: int,
               drift_score: float = 0.0) -> List[int]:
        """Select indices of samples to include in training.

        Args:
            new_data: Incoming data batch. Shape (N, D).
            new_labels: Corresponding labels. Shape (N,).
            label_budget: Maximum number of labeled samples to select.
            drift_score: Drift intensity score from the drift detector.
                         Higher values may trigger more aggressive selection.

        Returns:
            List of integer indices (into new_data) of selected samples.
            Length is at most label_budget.

        Raises:
            ValueError: If label_budget < 0 or new_data and new_labels have
                        different first dimensions.
        """
