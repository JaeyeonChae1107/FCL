"""BaseSampleSelector — PRD 12.3절."""

from abc import ABC, abstractmethod
from typing import List

import torch


class BaseSampleSelector(ABC):
    @abstractmethod
    def select(self, new_data: torch.Tensor, new_labels: torch.Tensor,
               label_budget: int, drift_score: float) -> List[int]:
        """길이 <= label_budget인 인덱스 리스트를 반환한다.

        `label_budget`은 호출부(CLClient, 13절 step 3a)가 9절 labeling_budget의
        mode/value를 이미 정수로 변환해 넘긴 값이다 — 이 메서드 내부에서 비율
        변환을 다시 할 필요는 없다.
        """
