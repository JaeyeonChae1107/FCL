"""BaseDriftDetector — PRD 12.2절.

표현 의존성 계약(12.1절): 기본값은 (A) 공유 표현 소비자(uses_shared_representation=
True) — CLClient가 현재 모델로 인코딩한 logit을 넘긴다. CADE만 (B) 독립 표현
소유자(False)이며, 이 경우 CLClient는 원본 data(전처리된 feature, 인코딩 전)를
그대로 넘긴다.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch


class BaseDriftDetector(ABC):
    uses_shared_representation: bool = True

    @abstractmethod
    def detect(self, new_data: torch.Tensor, buf_ref: Optional[torch.Tensor]) -> bool:
        """buf_ref가 None이면 반드시 False 반환."""

    @abstractmethod
    def get_drift_score(self, new_data: torch.Tensor, buf_ref: Optional[torch.Tensor]) -> float:
        """buf_ref가 None이면 반드시 0.0 반환."""

    def fit(self, data: torch.Tensor, labels: torch.Tensor) -> None:
        pass
