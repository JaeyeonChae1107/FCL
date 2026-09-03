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
        """buf_ref가 None이면 반드시 False 반환 — 단, uses_shared_representation=
        False인 독립 표현 소유자(CADE)는 예외다. 이런 컴포넌트는 buf_ref를
        애초에 쓰지 않고 자기 소유 상태(예: CADE의 centroid)로 판정하므로,
        그 상태가 아직 비어있을 때(=아직 fit()이 한 번도 안 됐을 때)를
        게이트로 쓴다(2026-09-03, components/cade/cade_drift_detector.py
        참고) — buf_ref 유무로 게이트를 걸면 memory_manager='none'일 때
        자기 상태가 이미 학습돼 있어도 항상 False가 되는 오류가 생긴다."""

    @abstractmethod
    def get_drift_score(self, new_data: torch.Tensor, buf_ref: Optional[torch.Tensor]) -> float:
        """buf_ref가 None이면 반드시 0.0 반환 — detect()와 동일한 예외(위 참고)."""

    def fit(self, data: torch.Tensor, labels: torch.Tensor) -> None:
        pass
