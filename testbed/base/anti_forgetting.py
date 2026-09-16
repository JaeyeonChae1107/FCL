"""BaseAntiForgetting — PRD 12.5절."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

from testbed.base.models import BaseCLModel


class BaseAntiForgetting(ABC):
    backbone_type: str  # 'classifier' | 'autoencoder'
    # GPMAntiForgetting 전용(2026-09-14) — True면 `cl_client.py` Step 3가
    # SVD 기저 계산용 activation 표본을 selected_data(라벨예산 서브셋)
    # 대신 new_data(라운드 전체)로 `set_full_round_data()`에 넘긴다(원
    # 논문에 라벨예산 개념 없음). 기본값 False를 여기 명시 — base/
    # anomaly_scorer.py, base/drift_detector.py, base/memory_manager.py의
    # 동일 속성과 계약 통일(2026-09-14 재검토로 이 파일만 빠졌던 것을
    # 발견·수정).
    consumes_full_round_data: bool = False

    @abstractmethod
    def compute_loss(self, model: BaseCLModel,
                      new_batch: Tuple[torch.Tensor, torch.Tensor],
                      replay_batch: Optional[Tuple[torch.Tensor, torch.Tensor]]
                      ) -> torch.Tensor:
        """new_batch는 (selected_data, selected_labels)다 — experience 전체의
        (X_i, y_i)가 아니라 CLClient step 3에서 sample_selector가 고른, 라벨이
        "공개된" 서브셋만 들어온다. replay_batch는 None일 수 있다(experience 0,
        또는 메모리가 비어있는 경우)."""

    def on_task_end(self, model: BaseCLModel) -> None:
        pass

    def project_gradients(self, model: BaseCLModel) -> None:
        """GPM 전용. 그 외 컴포넌트는 기본 no-op."""
        pass
