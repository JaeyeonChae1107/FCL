"""BaseAntiForgetting — PRD 12.5절."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

from testbed.base.models import BaseCLModel


class BaseAntiForgetting(ABC):
    backbone_type: str  # 'classifier' | 'autoencoder'

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
