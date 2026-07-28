"""BaseMemoryManager — PRD 12.4절."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class BaseMemoryManager(ABC):
    @abstractmethod
    def update(self, selected_data: torch.Tensor, selected_labels: torch.Tensor,
               drift_detected: bool) -> None:
        """selected_data/selected_labels는 CLClient step 3c에서 만든, 라벨 예산
        안에서 선택된 서브셋이다 — experience 전체가 아니다."""

    @abstractmethod
    def get_buffer(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        ...

    @abstractmethod
    def get_replay_batch(self, batch_size: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        ...

    @abstractmethod
    def size(self) -> int:
        ...
