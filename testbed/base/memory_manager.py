"""BaseMemoryManager — PRD 12.4절."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class BaseMemoryManager(ABC):
    # SPIDERMemoryManager 전용(Item 8a) — True면 `cl_client.py` Step 5가
    # 라벨 예산 서브셋(selected_data) 대신 그 라운드 전체(new_data)를
    # update()에 넘긴다(SPIDER 원문의 "이전 태스크 전체에서 무작위 샘플"
    # 요건). 기본값 False를 여기 명시(2026-09-14 — base/anomaly_scorer.py,
    # base/drift_detector.py의 동일 속성과 계약 통일).
    consumes_full_round_data: bool = False

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
