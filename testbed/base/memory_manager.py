from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch


class BaseMemoryManager(ABC):
    """파이프라인 Stage 3 & 4 — Replay 버퍼 관리자 추상 기반 클래스.

    Stage 3: update()로 선택 샘플을 버퍼에 추가 (drift 여부에 따라 교체 전략 조정 가능).
    Stage 4: get_replay_batch()로 Stage 5(anti-forgetting)에 과거 경험 제공.
             get_buffer()로 다음 라운드 Drift Detection의 참조 분포를 제공.

    구현 시 보장사항:
    - update() 이후 get_replay_batch()는 유효한 데이터를 반환할 수 있음
    - 버퍼가 비어있으면 (None, None) 반환 (CLClient가 replay_batch=None으로 처리)

    등록 키: 'none' | 'fixed' | 'ssf' | 'cndids'
    """

    @abstractmethod
    def update(self, selected_data: torch.Tensor,
               selected_labels: torch.Tensor,
               drift_detected: bool) -> None:
        """Update the replay buffer with newly selected samples.

        Args:
            selected_data: Data tensor to add. Shape (N, D).
            selected_labels: Label tensor to add. Shape (N,).
            drift_detected: Whether drift was detected this round.
                            Some managers (e.g. SSF) evict differently on drift.

        Returns:
            None

        Raises:
            ValueError: If selected_data and selected_labels have different
                        first dimensions.
        """

    @abstractmethod
    def get_replay_batch(self, batch_size: int) -> Tuple[Optional[torch.Tensor],
                                                         Optional[torch.Tensor]]:
        """Sample a replay batch from the buffer.

        Args:
            batch_size: Number of samples to draw.

        Returns:
            Tuple (data, labels) each of shape (min(batch_size, size), *).
            Returns (None, None) if the buffer is empty.
        """

    @abstractmethod
    def get_buffer(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return the entire buffer contents.

        Returns:
            Tuple (data, labels). Returns (None, None) if empty.
        """

    @abstractmethod
    def size(self) -> int:
        """Return the number of samples currently stored in the buffer.

        Returns:
            Integer >= 0.
        """
