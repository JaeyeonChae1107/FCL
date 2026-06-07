from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch


class BaseMemoryManager(ABC):
    """Abstract base class for memory (replay buffer) managers."""

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
