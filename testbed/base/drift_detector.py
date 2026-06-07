from abc import ABC, abstractmethod
from typing import Optional
import torch


class BaseDriftDetector(ABC):
    """Abstract base class for drift detectors."""

    @abstractmethod
    def detect(self, new_data: torch.Tensor,
               memory_buffer: Optional[torch.Tensor]) -> bool:
        """Detect whether distribution drift has occurred.

        Args:
            new_data: Incoming data batch. Shape (N, D).
            memory_buffer: Reference distribution from memory. Shape (M, D) or None.

        Returns:
            True if drift is detected, False otherwise.

        Raises:
            ValueError: If new_data is empty or has wrong shape.
        """

    @abstractmethod
    def get_drift_score(self, new_data: torch.Tensor,
                        memory_buffer: Optional[torch.Tensor]) -> float:
        """Compute a scalar drift intensity score.

        Args:
            new_data: Incoming data batch. Shape (N, D).
            memory_buffer: Reference distribution. Shape (M, D) or None.

        Returns:
            Drift score as a float. Higher values indicate stronger drift.
        """

    def reset(self):
        """Reset internal state. Override if the detector is stateful.

        Returns:
            None
        """
        pass
