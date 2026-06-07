from abc import ABC, abstractmethod
from typing import List
import torch


class BaseSampleSelector(ABC):
    """Abstract base class for sample selectors."""

    @abstractmethod
    def select(self, new_data: torch.Tensor,
               new_labels: torch.Tensor,
               label_budget: int,
               drift_score: float = 0.0) -> List[int]:
        """Select indices of samples to include in training.

        Args:
            new_data: Incoming data batch. Shape (N, D).
            new_labels: Corresponding labels. Shape (N,).
            label_budget: Maximum number of labeled samples to select.
            drift_score: Drift intensity score from the drift detector.
                         Higher values may trigger more aggressive selection.

        Returns:
            List of integer indices (into new_data) of selected samples.
            Length is at most label_budget.

        Raises:
            ValueError: If label_budget < 0 or new_data and new_labels have
                        different first dimensions.
        """
