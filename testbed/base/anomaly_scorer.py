from abc import ABC, abstractmethod
import torch


class BaseAnomalyScorer(ABC):
    """Abstract base class for anomaly / novelty scorers."""

    @abstractmethod
    def fit(self, normal_data: torch.Tensor) -> None:
        """Learn a reference distribution from normal (inlier) data.

        Args:
            normal_data: Tensor of normal samples. Shape (N, D).

        Returns:
            None

        Raises:
            ValueError: If normal_data is empty.
        """

    @abstractmethod
    def score(self, data: torch.Tensor) -> torch.Tensor:
        """Compute an anomaly score for each sample.

        Args:
            data: Input tensor. Shape (N, D).

        Returns:
            1-D float tensor of shape (N,). Higher values indicate greater
            likelihood of being anomalous.
        """

    def predict(self, data: torch.Tensor,
                threshold: float) -> torch.Tensor:
        """Classify samples as normal (0) or anomalous (1).

        Args:
            data: Input tensor. Shape (N, D).
            threshold: Decision boundary on the anomaly score.

        Returns:
            Long tensor of shape (N,) with values in {0, 1}.
        """
        return (self.score(data) > threshold).long()
