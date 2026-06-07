from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch


class BaseAntiForgetting(ABC):
    """Abstract base class for anti-catastrophic-forgetting strategies."""

    @abstractmethod
    def compute_loss(self,
                     model: torch.nn.Module,
                     new_batch: Tuple[torch.Tensor, torch.Tensor],
                     replay_batch: Optional[Tuple[torch.Tensor, torch.Tensor]],
                     old_model: Optional[torch.nn.Module] = None) -> torch.Tensor:
        """Compute the total training loss for one step.

        Args:
            model: The current (student) model being trained.
            new_batch: Tuple (data, labels) for the current new mini-batch.
            replay_batch: Tuple (data, labels) sampled from the replay buffer,
                          or None if the buffer is empty.
            old_model: Frozen teacher model snapshot, or None.

        Returns:
            Scalar loss tensor with requires_grad=True.

        Raises:
            ValueError: If new_batch contains tensors of mismatched shapes.
        """

    def on_task_end(self, model: torch.nn.Module) -> None:
        """Hook called after each task / round ends.

        Override to implement teacher-model updates, gradient projection
        memory updates, etc.

        Args:
            model: The current model after the task.

        Returns:
            None
        """
        pass
