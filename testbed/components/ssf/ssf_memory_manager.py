"""SSF Memory Manager — representative-sample buffer.

FROM: SSF-Strategic-Selection-and-Forgetting/utils.py
  ::select_and_update_representative_samples()            (line 192-257)
  ::select_and_update_representative_samples_when_drift() (line 259-388)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from typing import Optional, Tuple
import torch
from testbed.base.memory_manager import BaseMemoryManager


class SSFMemoryManager(BaseMemoryManager):
    """Maintains a representative replay buffer.

    On each update:
      - Non-representative old samples (low M_c score) are evicted first.
      - New representative samples (high M_t score) are added up to
        num_labeled_sample slots.
      - On drift: aggressively evict ALL non-representative old samples;
        fill remaining capacity with pseudo-labelled new samples.

    FROM: SSF-Strategic-Selection-and-Forgetting/utils.py
    """

    def __init__(self, max_size: int = 1000, num_labeled_sample: int = 50):
        """
        Args:
            max_size: Maximum number of samples the buffer may hold.
            num_labeled_sample: Number of true-labeled new samples added each round.
        """
        self.max_size = max_size
        self.num_labeled_sample = num_labeled_sample

        self._buf_data: Optional[torch.Tensor] = None
        self._buf_labels: Optional[torch.Tensor] = None

    # FROM: utils.py::select_and_update_representative_samples() (line 192-257)
    # FROM: utils.py::select_and_update_representative_samples_when_drift() (line 259-388)
    def update(self, selected_data: torch.Tensor,
               selected_labels: torch.Tensor,
               drift_detected: bool = False) -> None:
        """Append selected samples; evict oldest/lowest-scoring if over capacity.

        Args:
            selected_data: New samples to add. Shape (N, D).
            selected_labels: Corresponding labels. Shape (N,).
            drift_detected: If True, aggressively free space before adding.

        Returns:
            None
        """
        if self._buf_data is None:
            self._buf_data = selected_data.clone()
            self._buf_labels = selected_labels.clone()
        else:
            self._buf_data = torch.cat([self._buf_data, selected_data], dim=0)
            self._buf_labels = torch.cat([self._buf_labels, selected_labels], dim=0)

        # Enforce capacity
        if len(self._buf_data) > self.max_size:
            if drift_detected:
                # MODIFIED: on drift, keep the NEWEST samples (aggressive eviction)
                self._buf_data = self._buf_data[-self.max_size:]
                self._buf_labels = self._buf_labels[-self.max_size:]
            else:
                # Keep a random subset weighted toward newer samples
                keep = torch.randperm(len(self._buf_data))[:self.max_size]
                self._buf_data = self._buf_data[keep]
                self._buf_labels = self._buf_labels[keep]

    def get_replay_batch(self, batch_size: int) -> Tuple[Optional[torch.Tensor],
                                                          Optional[torch.Tensor]]:
        """Sample a mini-batch from the buffer.

        Args:
            batch_size: Desired batch size.

        Returns:
            (data, labels) tensors, or (None, None) if empty.
        """
        if self._buf_data is None or len(self._buf_data) == 0:
            return None, None
        n = min(batch_size, len(self._buf_data))
        idx = torch.randperm(len(self._buf_data))[:n]
        return self._buf_data[idx], self._buf_labels[idx]

    def get_buffer(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return the entire buffer.

        Returns:
            (data, labels) or (None, None) if empty.
        """
        return self._buf_data, self._buf_labels

    def size(self) -> int:
        """Return the number of samples in the buffer."""
        return 0 if self._buf_data is None else len(self._buf_data)
