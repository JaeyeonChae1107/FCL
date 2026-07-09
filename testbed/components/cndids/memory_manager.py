"""CND-IDS Memory Manager wrapper.

FROM: CND-IDS/FeatureExtractors/modules/memory.py::Memory (line 1-45)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from typing import Optional, Tuple
import torch
from testbed.base.memory_manager import BaseMemoryManager


class CNDIDSMemoryManager(BaseMemoryManager):
    """FIFO and Perfect memory buffer, wrapping CND-IDS Memory logic.

    FROM: CND-IDS/FeatureExtractors/modules/memory.py::Memory

    Modes:
      'fifo'    — ring buffer; evict oldest on overflow.
      'perfect' — stratified by label class; keeps equal-count exemplars.
    """

    def __init__(self, mode: str = 'fifo', capacity: int = 1000):
        """
        Args:
            mode: 'fifo' or 'perfect' (default 'fifo').
            capacity: Maximum number of samples to store (default 1000).
        """
        self.mode = mode.lower()
        self.capacity = capacity
        self._memory: Optional[torch.Tensor] = None
        self._labels: Optional[torch.Tensor] = None

    # FROM: memory.py::Memory.update() (line 19-40)
    def update(self, selected_data: torch.Tensor,
               selected_labels: torch.Tensor,
               drift_detected: bool = False) -> None:
        """Append new data; enforce capacity according to mode.

        FROM: CND-IDS/FeatureExtractors/modules/memory.py::Memory.update()

        Args:
            selected_data: Samples to add. Shape (N, D).
            selected_labels: Corresponding labels. Shape (N,).
            drift_detected: Ignored for FIFO; Perfect mode always resets on update.

        Returns:
            None
        """
        if self.mode == 'fifo':
            self._fifo_update(selected_data, selected_labels)
        elif self.mode == 'perfect':
            self._perfect_update(selected_data, selected_labels)
        else:
            raise ValueError(f"Unknown memory mode: {self.mode!r}")

    # FROM: memory.py::Memory.update() FIFO branch (line 35-38)
    def _fifo_update(self, new_data: torch.Tensor,
                     new_labels: torch.Tensor) -> None:
        if self._memory is None:
            self._memory = new_data.clone()
            self._labels = new_labels.clone()
        else:
            self._memory = torch.cat([self._memory, new_data], dim=0)
            self._labels = torch.cat([self._labels, new_labels], dim=0)

        if len(self._memory) > self.capacity:
            excess = len(self._memory) - self.capacity
            self._memory = self._memory[excess:]
            self._labels = self._labels[excess:]

    # FROM: memory.py::Memory.update() Perfect branch (line 23-33)
    def _perfect_update(self, new_data: torch.Tensor,
                        new_labels: torch.Tensor) -> None:
        """Store balanced exemplars across unique classes."""
        classes = new_labels.unique()
        n_per_class = max(1, self.capacity // len(classes))
        parts_data, parts_labels = [], []
        for cls in classes:
            mask = new_labels == cls
            d = new_data[mask][:n_per_class]
            l = new_labels[mask][:n_per_class]
            parts_data.append(d)
            parts_labels.append(l)
        self._memory = torch.cat(parts_data, dim=0)
        self._labels = torch.cat(parts_labels, dim=0)

    def get_replay_batch(self, batch_size: int) -> Tuple[Optional[torch.Tensor],
                                                          Optional[torch.Tensor]]:
        """Sample a random replay batch.

        Args:
            batch_size: Desired number of samples.

        Returns:
            (data, labels) or (None, None) if empty.
        """
        if self._memory is None or len(self._memory) == 0:
            return None, None
        n = min(batch_size, len(self._memory))
        idx = torch.randperm(len(self._memory))[:n]
        return self._memory[idx], self._labels[idx]

    def get_buffer(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._memory, self._labels

    def size(self) -> int:
        return 0 if self._memory is None else len(self._memory)
