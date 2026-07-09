"""DDM Drift Detector — lightweight statistical drift detection.

Inspired by: CND-IDS/AutonomousDCN/ADCNbasic.py::ADCN.driftDetection()

The original ADCN.driftDetection() is deeply coupled to the ADCN neural
architecture. This module implements the same 3-state DDM logic
(STABLE / WARNING / DRIFT) as a standalone, architecture-agnostic detector
that operates on raw feature batches.

FROM: CND-IDS/AutonomousDCN/ADCNbasic.py::ADCN.driftDetection() (line ~547-652)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from typing import Optional
import torch
from testbed.base.drift_detector import BaseDriftDetector


class DDMDriftDetector(BaseDriftDetector):
    """Three-state DDM detector (0=STABLE, 1=WARNING, 2=DRIFT).

    Tracks the running mean and variance of per-batch mean activations.
    When the current batch mean deviates by more than (mean + k*std) it
    advances the state toward DRIFT.

    FROM: CND-IDS/AutonomousDCN/ADCNbasic.py::ADCN.driftDetection()
    """

    STABLE = 0
    WARNING = 1
    DRIFT = 2

    def __init__(self, warning_alpha: float = 2.0,
                 drift_alpha: float = 3.0,
                 min_batches: int = 5):
        """
        Args:
            warning_alpha: z-score threshold for WARNING state (default 2.0).
            drift_alpha: z-score threshold for DRIFT state (default 3.0).
            min_batches: Minimum batches seen before drift can be declared.
        """
        self.warning_alpha = warning_alpha
        self.drift_alpha = drift_alpha
        self.min_batches = min_batches
        self.reset()

    def reset(self):
        self.drift_status: int = self.STABLE
        self._n_batches: int = 0
        self._running_mean: float = 0.0
        self._running_m2: float = 0.0  # Welford M2 accumulator

    def _update_stats(self, value: float):
        """Welford online mean/variance update."""
        self._n_batches += 1
        delta = value - self._running_mean
        self._running_mean += delta / self._n_batches
        delta2 = value - self._running_mean
        self._running_m2 += delta * delta2

    @property
    def _running_std(self) -> float:
        if self._n_batches < 2:
            return 1.0
        return (self._running_m2 / (self._n_batches - 1)) ** 0.5

    # FROM: ADCNbasic.py::ADCN.driftDetection() — batch feature comparison
    def detect(self, new_data: torch.Tensor,
               memory_buffer: Optional[torch.Tensor]) -> bool:
        """Update DDM state and return True when DRIFT (status==2) is confirmed.

        FROM: CND-IDS/AutonomousDCN/ADCNbasic.py::ADCN.driftDetection()

        Args:
            new_data: Current batch features. Shape (N, D) or (N,).
            memory_buffer: Unused (DDM is self-sufficient via running stats).

        Returns:
            True if driftStatus == DRIFT (2).
        """
        batch_mean = new_data.float().mean().item()

        if self._n_batches < self.min_batches:
            self._update_stats(batch_mean)
            self.drift_status = self.STABLE
            return False

        std = self._running_std
        z = (batch_mean - self._running_mean) / max(std, 1e-8)

        if abs(z) > self.drift_alpha:
            self.drift_status = self.DRIFT
            self.reset()            # Reset after confirmed drift
            return True
        elif abs(z) > self.warning_alpha:
            self.drift_status = self.WARNING
        else:
            self.drift_status = self.STABLE
            self._update_stats(batch_mean)

        return False

    def get_drift_score(self, new_data: torch.Tensor,
                        memory_buffer: Optional[torch.Tensor]) -> float:
        """Return the current drift status as a float (0, 1, or 2).

        FROM: ADCNbasic.py — driftStatus values: 0=STABLE, 1=WARNING, 2=DRIFT

        Args:
            new_data: Current batch.
            memory_buffer: Unused.

        Returns:
            Float value of self.drift_status after calling detect().
        """
        self.detect(new_data, memory_buffer)
        return float(self.drift_status)
