"""SSF Drift Detector — Kolmogorov-Smirnov two-sample test.

FROM: SSF-Strategic-Selection-and-Forgetting/utils.py::detect_drift()
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from typing import Optional
import torch
import numpy as np
from testbed.base.drift_detector import BaseDriftDetector


def _ks_2samp(a: np.ndarray, b: np.ndarray):
    """Pure-numpy Kolmogorov-Smirnov two-sample test.

    MODIFIED: Replaces scipy.stats.ks_2samp to avoid binary-incompatibility
    between scipy and numpy >= 2.x.  Returns (ks_statistic, p_value).
    p_value is approximated via the Kolmogorov distribution.
    """
    a = np.sort(a.flatten())
    b = np.sort(b.flatten())
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    combined = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, combined, side='right') / n1
    cdf_b = np.searchsorted(b, combined, side='right') / n2
    stat = float(np.max(np.abs(cdf_a - cdf_b)))

    # Kolmogorov distribution approximation for p-value
    en = np.sqrt(n1 * n2 / (n1 + n2))
    z = (en + 0.12 + 0.11 / en) * stat
    # P(K > z) ≈ 2 * sum_{k=1}^{inf} (-1)^{k+1} exp(-2 k^2 z^2)
    p = 0.0
    for k in range(1, 101):
        p += (-1) ** (k + 1) * np.exp(-2 * k * k * z * z)
    p_value = max(0.0, min(1.0, 2.0 * p))
    return stat, p_value


class SSFDriftDetector(BaseDriftDetector):
    """Detects distribution drift via the Kolmogorov-Smirnov 2-sample test.

    FROM: SSF-Strategic-Selection-and-Forgetting/utils.py::detect_drift() (line 646-658)
    """

    def __init__(self, drift_threshold: float = 0.05):
        """
        Args:
            drift_threshold: KS p-value below which drift is declared (default 0.05).
        """
        self.drift_threshold = drift_threshold
        self._last_ks_stat: float = 0.0
        self._last_p_value: float = 1.0

    # FROM: utils.py::detect_drift() — KS 2-sample test logic
    def detect(self, new_data: torch.Tensor,
               memory_buffer: Optional[torch.Tensor]) -> bool:
        """Return True if KS p-value < drift_threshold.

        Args:
            new_data: Incoming 1-D score array or (N,) / (N,1) tensor.
            memory_buffer: Reference score array from memory, same shape convention.

        Returns:
            True if drift is detected.

        Raises:
            ValueError: If memory_buffer is None (reference required).
        """
        if memory_buffer is None:
            return False

        ctrl = self._flatten(memory_buffer)
        treat = self._flatten(new_data)

        ks_stat, p_value = _ks_2samp(ctrl, treat)
        self._last_ks_stat = float(ks_stat)
        self._last_p_value = float(p_value)

        if p_value < self.drift_threshold:
            return True
        return False

    def get_drift_score(self, new_data: torch.Tensor,
                        memory_buffer: Optional[torch.Tensor]) -> float:
        """Return the KS statistic (0–1, higher = more drift).

        Args:
            new_data: Incoming score tensor.
            memory_buffer: Reference score tensor.

        Returns:
            KS statistic as float.
        """
        self.detect(new_data, memory_buffer)
        return self._last_ks_stat

    def reset(self):
        self._last_ks_stat = 0.0
        self._last_p_value = 1.0

    @staticmethod
    def _flatten(t: torch.Tensor):
        if hasattr(t, 'cpu'):
            t = t.cpu()
        if hasattr(t, 'numpy'):
            t = t.numpy()
        return t.flatten()
