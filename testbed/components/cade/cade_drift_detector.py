"""CADE Drift Detector — MAD-based anomaly scoring in latent space.

PORTED FROM: CADE/cade/detect.py::detect_drift_samples()       (line 45-105)
             CADE/cade/detect.py::get_MAD_for_each_family()    (line 150-160)
             CADE/cade/detect.py::get_latent_distance_between_sample_and_centroid() (line 137-147)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from typing import Optional, Dict
import torch
from testbed.base.drift_detector import BaseDriftDetector


class CADEDriftDetector(BaseDriftDetector):
    """Detects drift via MAD-normalised distance in encoder latent space.

    PORTED FROM: CADE/cade/detect.py::detect_drift_samples() (line 45-105)

    Usage::
        detector = CADEDriftDetector(threshold=3.5)
        detector.fit(z_train, y_train)          # compute centroids + MAD
        is_drift = detector.detect(z_new, None) # z_new = encoder(x_new)
    """

    def __init__(self, threshold: float = 3.5):
        """
        Args:
            threshold: MAD-normalised anomaly score above which drift is declared.
                       Default 3.5 (≈ 3.5 σ assuming Gaussian, from original CADE).
        """
        self.threshold = threshold
        self._centroids: Dict[int, torch.Tensor] = {}
        self._medians: Dict[int, float] = {}
        self._mads: Dict[int, float] = {}
        self._fitted = False

    # PORTED FROM: detect.py::get_latent_data_for_each_family() (line 123-134)
    # PORTED FROM: detect.py::get_latent_distance_between_sample_and_centroid() (line 137-147)
    # PORTED FROM: detect.py::get_MAD_for_each_family() (line 150-160)
    def fit(self, z_train: torch.Tensor, y_train: torch.Tensor) -> None:
        """Compute per-family centroids and MAD from training latent vectors.

        PORTED FROM: CADE/cade/detect.py (line 55-79)
          - centroids[i] = mean(z_family[i])
          - dis[i][j] = ||z - centroid[i]||_2
          - MAD[i] = 1.4826 * median(|dis - median(dis)|)

        Args:
            z_train: Encoded training samples. Shape (N, latent_dim).
            y_train: Integer family labels. Shape (N,).
        """
        families = y_train.unique().tolist()
        for f in families:
            f = int(f)
            mask = (y_train == f)
            z_f = z_train[mask]
            centroid = z_f.mean(dim=0)
            dis = torch.norm(z_f - centroid.unsqueeze(0), dim=1)  # (M,)
            median_dis = dis.median().item()
            mad = 1.4826 * (dis - median_dis).abs().median().item()
            if mad < 1e-8:
                mad = 1e-8  # prevent division by zero

            self._centroids[f] = centroid
            self._medians[f] = median_dis
            self._mads[f] = mad

        self._fitted = True

    # PORTED FROM: detect.py::detect_drift_samples() (line 87-104)
    def _anomaly_score(self, z: torch.Tensor) -> float:
        """Compute the minimum MAD-normalised distance to any known family.

        PORTED FROM: CADE/cade/detect.py (line 89-97):
          anomaly_k[i] = |dis_k[i] - median(dis[i])| / MAD[i]
          min_anomaly_score = min(anomaly_k)

        Args:
            z: Single latent vector. Shape (latent_dim,).

        Returns:
            Float anomaly score (higher = more anomalous).
        """
        if not self._fitted or not self._centroids:
            return 0.0
        scores = []
        for f, centroid in self._centroids.items():
            dist = torch.norm(z - centroid).item()
            score = abs(dist - self._medians[f]) / self._mads[f]
            scores.append(score)
        return min(scores)

    def detect(self, new_data: torch.Tensor,
               memory_buffer: Optional[torch.Tensor]) -> bool:
        """Return True if any new sample exceeds the anomaly threshold.

        Args:
            new_data: Latent vectors of new samples. Shape (N, D).
            memory_buffer: Unused (CADE uses fitted statistics).

        Returns:
            True if drift detected.
        """
        if not self._fitted:
            return False
        scores = self._batch_scores(new_data)
        return bool((scores > self.threshold).any())

    def get_drift_score(self, new_data: torch.Tensor,
                        memory_buffer: Optional[torch.Tensor]) -> float:
        """Return the minimum anomaly score across the batch.

        Args:
            new_data: Latent vectors. Shape (N, D).
            memory_buffer: Unused.

        Returns:
            Float — min anomaly score (conservative: flag if ANY sample drifts).
        """
        if not self._fitted:
            return 0.0
        return float(self._batch_scores(new_data).max().item())

    def _batch_scores(self, z_batch: torch.Tensor) -> torch.Tensor:
        """Compute per-sample anomaly scores for a batch."""
        scores = torch.zeros(len(z_batch))
        for i, z in enumerate(z_batch):
            scores[i] = self._anomaly_score(z)
        return scores

    def reset(self):
        self._centroids = {}
        self._medians = {}
        self._mads = {}
        self._fitted = False
