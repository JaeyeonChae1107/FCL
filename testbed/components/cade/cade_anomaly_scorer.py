"""CADE Anomaly Scorer — single-class MAD distance scorer.

PORTED FROM: CADE/cade/detect.py::detect_drift_samples() (line 45-105)
(Single normal-class variant: treats all training data as one 'normal' family.)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import torch
from testbed.base.anomaly_scorer import BaseAnomalyScorer


class CADEAnomalyScorer(BaseAnomalyScorer):
    """Anomaly scorer based on MAD-normalised latent-space distance.

    Fits on normal data to learn a centroid and distance distribution;
    at inference time returns the normalised deviation from that centroid.

    PORTED FROM: CADE/cade/detect.py::detect_drift_samples() (line 45-105)
    """

    def __init__(self):
        self._centroid: torch.Tensor = None
        self._median_dis: float = 0.0
        self._mad: float = 1.0

    # PORTED FROM: detect.py::get_latent_data_for_each_family() + get_MAD_for_each_family()
    def fit(self, normal_data: torch.Tensor) -> None:
        """Learn centroid and MAD from normal (inlier) samples.

        Args:
            normal_data: Normal samples in latent or feature space. Shape (N, D).

        Raises:
            ValueError: If normal_data is empty.
        """
        if len(normal_data) == 0:
            raise ValueError("normal_data must not be empty")

        self._centroid = normal_data.float().mean(dim=0)
        dis = torch.norm(normal_data.float() - self._centroid.unsqueeze(0), dim=1)
        self._median_dis = dis.median().item()
        mad = 1.4826 * (dis - self._median_dis).abs().median().item()
        self._mad = max(mad, 1e-8)

    # PORTED FROM: detect.py (line 89-97): anomaly = |dist - median| / MAD
    def score(self, data: torch.Tensor) -> torch.Tensor:
        """Return MAD-normalised anomaly scores.

        PORTED FROM: CADE/cade/detect.py (line 91):
          anomaly_k[i] = |dis_k[i] - median(dis[i])| / MAD[i]

        Args:
            data: Samples in same space as fit(). Shape (N, D).

        Returns:
            1-D float tensor of shape (N,). Higher = more anomalous.
        """
        if self._centroid is None:
            raise RuntimeError("CADEAnomalyScorer.fit() must be called before score().")
        centroid = self._centroid.to(data.device)
        dis = torch.norm(data.float() - centroid.unsqueeze(0), dim=1)
        return (dis - self._median_dis).abs() / self._mad
