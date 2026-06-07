"""PCA Anomaly Scorer — reconstruction-error-based scorer.

FROM: CND-IDS/AnomolyDetectors/PCA.py::PCA_model (line 1-34)

MODIFIED: sklearn import is lazy (inside methods) to avoid binary
incompatibility between sklearn/scipy and numpy >= 2.x at import time.
Fallback: pure numpy/torch SVD-based PCA if sklearn is unavailable.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import torch
import numpy as np
from testbed.base.anomaly_scorer import BaseAnomalyScorer


class PCAAnomalyScorer(BaseAnomalyScorer):
    """Anomaly scorer based on PCA reconstruction error.

    FROM: CND-IDS/AnomolyDetectors/PCA.py::PCA_model

    score(x) = mean |x - PCA.inverse_transform(PCA.transform(x))|  per sample.
    """

    def __init__(self, pca_dim='auto', svd_solver='full'):
        self.pca_dim = pca_dim
        self.svd_solver = svd_solver
        self._use_sklearn = True
        self._pca = None
        self._is_fit = False
        # Pure-numpy fallback state
        self._mean = None
        self._components = None

    # FROM: CND-IDS/AnomolyDetectors/PCA.py::PCA_model.fit() (line 11-24)
    def fit(self, normal_data: torch.Tensor) -> None:
        """Fit PCA to normal data.

        MODIFIED: lazy sklearn import; falls back to numpy SVD if sklearn fails.

        Args:
            normal_data: Normal samples. Shape (N, D).
        """
        if len(normal_data) == 0:
            raise ValueError("normal_data must not be empty")

        X = self._to_numpy(normal_data)

        try:
            from sklearn.decomposition import PCA as SkPCA
            if self.pca_dim == 'auto':
                pca_full = SkPCA()
                pca_full.fit(X)
                cumvar = np.cumsum(pca_full.explained_variance_ratio_)
                dim = int(np.argmax(cumvar >= 0.95)) + 1
            else:
                dim = int(self.pca_dim)
            self._pca = SkPCA(n_components=dim, svd_solver=self.svd_solver)
            self._pca.fit(X)
            self._use_sklearn = True
        except Exception:
            # Fallback: numpy SVD
            self._use_sklearn = False
            self._fit_numpy(X)

        self._is_fit = True

    def _fit_numpy(self, X: np.ndarray) -> None:
        """Pure-numpy SVD PCA fallback."""
        self._mean = X.mean(axis=0)
        Xc = X - self._mean
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        # Keep components explaining >= 95% variance
        energy = S ** 2
        cumvar = np.cumsum(energy) / (energy.sum() + 1e-10)
        if self.pca_dim == 'auto':
            k = int(np.argmax(cumvar >= 0.95)) + 1
        else:
            k = int(self.pca_dim)
        k = max(1, min(k, len(S)))
        self._components = Vt[:k]  # (k, D)

    def score(self, data: torch.Tensor) -> torch.Tensor:
        """Return mean absolute reconstruction error per sample.

        FROM: CND-IDS/AnomolyDetectors/PCA.py::PCA_model.predict()

        Args:
            data: Input samples. Shape (N, D).

        Returns:
            Float tensor of shape (N,).
        """
        if not self._is_fit:
            raise RuntimeError("PCAAnomalyScorer.fit() must be called before score().")

        X = self._to_numpy(data)

        if self._use_sklearn and self._pca is not None:
            try:
                latent = self._pca.transform(X)
                recon = self._pca.inverse_transform(latent)
                scores = np.abs(X - recon).mean(axis=1)
                return torch.from_numpy(scores.astype(np.float32))
            except Exception:
                pass  # fall through to numpy path

        # Numpy SVD fallback
        Xc = X - self._mean
        z = Xc @ self._components.T        # (N, k)
        recon = z @ self._components + self._mean  # (N, D)
        scores = np.abs(X - recon).mean(axis=1)
        return torch.from_numpy(scores.astype(np.float32))

    @staticmethod
    def _to_numpy(t: torch.Tensor) -> np.ndarray:
        if hasattr(t, 'detach'):
            t = t.detach()
        if hasattr(t, 'cpu'):
            t = t.cpu()
        return t.numpy() if hasattr(t, 'numpy') else np.array(t)
