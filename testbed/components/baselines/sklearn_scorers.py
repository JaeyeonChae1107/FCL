"""AnoShift sklearn-based anomaly scorers.

MODIFIED: All sklearn imports are lazy (inside methods) to avoid binary
incompatibility between sklearn/scipy and numpy >= 2.x at import time.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from typing import Optional, List
import torch
import torch.nn as nn
import numpy as np
from testbed.base.anomaly_scorer import BaseAnomalyScorer


# ── Helper ─────────────────────────────────────────────────────────────────
def _to_numpy(t: torch.Tensor) -> np.ndarray:
    if hasattr(t, 'detach'):
        t = t.detach()
    if hasattr(t, 'cpu'):
        t = t.cpu()
    return t.numpy() if hasattr(t, 'numpy') else np.array(t)


# ── LOF Scorer ─────────────────────────────────────────────────────────────
class LOFScorer(BaseAnomalyScorer):
    """Local Outlier Factor anomaly scorer.

    FROM: Anoshift/AnoShift/baselines_OOD_setup/baseline_LOF.py
    MODIFIED: lazy sklearn import.
    """

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.1):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self._lof = None

    def fit(self, normal_data: torch.Tensor) -> None:
        try:
            from sklearn.neighbors import LocalOutlierFactor
            X = _to_numpy(normal_data)
            self._lof = LocalOutlierFactor(
                n_neighbors=self.n_neighbors,
                novelty=True,
                contamination=self.contamination,
            )
            self._lof.fit(X)
        except ImportError:
            # Fallback: store training data for manual LOF computation
            self._train_data = _to_numpy(normal_data)

    def score(self, data: torch.Tensor) -> torch.Tensor:
        X = _to_numpy(data)
        if self._lof is not None:
            try:
                scores = -1.0 * self._lof.score_samples(X)
                return torch.from_numpy(scores.astype(np.float32))
            except Exception:
                pass
        # Fallback: mean L2 distance to k nearest neighbours in training set
        if hasattr(self, '_train_data'):
            dists = np.linalg.norm(
                X[:, None, :] - self._train_data[None, :, :], axis=2)
            k = min(self.n_neighbors, len(self._train_data))
            scores = np.sort(dists, axis=1)[:, :k].mean(axis=1)
            return torch.from_numpy(scores.astype(np.float32))
        raise RuntimeError("LOFScorer.fit() must be called before score().")


# ── IsolationForest Scorer ─────────────────────────────────────────────────
class IsoForestScorer(BaseAnomalyScorer):
    """Isolation Forest anomaly scorer.

    FROM: Anoshift/AnoShift/baselines_ID_setup/baseline_isoforest.py
    MODIFIED: lazy sklearn import.
    """

    def __init__(self, n_estimators: int = 100,
                 contamination: float = 0.1, random_state: int = 42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self._model = None

    def fit(self, normal_data: torch.Tensor) -> None:
        try:
            from sklearn.ensemble import IsolationForest
            X = _to_numpy(normal_data)
            self._model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=self.random_state,
            )
            self._model.fit(X)
        except ImportError:
            # Fallback: store reference data
            self._ref_data = _to_numpy(normal_data)

    def score(self, data: torch.Tensor) -> torch.Tensor:
        X = _to_numpy(data)
        if self._model is not None:
            try:
                scores = -1.0 * self._model.score_samples(X)
                return torch.from_numpy(scores.astype(np.float32))
            except Exception:
                pass
        if hasattr(self, '_ref_data'):
            # Fallback: mean distance to training centroid
            centroid = self._ref_data.mean(axis=0)
            scores = np.linalg.norm(X - centroid[None, :], axis=1)
            return torch.from_numpy(scores.astype(np.float32))
        raise RuntimeError("IsoForestScorer.fit() must be called before score().")


# ── Deep SVDD Scorer ───────────────────────────────────────────────────────
class _SVDDEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepSVDDScorer(BaseAnomalyScorer):
    """Deep SVDD anomaly scorer.

    FROM: Anoshift/AnoShift/baselines_OOD_setup/baseline_deep_svdd/
    MODIFIED: pure PyTorch reimplementation.
    """

    def __init__(self, input_dim: int = 121,
                 hidden_dims: Optional[List[int]] = None,
                 latent_dim: int = 32,
                 nu: float = 0.1,
                 n_epochs: int = 50,
                 lr: float = 1e-4,
                 batch_size: int = 128,
                 device: str = 'cpu'):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.latent_dim = latent_dim
        self.nu = nu
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self._encoder: Optional[_SVDDEncoder] = None
        self._center: Optional[torch.Tensor] = None

    def _init_center(self, embeddings: torch.Tensor, eps: float = 0.1):
        c = embeddings.mean(dim=0)
        c[(c.abs() < eps) & (c >= 0)] = eps
        c[(c.abs() < eps) & (c < 0)] = -eps
        return c

    def fit(self, normal_data: torch.Tensor) -> None:
        device = torch.device(self.device)
        X = normal_data.float().to(device)
        in_dim = X.shape[1]
        self._encoder = _SVDDEncoder(in_dim, self.hidden_dims,
                                     self.latent_dim).to(device)
        optimizer = torch.optim.Adam(self._encoder.parameters(), lr=self.lr)

        self._encoder.eval()
        with torch.no_grad():
            all_z = self._encoder(X)
        self._center = self._init_center(all_z).to(device)
        R = torch.tensor(0.0, device=device)

        self._encoder.train()
        warm_up = max(1, self.n_epochs // 10)

        for epoch in range(self.n_epochs):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), self.batch_size):
                batch = X[perm[i:i + self.batch_size]]
                optimizer.zero_grad()
                z = self._encoder(batch)
                dist = torch.sum((z - self._center) ** 2, dim=1)
                scores = dist - R ** 2
                loss = R ** 2 + (1.0 / self.nu) * torch.mean(
                    torch.clamp(scores, min=0.0))
                loss.backward()
                optimizer.step()

            if epoch >= warm_up:
                with torch.no_grad():
                    z_all = self._encoder(X)
                    dist_all = torch.sum((z_all - self._center) ** 2, dim=1)
                R = torch.quantile(dist_all.sqrt(), 1 - self.nu).detach()

        self._encoder.eval()

    def score(self, data: torch.Tensor) -> torch.Tensor:
        if self._encoder is None:
            raise RuntimeError("DeepSVDDScorer.fit() must be called before score().")
        device = next(self._encoder.parameters()).device
        X = data.float().to(device)
        self._encoder.eval()
        with torch.no_grad():
            z = self._encoder(X)
        return torch.sum((z - self._center) ** 2, dim=1).cpu()
