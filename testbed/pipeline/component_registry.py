"""Component registry — maps string keys to component classes.

All dummy/fallback classes are also defined here.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F

from testbed.base import (BaseDriftDetector, BaseSampleSelector,
                           BaseMemoryManager, BaseAntiForgetting,
                           BaseAnomalyScorer)

# ── Component imports ──────────────────────────────────────────────────────
from testbed.components.ssf import (SSFDriftDetector, SSFSampleSelector,
                                     SSFMemoryManager, SSFAntiForgetting)
from testbed.components.cade import CADEDriftDetector, CADEAnomalyScorer
from testbed.components.cndids import (DDMDriftDetector, CNDIDSMemoryManager,
                                        CNDIDSAntiForgetting, CFEExtractor,
                                        PCAAnomalyScorer)
from testbed.components.baselines import LOFScorer, IsoForestScorer, DeepSVDDScorer
from testbed.components.gpm import GPMAntiForgetting


# ── Dummy / Fallback classes ───────────────────────────────────────────────

class NoDriftDetector(BaseDriftDetector):
    """Always returns no drift."""

    def detect(self, new_data, memory_buffer=None) -> bool:
        return False

    def get_drift_score(self, new_data, memory_buffer=None) -> float:
        return 0.0


class RandomSelector(BaseSampleSelector):
    """Selects label_budget samples uniformly at random."""

    def select(self, new_data: torch.Tensor,
               new_labels: torch.Tensor,
               label_budget: int,
               drift_score: float = 0.0) -> List[int]:
        n = len(new_data)
        k = min(label_budget, n)
        return list(np.random.choice(n, k, replace=False))


class NoMemoryManager(BaseMemoryManager):
    """No-op memory manager."""

    def update(self, selected_data, selected_labels, drift_detected=False):
        pass

    def get_replay_batch(self, batch_size) -> Tuple[None, None]:
        return None, None

    def get_buffer(self) -> Tuple[None, None]:
        return None, None

    def size(self) -> int:
        return 0


class FixedBufferManager(BaseMemoryManager):
    """Reservoir-sampling buffer with fixed capacity."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._buf_data: Optional[torch.Tensor] = None
        self._buf_labels: Optional[torch.Tensor] = None
        self._n_seen: int = 0

    def update(self, selected_data: torch.Tensor,
               selected_labels: torch.Tensor,
               drift_detected: bool = False) -> None:
        for i in range(len(selected_data)):
            self._n_seen += 1
            if self._buf_data is None:
                self._buf_data = selected_data[i:i+1].clone()
                self._buf_labels = selected_labels[i:i+1].clone()
            elif len(self._buf_data) < self.max_size:
                self._buf_data = torch.cat([self._buf_data,
                                             selected_data[i:i+1]], dim=0)
                self._buf_labels = torch.cat([self._buf_labels,
                                               selected_labels[i:i+1]], dim=0)
            else:
                j = np.random.randint(0, self._n_seen)
                if j < self.max_size:
                    self._buf_data[j] = selected_data[i]
                    self._buf_labels[j] = selected_labels[i]

    def get_replay_batch(self, batch_size: int) -> Tuple[Optional[torch.Tensor],
                                                          Optional[torch.Tensor]]:
        if self._buf_data is None:
            return None, None
        n = min(batch_size, len(self._buf_data))
        idx = torch.randperm(len(self._buf_data))[:n]
        return self._buf_data[idx], self._buf_labels[idx]

    def get_buffer(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._buf_data, self._buf_labels

    def size(self) -> int:
        return 0 if self._buf_data is None else len(self._buf_data)


class ReplayOnlyLoss(BaseAntiForgetting):
    """Replay buffer MSE loss only; no explicit forgetting prevention."""

    def compute_loss(self,
                     model,
                     new_batch: Tuple[torch.Tensor, torch.Tensor],
                     replay_batch: Optional[Tuple[torch.Tensor, torch.Tensor]],
                     old_model=None) -> torch.Tensor:
        data, labels = new_batch
        device = data.device
        model = model.to(device)

        out = model(data)
        recon = out[1] if isinstance(out, (tuple, list)) else out
        # MODIFIED: if output shape != input shape (non-AE model), use L2 regulariser
        if recon.shape == data.shape:
            loss = F.mse_loss(recon, data)
        else:
            loss = recon.pow(2).mean()

        if replay_batch is not None and replay_batch[0] is not None:
            r_data, r_labels = replay_batch
            r_data = r_data.to(device)
            r_out = model(r_data)
            r_recon = r_out[1] if isinstance(r_out, (tuple, list)) else r_out
            if r_recon.shape == r_data.shape:
                loss = loss + F.mse_loss(r_recon, r_data)
            else:
                loss = loss + r_recon.pow(2).mean()

        return loss


# ── Registry ───────────────────────────────────────────────────────────────

REGISTRY = {
    "drift_detector": {
        "none":  NoDriftDetector,
        "ssf":   SSFDriftDetector,
        "cade":  CADEDriftDetector,
        "ddm":   DDMDriftDetector,
    },
    "sample_selector": {
        "random": RandomSelector,
        "ssf":    SSFSampleSelector,
    },
    "memory_manager": {
        "none":   NoMemoryManager,
        "fixed":  FixedBufferManager,
        "ssf":    SSFMemoryManager,
        "cndids": CNDIDSMemoryManager,
    },
    "anti_forgetting": {
        "none":    ReplayOnlyLoss,
        "lwf_ssf": SSFAntiForgetting,
        "cfe":     CFEExtractor,
        "cndids":  CNDIDSAntiForgetting,
        "gpm":     GPMAntiForgetting,
    },
    "anomaly_scorer": {
        "pca":       PCAAnomalyScorer,
        "deep_svdd": DeepSVDDScorer,
        "lof":       LOFScorer,
        "isoforest": IsoForestScorer,
        "cade_mad":  CADEAnomalyScorer,
    },
}


def build(slot: str, name: str, **kwargs):
    """Instantiate a component from the registry.

    Args:
        slot: Component slot name (e.g. 'drift_detector').
        name: Component key within that slot (e.g. 'ssf').
        **kwargs: Constructor keyword arguments forwarded to the class.

    Returns:
        Instantiated component object.

    Raises:
        KeyError: If slot or name is not registered.
    """
    if slot not in REGISTRY:
        raise KeyError(f"Unknown slot: {slot!r}. Available: {list(REGISTRY)}")
    slot_map = REGISTRY[slot]
    if name not in slot_map:
        raise KeyError(
            f"Unknown component {name!r} in slot {slot!r}. "
            f"Available: {list(slot_map)}"
        )
    cls = slot_map[name]
    try:
        return cls(**kwargs)
    except TypeError:
        # Filter kwargs to only those the constructor accepts
        import inspect
        sig = inspect.signature(cls.__init__)
        valid = set(sig.parameters) - {'self'}
        filtered = {k: v for k, v in kwargs.items() if k in valid}
        return cls(**filtered)
