"""Component registry — maps string keys to component classes.

Paper → component mapping (슬롯 순서: drift / sample / memory / anti / anomaly):
  CND-IDS : ddm  / random / cndids / cndids  / pca
  SSF     : ssf  / ssf    / ssf    / lwf_ssf / pca
  CADE    : cade / random / none   / none    / cade_mad
  SPIDER  : none / random / fifo   / gpm     / pca

그리드: 4 × 2 × 4 × 4 × 2 = 256 조합 / 데이터셋
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
from testbed.components.cndids import (CNDIDSAntiForgetting, PCAAnomalyScorer,
                                        DDMDriftDetector, CNDIDSMemoryManager)
from testbed.components.gpm import GPMAntiForgetting


# ── Fallback / paper-agnostic classes ─────────────────────────────────────

class NoDriftDetector(BaseDriftDetector):
    """No drift detection (pass-through)."""

    def detect(self, new_data, memory_buffer=None) -> bool:
        return False

    def get_drift_score(self, new_data, memory_buffer=None) -> float:
        return 0.0


class AllSampleSelector(BaseSampleSelector):
    """Selects the first label_budget samples without any active-learning filter.

    Represents the CND-IDS 'use all available samples' strategy — no sample
    selection overhead.
    """

    def select(self, new_data: torch.Tensor,
               new_labels: torch.Tensor,
               label_budget: int,
               drift_score: float = 0.0) -> List[int]:
        n = min(label_budget, len(new_data))
        return list(range(n))


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
    """No-op memory manager (CND-IDS does not use a replay buffer)."""

    def update(self, selected_data, selected_labels, drift_detected=False):
        pass

    def get_replay_batch(self, batch_size) -> Tuple[None, None]:
        return None, None

    def get_buffer(self) -> Tuple[None, None]:
        return None, None

    def size(self) -> int:
        return 0


class FIFOMemoryManager(BaseMemoryManager):
    """FIFO ring buffer — evicts oldest samples on overflow.

    Used for SPIDER's unlabeled privacy-preserving replay buffer.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._buf_data: Optional[torch.Tensor] = None
        self._buf_labels: Optional[torch.Tensor] = None

    def update(self, selected_data: torch.Tensor,
               selected_labels: torch.Tensor,
               drift_detected: bool = False) -> None:
        if self._buf_data is None:
            self._buf_data = selected_data.clone()
            self._buf_labels = selected_labels.clone()
        else:
            self._buf_data = torch.cat([self._buf_data, selected_data], dim=0)
            self._buf_labels = torch.cat([self._buf_labels, selected_labels], dim=0)

        if len(self._buf_data) > self.max_size:
            excess = len(self._buf_data) - self.max_size
            self._buf_data = self._buf_data[excess:]
            self._buf_labels = self._buf_labels[excess:]

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
    """Replay MSE loss only; no explicit forgetting penalty.

    'none' anti-forgetting baseline: uses reconstruction loss on
    new batch + replay batch. Gradient flows only through the decoder.
    """

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
        if recon.shape == data.shape:
            loss = F.mse_loss(recon, data)
        else:
            loss = recon.pow(2).mean()

        if replay_batch is not None and replay_batch[0] is not None:
            r_data = replay_batch[0].to(device)
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
        "none": NoDriftDetector,       # CADE·SPIDER — 드리프트 감지 없음
        "ssf":  SSFDriftDetector,      # SSF — KS 검정
        "cade": CADEDriftDetector,     # CADE — MAD 거리
        "ddm":  DDMDriftDetector,      # CND-IDS — z-스코어 3상태 감지
    },
    "sample_selector": {
        "random": RandomSelector,      # CADE·SPIDER·CND-IDS — 무작위 선택
        "ssf":    SSFSampleSelector,   # SSF — KL-div 마스크 최적화
    },
    "memory_manager": {
        "none":   NoMemoryManager,     # CADE — 버퍼 없음
        "fifo":   FIFOMemoryManager,   # SPIDER — 링 버퍼
        "ssf":    SSFMemoryManager,    # SSF — 표류 시 공격적 교체
        "cndids": CNDIDSMemoryManager, # CND-IDS — 클래스 균형 버퍼
    },
    "anti_forgetting": {
        "none":    ReplayOnlyLoss,      # CADE — 재구성 손실만
        "lwf_ssf": SSFAntiForgetting,   # SSF — InfoNCE + LwF
        "cndids":  CNDIDSAntiForgetting,# CND-IDS — L_CS + L_R + L_CL
        "gpm":     GPMAntiForgetting,   # SPIDER — 그래디언트 정사영
    },
    "anomaly_scorer": {
        "pca":      PCAAnomalyScorer,  # CND-IDS·SSF·SPIDER — 재구성 오차
        "cade_mad": CADEAnomalyScorer, # CADE·SSF — MAD 정규화 거리
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
        import inspect
        sig = inspect.signature(cls.__init__)
        valid = set(sig.parameters) - {'self'}
        filtered = {k: v for k, v in kwargs.items() if k in valid}
        return cls(**filtered)
