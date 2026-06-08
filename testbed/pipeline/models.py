"""FCL per-paper model implementations.

Each paper's original encoder/decoder architecture is preserved exactly.
All models expose the unified interface:
  forward(x) → (z, x_hat, logit)

  z     : latent representation  (N, latent_dim)
  x_hat : reconstruction         (N, input_dim)
  logit : binary logit           (N,)   [pre-sigmoid, for BCEWithLogitsLoss]

Paper → model:
  SSF     → SSFModel     (AE_Classifier from utils.py)
  CND-IDS → CNDIDSModel  (AE_Extractor from AE_Exactor.py)
  CADE    → CADEModel    (symmetric CAE from cade/autoencoder.py, PyTorch port)
  SPIDER  → SSFModel     (no dedicated architecture in repo)
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nearest_power_of_2(n: int) -> int:
    """Return smallest power of 2 >= n.

    Mirrors SSF utils.py: max(2 ** math.ceil(math.log2(input_dim)), 2)
    """
    if n <= 1:
        return 2
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------
# SSF Model
# ---------------------------------------------------------------------------

class SSFModel(nn.Module):
    """SSF AutoEncoder with binary classifier head.

    FROM: SSF-Strategic-Selection-and-Forgetting/utils.py  AE_Classifier (lines 28-63)

    Architecture:
      Encoder:    input → nearest_pow2//2 (ReLU) → nearest_pow2//4   (no act on latent)
      Decoder:    ReLU(latent) → nearest_pow2//2 (ReLU) → input
      Classifier: ReLU(latent) → 1                                     (logit, no sigmoid)

    Dimensions for common datasets:
      NSL-KDD  (dim=121): nearest_pow2=128, hidden=64,  latent=32
      UNSW-NB15(dim=196): nearest_pow2=256, hidden=128, latent=64

    Note: Original SSF uses Sigmoid on the classifier for BCELoss. This model
    omits Sigmoid to produce raw logits — BCEWithLogitsLoss is numerically
    equivalent and more stable.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        n = _nearest_power_of_2(input_dim)
        self.hidden_dim = n // 2
        self.latent_dim = n // 4

        # FROM: utils.py lines 28-40 — encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
            # No activation — latent vector is linear (per SSF)
        )
        # FROM: utils.py lines 42-47 — decoder (starts with ReLU on latent)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, input_dim),
        )
        # FROM: utils.py lines 49-53 — classifier (starts with ReLU on latent)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.latent_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        logit = self.classifier(z).squeeze(-1)   # (N,)
        return z, x_hat, logit


# ---------------------------------------------------------------------------
# CND-IDS Model
# ---------------------------------------------------------------------------

class CNDIDSModel(nn.Module):
    """CND-IDS deep autoencoder with binary classifier head.

    FROM: CND-IDS/FeatureExtractors/AE_Exactor.py

    Architecture:
      Encoder: input → 256 → 128 → 128 → 96 → latent  (all ReLU, no act on latent)
      Decoder: latent → 96 → 128 → 128 → 256 → input  (ReLU hidden, Sigmoid output)
      Classifier: latent → 1                             (logit, no sigmoid)

    Default latent_dim=96 matches the last intermediate size in AE_Exactor.py.

    Note: Original AE_Extractor.forward() returns encoder(x).detach() — gradient
    is detached. This model keeps gradient flow for backpropagation.
    """

    def __init__(self, input_dim: int, latent_dim: int = 96):
        super().__init__()
        self.latent_dim = latent_dim

        # FROM: AE_Exactor.py — encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, latent_dim),
        )
        # FROM: AE_Exactor.py — decoder (Sigmoid on output)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )
        # Classifier head added for pipeline compatibility
        self.classifier = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        logit = self.classifier(z).squeeze(-1)   # (N,)
        return z, x_hat, logit


# ---------------------------------------------------------------------------
# CADE Model
# ---------------------------------------------------------------------------

class CADEModel(nn.Module):
    """CADE Contrastive AutoEncoder (PyTorch port).

    FROM: CADE/cade/autoencoder.py — symmetric configurable AE

    Architecture (default dims=[input_dim, 64, 32]):
      Encoder: input → 64 (ReLU) → 32          (no activation on latent, per CADE)
      Decoder: 32 (ReLU) → 64 (ReLU) → input   (no activation on output, per CADE)
      Classifier: latent → 1                    (logit, no sigmoid)

    The dims list controls depth: dims[0]=input, dims[1:-1]=hidden, dims[-1]=latent.
    """

    def __init__(self, input_dim: int, dims: Optional[List[int]] = None):
        super().__init__()
        if dims is None:
            dims = [input_dim, 64, 32]
        assert dims[0] == input_dim, "dims[0] must equal input_dim"
        self.latent_dim = dims[-1]

        # FROM: autoencoder.py — encoder (no activation on final latent)
        enc_layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)

        # FROM: autoencoder.py — decoder (symmetric, no activation on output)
        rev = list(reversed(dims))
        dec_layers: List[nn.Module] = []
        for i in range(len(rev) - 1):
            dec_layers.append(nn.Linear(rev[i], rev[i + 1]))
            if i < len(rev) - 2:
                dec_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec_layers)

        # Classifier head added for pipeline compatibility
        self.classifier = nn.Linear(self.latent_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        logit = self.classifier(z).squeeze(-1)   # (N,)
        return z, x_hat, logit


# ---------------------------------------------------------------------------
# Model selection utilities
# ---------------------------------------------------------------------------

def select_paper(combo_dict: dict) -> str:
    """Return the canonical paper identifier for a component combination.

    Priority: cndids > cade > ssf (default)

    CND-IDS model is used when cndids memory/anti-forgetting is present,
    because those components are designed for CND-IDS latent space dimensions.
    CADE model is used when cade drift detector or cade_mad scorer is present.
    SSF model is the default for all other combinations.
    """
    anti   = combo_dict.get('anti_forgetting', 'none')
    mem    = combo_dict.get('memory_manager',  'none')
    drift  = combo_dict.get('drift_detector',  'none')
    scorer = combo_dict.get('anomaly_scorer',  'pca')

    if anti == 'cndids' or mem == 'cndids':
        return 'cndids'
    if drift == 'cade' or scorer == 'cade_mad':
        return 'cade'
    return 'ssf'


def build_model(paper: str, input_dim: int, **kwargs) -> nn.Module:
    """Build the paper-appropriate model for a given input dimension.

    Args:
        paper:     'ssf' | 'cndids' | 'cade'
        input_dim: Feature dimension of input data.
        **kwargs:  Optional overrides (latent_dim for cndids, dims for cade).

    Returns:
        nn.Module with forward(x) → (z, x_hat, logit).
    """
    if paper == 'cndids':
        return CNDIDSModel(input_dim, latent_dim=kwargs.get('latent_dim', 96))
    if paper == 'cade':
        dims = kwargs.get('dims', [input_dim, 64, 32])
        return CADEModel(input_dim, dims=dims)
    return SSFModel(input_dim)
