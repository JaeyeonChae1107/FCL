"""FCL shared model: autoencoder + binary classifier.

All anti-forgetting components and the anomaly scorer receive model output
as a 3-tuple (z, x_hat, logit):
  z     : latent representation (N, latent_dim)  — anomaly scorer input
  x_hat : reconstruction (N, input_dim)          — CND-IDS L_R, GPM, replay loss
  logit : binary logit (N, 1)                    — SSF L_task
"""

import torch
import torch.nn as nn
from typing import Tuple


class FCLAutoEncoder(nn.Module):
    """Shared autoencoder with encoder, decoder, and binary classifier head.

    forward(x) → (z, x_hat, logit)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.classifier = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        logit = self.classifier(z)
        return z, x_hat, logit
