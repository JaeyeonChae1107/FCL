"""ContrastiveAE — PyTorch reimplementation.

PORTED FROM: CADE/cade/autoencoder.py::ContrastiveAE (line 175-293)
             (Original uses Keras/TF1; this is a clean PyTorch port.)
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveAE(nn.Module):
    """Symmetric autoencoder trained with MSE + contrastive loss.

    PORTED FROM: CADE/cade/autoencoder.py::Autoencoder.build() (line 69-107)
                 CADE/cade/autoencoder.py::ContrastiveAE.train() (line 181-293)

    Args:
        dims: Layer sizes including input and latent.
              e.g. [121, 64, 32, 16] → encoder 121→64→32→16,
              decoder 16→32→64→121.
        activation: Activation function name ('relu', 'tanh', etc.).
    """

    def __init__(self, dims: List[int], activation: str = 'relu'):
        super().__init__()
        act_fn = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'elu': nn.ELU}[activation]

        # Encoder: input → latent (no activation on last layer)
        encoder_layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                encoder_layers.append(act_fn())
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: latent → input (symmetric, no activation on output)
        decoder_dims = list(reversed(dims))
        decoder_layers: List[nn.Module] = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            if i < len(decoder_dims) - 2:
                decoder_layers.append(act_fn())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode then decode.

        Args:
            x: Input tensor, shape (N, input_dim).

        Returns:
            Tuple (z, x_recon) where z is latent and x_recon is reconstruction.
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

    # PORTED FROM: CADE/cade/autoencoder.py::ContrastiveAE.train() (line 220-232)
    @staticmethod
    def contrastive_loss(z_i: torch.Tensor, z_j: torch.Tensor,
                         is_same: torch.Tensor,
                         margin: float = 1.0) -> torch.Tensor:
        """Contrastive loss between paired embeddings.

        PORTED FROM: CADE/cade/autoencoder.py::ContrastiveAE.train() (line 220-226)
        Original TF: contrastive_loss = is_same * dist + (1-is_same)*relu(margin-dist)

        Args:
            z_i: Left embeddings. Shape (B, latent_dim).
            z_j: Right embeddings. Shape (B, latent_dim).
            is_same: Float tensor of 1.0 (same class) / 0.0 (diff class). Shape (B,).
            margin: Margin for dissimilar pairs (default 1.0).

        Returns:
            Scalar loss.
        """
        dist = torch.norm(z_i - z_j, p=2, dim=1)  # (B,)
        same_loss = is_same * dist
        diff_loss = (1.0 - is_same) * F.relu(margin - dist)
        return (same_loss + diff_loss).mean()

    def reconstruction_loss(self, x: torch.Tensor,
                             x_recon: torch.Tensor) -> torch.Tensor:
        """Mean squared error reconstruction loss.

        Args:
            x: Original input. Shape (N, D).
            x_recon: Reconstruction. Shape (N, D).

        Returns:
            Scalar MSE loss.
        """
        return F.mse_loss(x_recon, x)

    def combined_loss(self, x: torch.Tensor, y: torch.Tensor,
                      lambda_1: float = 0.1,
                      margin: float = 1.0) -> torch.Tensor:
        """Full CADE training loss for a batch.

        PORTED FROM: CADE/cade/autoencoder.py::ContrastiveAE.train() (line 228-233)
        loss = lambda_1 * contrastive_loss + ae_loss

        Creates random pairs within the batch for contrastive supervision.

        Args:
            x: Input batch. Shape (N, D).
            y: Class labels. Shape (N,).
            lambda_1: Contrastive loss weight (default 0.1).
            margin: Contrastive margin (default 1.0).

        Returns:
            Scalar total loss.
        """
        z, x_recon = self.forward(x)
        ae_loss = self.reconstruction_loss(x, x_recon)

        # Build random pairs
        n = len(x)
        half = n // 2
        if half == 0:
            return ae_loss

        idx_i = torch.randperm(n, device=x.device)[:half]
        idx_j = torch.randperm(n, device=x.device)[:half]
        is_same = (y[idx_i] == y[idx_j]).float()
        c_loss = self.contrastive_loss(z[idx_i], z[idx_j], is_same, margin)
        return lambda_1 * c_loss + ae_loss
