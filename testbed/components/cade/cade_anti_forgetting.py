"""CADE Anti-Forgetting — contrastive + reconstruction loss.

FROM: CADE/cade/autoencoder.py::ContrastiveAE.train() (line 220-233)

Training objective:
  L_total = lambda_1 * L_contrastive + L_reconstruction

  L_contrastive: margin-based pair loss (margin=10.0, lambda_1=0.1 per paper)
  L_reconstruction: MSE between model output and input

Both losses operate directly on the CADEModel's encoder/decoder outputs.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from testbed.base.anti_forgetting import BaseAntiForgetting


class CADEAntiForgetting(BaseAntiForgetting):
    """CADE contrastive + reconstruction loss.

    FROM: CADE/cade/autoencoder.py::ContrastiveAE.train() (line 220-233)
    lambda_1=0.1 and margin=10.0 from CADE/cade/utils.py line 72.
    """

    def __init__(self, lambda_1: float = 0.1, margin: float = 10.0):
        """
        Args:
            lambda_1: Contrastive loss weight (default 0.1, per paper).
            margin:   Margin for dissimilar pairs (default 10.0, per paper).
        """
        self.lambda_1 = lambda_1
        self.margin = margin

    def compute_loss(self,
                     model: nn.Module,
                     new_batch: Tuple[torch.Tensor, torch.Tensor],
                     replay_batch: Optional[Tuple[torch.Tensor, torch.Tensor]],
                     old_model: Optional[nn.Module] = None) -> torch.Tensor:
        """lambda_1 * contrastive_loss + mse_loss.

        FROM: CADE/cade/autoencoder.py::ContrastiveAE.train() (line 228-233)

        Args:
            model:        CADEModel. forward(x) → (z, x_hat, logit).
            new_batch:    (data, labels) for current round.
            replay_batch: (r_data, r_labels) or None.

        Returns:
            Scalar loss tensor.
        """
        data, labels = new_batch
        device = data.device
        model = model.to(device)

        if replay_batch is not None and replay_batch[0] is not None:
            data   = torch.cat([data,   replay_batch[0].to(device)], dim=0)
            labels = torch.cat([labels, replay_batch[1].to(device)], dim=0)

        out = model(data)
        z     = out[0] if isinstance(out, (tuple, list)) else out
        x_hat = out[1] if isinstance(out, (tuple, list)) and len(out) >= 2 else data

        # Reconstruction (AE) loss — FROM: autoencoder.py train() ae_loss
        ae_loss = F.mse_loss(x_hat, data)

        # Contrastive loss on random pairs — FROM: autoencoder.py train() lines 220-233
        n = len(data)
        half = n // 2
        if half == 0:
            return ae_loss

        idx_i = torch.randperm(n, device=device)[:half]
        idx_j = torch.randperm(n, device=device)[:half]
        is_same = (labels[idx_i] == labels[idx_j]).float()

        dist = torch.norm(z[idx_i] - z[idx_j], p=2, dim=1)
        same_loss = is_same * dist
        diff_loss = (1.0 - is_same) * F.relu(self.margin - dist)
        c_loss = (same_loss + diff_loss).mean()

        return self.lambda_1 * c_loss + ae_loss

    def on_task_end(self, model: nn.Module) -> None:
        pass
