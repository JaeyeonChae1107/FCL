"""CND-IDS Anti-Forgetting.

Loss = L_CS + λ_R · L_R + λ_CL · L_CL

FROM: Fuhrman et al. "CND-IDS: Continual Network Defence IDS" (DAC 2025)
  L_CS : Cluster Separation Loss
         K-Means (K=2) pseudo-labels on z → TripletMarginLoss
         pushes normal / attack latent clusters apart
  L_R  : Reconstruction loss  MSE(x_hat, x)
  L_CL : Continual Learning loss  Σ_i MSE(z_current, old_model_i(x))
         multi-teacher LwF on latent space
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from typing import Optional, Tuple, List
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from testbed.base.anti_forgetting import BaseAntiForgetting


class CNDIDSAntiForgetting(BaseAntiForgetting):
    """Full CND-IDS loss: L_CS + λ_R·L_R + λ_CL·L_CL.

    FROM: Fuhrman et al. DAC 2025 — CND_IDS.py::CND_IDS.fit() (lines 100-195)
    """

    def __init__(self, lambda_r: float = 0.1, lambda_cl: float = 0.1,
                 triplet_margin: float = 1.0, kmeans_iters: int = 10):
        """
        Args:
            lambda_r:       Reconstruction loss weight (default 0.1).
            lambda_cl:      Continual learning (LwF) loss weight (default 0.1).
            triplet_margin: Margin for TripletMarginLoss (default 1.0).
            kmeans_iters:   K-Means iterations for pseudo-labelling (default 10).
        """
        self.lambda_r = lambda_r
        self.lambda_cl = lambda_cl
        self.triplet_margin = triplet_margin
        self.kmeans_iters = kmeans_iters
        self.old_models: List[nn.Module] = []

    # ------------------------------------------------------------------ L_CS
    def _cluster_separation_loss(self, z: torch.Tensor) -> torch.Tensor:
        """K-Means pseudo-labels → TripletMarginLoss.

        FROM: CND_IDS.py::CND_IDS.metric_loss() (line 76-78)
        Runs K=2 K-Means on z.detach() to obtain pseudo cluster assignments,
        then computes TripletMarginLoss(anchor, positive, negative).
        """
        n = len(z)
        if n < 4:
            return torch.tensor(0.0, device=z.device)

        # K-Means with K=2 (no gradients through cluster assignment)
        with torch.no_grad():
            z_d = z.detach()
            # Initialise centres from two random samples
            perm = torch.randperm(n, device=z.device)
            c0, c1 = z_d[perm[0]].clone(), z_d[perm[1]].clone()

            labels = torch.zeros(n, dtype=torch.long, device=z.device)
            for _ in range(self.kmeans_iters):
                d0 = torch.norm(z_d - c0, dim=1)
                d1 = torch.norm(z_d - c1, dim=1)
                labels = (d1 < d0).long()
                mask0, mask1 = labels == 0, labels == 1
                if mask0.any():
                    c0 = z_d[mask0].mean(0)
                if mask1.any():
                    c1 = z_d[mask1].mean(0)

            # Assign smaller cluster = class 1 (attack), larger = class 0 (normal)
            if mask0.sum() < mask1.sum():
                labels = 1 - labels
                mask0, mask1 = mask1, mask0

            # Need both classes present to form triplets
            if not (mask0.any() and mask1.any()):
                return torch.tensor(0.0, device=z.device)

            # Build triplet indices
            idx0 = mask0.nonzero(as_tuple=True)[0]
            idx1 = mask1.nonzero(as_tuple=True)[0]

        # Sample min(n, max_triplets) triplets (cap to avoid OOM on large batches)
        max_triplets = min(n, 64)
        anchors, positives, negatives = [], [], []
        for i in range(max_triplets):
            label_i = labels[i % n].item()
            same_idx = idx0 if label_i == 0 else idx1
            diff_idx = idx1 if label_i == 0 else idx0
            # Need at least one sample different from self in same cluster
            same_excl = same_idx[same_idx != (i % n)]
            if len(same_excl) == 0 or len(diff_idx) == 0:
                continue
            p = same_excl[torch.randint(len(same_excl), (1,), device=z.device).item()]
            neg = diff_idx[torch.randint(len(diff_idx), (1,), device=z.device).item()]
            anchors.append(z[i % n])
            positives.append(z[p])
            negatives.append(z[neg])

        if not anchors:
            return torch.tensor(0.0, device=z.device)

        a = torch.stack(anchors)
        p = torch.stack(positives)
        n_t = torch.stack(negatives)
        return F.triplet_margin_loss(a, p, n_t, margin=self.triplet_margin)

    # ------------------------------------------------------------------ L_CL
    def _lcl_loss(self, model: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """Multi-teacher LwF on latent space.

        FROM: CND_IDS.py::CND_IDS.LwFloss() (line 54-69)
        L_CL = Σ_i MSE(z_current, old_model_i(x))
        """
        if not self.old_models:
            return torch.tensor(0.0, device=data.device)

        out = model(data)
        z = out[0] if isinstance(out, (tuple, list)) else out

        total = torch.tensor(0.0, device=data.device)
        for old_model in self.old_models:
            old_model.eval()
            with torch.no_grad():
                o = old_model(data.cpu())
                z_old = (o[0] if isinstance(o, (tuple, list)) else o).to(data.device)
            total = total + F.mse_loss(z, z_old)
        return total

    # ------------------------------------------------------------------ API
    def compute_loss(self,
                     model: nn.Module,
                     new_batch: Tuple[torch.Tensor, torch.Tensor],
                     replay_batch: Optional[Tuple[torch.Tensor, torch.Tensor]],
                     old_model: Optional[nn.Module] = None) -> torch.Tensor:
        """L_total = L_CS + λ_R·L_R + λ_CL·L_CL.

        FROM: CND_IDS.py::CND_IDS.fit() (line 156-165)
        replay_batch is ignored — CND-IDS uses teacher distillation, not replay.
        """
        data, labels = new_batch
        device = data.device
        model = model.to(device)

        out = model(data)
        z = out[0] if isinstance(out, (tuple, list)) else out
        x_hat = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None

        # L_R: reconstruction loss
        l_r = (F.mse_loss(x_hat, data)
               if x_hat is not None and x_hat.shape == data.shape
               else torch.tensor(0.0, device=device))

        # L_CS: cluster separation loss
        l_cs = self._cluster_separation_loss(z)

        # L_CL: multi-teacher LwF
        l_cl = self._lcl_loss(model, data)

        total = l_cs + self.lambda_r * l_r + self.lambda_cl * l_cl

        # Ensure gradient flows even when sub-losses are all zero
        if not total.requires_grad:
            total = total + 0.0 * z.sum()

        return total

    def on_task_end(self, model: nn.Module) -> None:
        """Snapshot current model as a new frozen teacher.

        FROM: CND_IDS.py::CND_IDS.fit() (line 195): old_models.append(deepclone(self))
        """
        frozen = deepcopy(model).cpu()
        frozen.eval()
        for p in frozen.parameters():
            p.requires_grad_(False)
        self.old_models.append(frozen)
