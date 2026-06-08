"""SSF Anti-Forgetting — weighted BCE + LwF distillation.

FROM: Zhang et al. "SSF: Strategic Selection and Forgetting for Federated
      Continual Learning" INFOCOM 2025.
      ssf.py (line 262-334)

Loss rules:
  No drift  →  L_total = L_task + λ · L_reg
  Drift     →  L_total = L_task   (fast adaptation; skip regularisation)

  L_task : weighted binary cross-entropy on classifier logit
           new samples receive new_sample_weight; replay gets weight 1.0
  L_reg  : MSE(z_current, z_teacher)  LwF latent-space distillation
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from testbed.base.anti_forgetting import BaseAntiForgetting


class SSFAntiForgetting(BaseAntiForgetting):
    """Weighted binary cross-entropy + optional LwF distillation.

    FROM: SSF-Strategic-Selection-and-Forgetting/ssf.py (line 262-334)
    """

    def __init__(self, lwf_lambda: float = 0.5,
                 new_sample_weight: float = 100.0):
        """
        Args:
            lwf_lambda:        Weight of the LwF regularisation term (default 0.5).
            new_sample_weight: Loss multiplier for newly selected samples
                               (replay samples get weight 1.0). Default 100.0.
        """
        self.lwf_lambda = lwf_lambda
        self.new_sample_weight = new_sample_weight
        self.teacher: Optional[nn.Module] = None

    def compute_loss(self,
                     model: nn.Module,
                     new_batch: Tuple[torch.Tensor, torch.Tensor],
                     replay_batch: Optional[Tuple[torch.Tensor, torch.Tensor]],
                     old_model: Optional[nn.Module] = None) -> torch.Tensor:
        """Weighted BCE + optional LwF.

        FROM: ssf.py (line 262-334)

        Args:
            model: Current model. Expected forward: x → (z, x_hat, logit).
            new_batch: (data, labels) — newly selected samples this round.
            replay_batch: (mem_data, mem_labels) or None.
                          None signals drift mode → skip regularisation.
            old_model: Explicit teacher override (falls back to self.teacher).

        Returns:
            Scalar loss tensor.
        """
        data, labels = new_batch
        device = data.device
        model = model.to(device)

        drift_mode = (replay_batch is None)

        # Build combined batch (replay + new)
        if replay_batch is not None and replay_batch[0] is not None:
            mem_data = replay_batch[0].to(device)
            mem_labels = replay_batch[1].to(device)
            inputs = torch.cat([mem_data, data], dim=0)
            combined_labels = torch.cat([mem_labels, labels], dim=0)
            n_mem = len(mem_data)
        else:
            inputs = data
            combined_labels = labels
            n_mem = 0

        # Forward
        out = model(inputs)
        z = out[0] if isinstance(out, (tuple, list)) else out
        # logit at position 2; fall back to a linear projection of z if absent
        if isinstance(out, (tuple, list)) and len(out) >= 3:
            logit = out[2].squeeze(-1)          # (N,)
        else:
            logit = z[:, 0]                     # fallback: first latent dim

        # Weighted binary cross-entropy (FROM: ssf.py line 280-288)
        # Only apply differential weighting when both replay and new samples exist.
        # When n_mem=0 (no replay yet), all samples are "new" — use uniform weight 1.0
        # to avoid 100× loss inflation in early rounds.
        weights = torch.ones(len(inputs), device=device)
        if 0 < n_mem < len(inputs):
            weights[n_mem:] = self.new_sample_weight

        l_task = F.binary_cross_entropy_with_logits(
            logit, combined_labels.float(), weight=weights
        )

        if drift_mode or self.teacher is None:
            # Drift mode: fast adaptation without distillation
            # FROM: ssf.py line 262-291
            return l_task

        # No-drift mode: add LwF regularisation (FROM: ssf.py line 323-334)
        teacher = (old_model or self.teacher).to(device)
        teacher.eval()
        with torch.no_grad():
            t_out = teacher(inputs)
            teacher_z = t_out[0] if isinstance(t_out, (tuple, list)) else t_out

        l_reg = F.mse_loss(z, teacher_z.to(device))
        return l_task + self.lwf_lambda * l_reg

    def on_task_end(self, model: nn.Module) -> None:
        """Update teacher with current student weights.

        FROM: ssf.py line 336: teacher_model.load_state_dict(model.state_dict())
        """
        self.teacher = deepcopy(model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
