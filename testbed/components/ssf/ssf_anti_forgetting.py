"""SSF Anti-Forgetting — Learning without Forgetting (LwF) with InfoNCE.

FROM: SSF-Strategic-Selection-and-Forgetting/ssf.py (line 262-334)
      SSF-Strategic-Selection-and-Forgetting/utils.py::InfoNCELoss (line 458-492)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from testbed.base.anti_forgetting import BaseAntiForgetting


# FROM: utils.py::InfoNCELoss (line 458-492)
class _InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss that separates normal vs. abnormal samples."""

    def __init__(self, temperature: float = 0.02):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_mask = (torch.ones_like(mask)
                       - torch.eye(batch_size, device=mask.device))
        logits_wo_diag = logits * logits_mask

        normal_mask = (labels == 0).squeeze()
        abnormal_mask = (labels > 0).squeeze()

        if normal_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=features.device)

        logits_normal = logits_wo_diag[normal_mask]
        logits_nn = logits_normal[:, normal_mask]
        logits_na = logits_normal[:, abnormal_mask]

        if logits_na.numel() == 0:
            # All samples are normal — no negatives, return 0
            return torch.tensor(0.0, requires_grad=True, device=features.device)

        sum_neg = torch.sum(torch.exp(logits_na), dim=1, keepdim=True)
        denom = torch.exp(logits_nn) + sum_neg
        log_probs = logits_nn - torch.log(denom.clamp(min=1e-12))
        loss = -log_probs * self.temperature
        return loss


class SSFAntiForgetting(BaseAntiForgetting):
    """LwF distillation (no drift) + weighted InfoNCE.

    FROM: SSF-Strategic-Selection-and-Forgetting/ssf.py (line 262-334)

    When drift is detected (replay_batch is None):
        loss = weighted_InfoNCE only  (fast adaptation)
    When no drift (replay_batch provided):
        loss = weighted_InfoNCE + lwf_lambda * MSE(student_output, teacher_output)
    """

    def __init__(self, lwf_lambda: float = 0.5, temperature: float = 0.02,
                 new_sample_weight: float = 100.0):
        """
        Args:
            lwf_lambda: Weight of the distillation loss (default 0.5).
            temperature: InfoNCE temperature (default 0.02).
            new_sample_weight: Loss scaling for newly added samples (default 100.0).
        """
        self.lwf_lambda = lwf_lambda
        self.temperature = temperature
        self.new_sample_weight = new_sample_weight
        self.teacher: Optional[nn.Module] = None
        self._criterion = _InfoNCELoss(temperature)

    def compute_loss(self,
                     model: nn.Module,
                     new_batch: Tuple[torch.Tensor, torch.Tensor],
                     replay_batch: Optional[Tuple[torch.Tensor, torch.Tensor]],
                     old_model: Optional[nn.Module] = None) -> torch.Tensor:
        """Compute weighted InfoNCE + optional LwF distillation.

        FROM: ssf.py (line 262-334)

        Args:
            model: Current model. Expected to return (features, recon_vec)
                   or (features, recon_vec, logits).
            new_batch: (data, labels) — newly selected samples this round.
            replay_batch: (data, labels) from memory buffer. If None, drift
                          is assumed → distillation is skipped.
            old_model: Unused (teacher stored internally); kept for API compat.

        Returns:
            Scalar loss tensor.
        """
        data, labels = new_batch
        device = data.device
        model = model.to(device)
        self._criterion = self._criterion.to(device) if hasattr(self._criterion, 'to') else self._criterion

        drift_mode = (replay_batch is None)

        # Build training batch: memory + new samples
        if replay_batch is not None and replay_batch[0] is not None:
            mem_data, mem_labels = replay_batch
            mem_data = mem_data.to(device)
            mem_labels = mem_labels.to(device)
            inputs = torch.cat([mem_data, data], dim=0)
            combined_labels = torch.cat([mem_labels, labels], dim=0)
            # new_sample_mask: 1 for new samples, 0 for replay
            new_mask = torch.cat([
                torch.zeros(len(mem_data), device=device),
                torch.ones(len(data), device=device)
            ])
        else:
            inputs = data
            combined_labels = labels
            new_mask = torch.ones(len(data), device=device)

        # Forward pass
        out = model(inputs)
        if isinstance(out, (tuple, list)):
            recon_vec = out[1]
        else:
            recon_vec = out

        # Weighted InfoNCE loss
        # FROM: ssf.py (line 280-288)
        normal_new_mask = new_mask[combined_labels == 0]
        con_loss = self._criterion(recon_vec, combined_labels)
        if isinstance(con_loss, torch.Tensor) and con_loss.dim() > 0:
            weight = (1 - normal_new_mask) + normal_new_mask * self.new_sample_weight
            if len(weight) == len(con_loss):
                weighted_loss = (con_loss * weight).mean()
            else:
                weighted_loss = con_loss.mean()
        else:
            weighted_loss = con_loss if isinstance(con_loss, torch.Tensor) else torch.tensor(0.0)

        if drift_mode or self.teacher is None:
            # FROM: ssf.py (line 262-291) — drift: no distillation
            return weighted_loss

        # FROM: ssf.py (line 323-334) — no drift: add LwF distillation
        teacher = (old_model or self.teacher).to(device)
        teacher.eval()
        with torch.no_grad():
            t_out = teacher(inputs)
            teacher_recon = t_out[1] if isinstance(t_out, (tuple, list)) else t_out

        distillation_loss = F.mse_loss(recon_vec, teacher_recon)
        return weighted_loss + self.lwf_lambda * distillation_loss

    # FROM: ssf.py (line 336) — teacher_model.load_state_dict(model.state_dict())
    def on_task_end(self, model: nn.Module) -> None:
        """Update the teacher model with the current student weights.

        FROM: ssf.py line 336: teacher_model.load_state_dict(model.state_dict())
        """
        self.teacher = deepcopy(model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
