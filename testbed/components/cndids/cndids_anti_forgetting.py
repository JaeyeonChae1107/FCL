"""CND-IDS Anti-Forgetting — LwF MSE distillation + metric loss.

FROM: CND-IDS/FeatureExtractors/CND_IDS.py::CND_IDS.LwFloss()      (line 54-69)
      CND-IDS/FeatureExtractors/CND_IDS.py::CND_IDS.fit()            (line 100-195)
      CND-IDS/FeatureExtractors/CND_IDS.py::CND_IDS.metric_loss()    (line 76-78)
      CND-IDS/FeatureExtractors/CND_IDS.py::CND_IDS.reconstruction_loss() (line 71-74)
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
    """Combines LwF MSE distillation + reconstruction loss.

    FROM: CND-IDS/FeatureExtractors/CND_IDS.py::CND_IDS.LwFloss() (line 54-69)

    At task end, saves a frozen copy of the current model; subsequent rounds
    use all saved old models as teachers (multi-task LwF).
    """

    def __init__(self, reg_strength: float = 0.1,
                 lwf_strength: float = 0.1):
        """
        Args:
            reg_strength: Reconstruction loss weight (default 0.1).
            lwf_strength: LwF distillation loss weight (default 0.1).
        """
        self.reg_strength = reg_strength
        self.lwf_strength = lwf_strength
        self.old_models: List[nn.Module] = []

    # FROM: CND_IDS.py::CND_IDS.LwFloss() (line 54-69)
    def _lwf_loss(self, model: nn.Module,
                  data: torch.Tensor) -> torch.Tensor:
        """Compute LwF distillation loss against all saved old models.

        FROM: CND-IDS/FeatureExtractors/CND_IDS.py::CND_IDS.LwFloss() (line 54-69)
        loss = sum(reg_strength * MSE(current_output, old_model(data)))
        """
        if not self.old_models:
            return torch.tensor(0.0, device=data.device)

        criterion = nn.MSELoss()
        total = torch.tensor(0.0, device=data.device)
        out = model(data)
        current_out = out[0] if isinstance(out, (tuple, list)) else out

        for old_model in self.old_models:
            old_model.eval()
            with torch.no_grad():
                o = old_model(data.cpu())
                old_out = (o[0] if isinstance(o, (tuple, list)) else o).to(data.device)
            total = total + self.reg_strength * criterion(current_out, old_out)

        return total

    def compute_loss(self,
                     model: nn.Module,
                     new_batch: Tuple[torch.Tensor, torch.Tensor],
                     replay_batch: Optional[Tuple[torch.Tensor, torch.Tensor]],
                     old_model: Optional[nn.Module] = None) -> torch.Tensor:
        """Reconstruction + LwF distillation loss.

        FROM: CND-IDS/FeatureExtractors/CND_IDS.py::CND_IDS.fit() (line 156-165)
        loss = metric_loss + LwF_loss * lwf_strength + recon_loss * reg_strength

        Args:
            model: Current model. Expected to return (z, x_recon) or z.
            new_batch: (data, labels) for the current batch.
            replay_batch: Unused (LwF uses stored old models instead).
            old_model: Optional explicit teacher override.

        Returns:
            Scalar loss tensor.
        """
        data, labels = new_batch
        device = data.device
        model = model.to(device)

        out = model(data)
        if isinstance(out, (tuple, list)):
            z, x_recon = out[0], out[1]
        else:
            z = out
            x_recon = None

        # Reconstruction loss
        recon_loss = (F.mse_loss(x_recon, data)
                      if x_recon is not None else torch.tensor(0.0, device=device))

        # LwF distillation loss
        # FROM: CND_IDS.py::LwFloss() (line 54-69)
        lwf_loss = self._lwf_loss(model, data)

        total = (self.reg_strength * recon_loss
                 + self.lwf_strength * lwf_loss)

        # Ensure gradient flows even when all sub-losses are zero
        if total.item() == 0.0:
            total = total + 0.0 * z.sum()

        return total

    # FROM: CND_IDS.py::CND_IDS.fit() (line 195) — old_models.append(deepclone(self))
    def on_task_end(self, model: nn.Module) -> None:
        """Snapshot current model into old_models list.

        FROM: CND-IDS/FeatureExtractors/CND_IDS.py (line 195):
          self.old_models.append(deepclone(self))
        """
        frozen = deepcopy(model).cpu()
        frozen.eval()
        for p in frozen.parameters():
            p.requires_grad_(False)
        self.old_models.append(frozen)
