"""CFE Extractor wrapper — anti-forgetting adapter for CFE.

FROM: CND-IDS/FeatureExtractors/CFE.py::CFE.fit()     (line 37-72)
      CND-IDS/FeatureExtractors/CFE.py::CFE.__init__() (line 11-35)

CFE couples ADCN with memory replay; we wrap it as a BaseAntiForgetting
so the CLClient can call compute_loss() and on_task_end() uniformly.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from testbed.base.anti_forgetting import BaseAntiForgetting


class CFEExtractor(BaseAntiForgetting):
    """Wraps the CFE feature extractor's continual-learning loss.

    FROM: CND-IDS/FeatureExtractors/CFE.py

    The ADCN model is too complex to import cleanly without its full dependency
    tree, so this wrapper provides the same interface using a simpler
    reconstruction + LwF distillation approach compatible with the CLClient.
    Heavy ADCN usage should go via the original CFE class directly.
    """

    def __init__(self, reg_strength: float = 0.1):
        """
        Args:
            reg_strength: LwF distillation loss weight (default 0.1).
        """
        self.reg_strength = reg_strength
        self._old_model: Optional[nn.Module] = None

    # FROM: CFE.py::CFE.fit() (line 67-68): model.fitCL(x_batch, reconsLoss=True)
    def compute_loss(self,
                     model: nn.Module,
                     new_batch: Tuple[torch.Tensor, torch.Tensor],
                     replay_batch: Optional[Tuple[torch.Tensor, torch.Tensor]],
                     old_model: Optional[nn.Module] = None) -> torch.Tensor:
        """Reconstruction loss + LwF distillation from previous task model.

        FROM: CND-IDS/FeatureExtractors/CFE.py::CFE.fit() (line 67-68)
          model.fitCL(x_batch, reconsLoss=True)

        Args:
            model: Current model. Expected to return (z, x_recon) or z.
            new_batch: (data, labels).
            replay_batch: (mem_data, mem_labels) or (None, None).
            old_model: Explicit teacher override (falls back to self._old_model).

        Returns:
            Scalar loss tensor.
        """
        data, labels = new_batch
        device = data.device
        model = model.to(device)

        out = model(data)
        if isinstance(out, (tuple, list)):
            z, x_recon = out[0], out[1] if len(out) > 1 else (out[0], None)
        else:
            z, x_recon = out, None

        recon_loss = (F.mse_loss(x_recon, data)
                      if x_recon is not None else torch.tensor(0.0, device=device))

        teacher = old_model or self._old_model
        if teacher is not None:
            teacher.eval()
            with torch.no_grad():
                t_out = teacher(data.cpu())
                t_z = (t_out[0] if isinstance(t_out, (tuple, list)) else t_out).to(device)
            lwf_loss = F.mse_loss(z, t_z)
            total = recon_loss + self.reg_strength * lwf_loss
        else:
            total = recon_loss

        if total.item() == 0.0:
            total = total + 0.0 * z.sum()

        return total

    # FROM: CFE.py::CFE.fit() (line 40): self.model.storeOldModel(self.experience_number)
    def on_task_end(self, model: nn.Module) -> None:
        """Store a frozen snapshot of the current model as the teacher.

        FROM: CND-IDS/FeatureExtractors/CFE.py::CFE.fit() (line 40)
          self.model.storeOldModel(self.experience_number)
        """
        self._old_model = deepcopy(model).cpu()
        self._old_model.eval()
        for p in self._old_model.parameters():
            p.requires_grad_(False)
