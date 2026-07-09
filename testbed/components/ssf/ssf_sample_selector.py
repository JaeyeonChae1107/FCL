"""SSF Sample Selector — KL-div mask optimisation.

FROM: SSF-Strategic-Selection-and-Forgetting/utils.py
  ::optimize_old_mask()   (line 109-145)
  ::optimize_new_mask()   (line 147-190)
  ::select_and_update_representative_samples()      (line 192-257)
  ::select_and_update_representative_samples_when_drift() (line 259-388)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from typing import List
import torch
import torch.nn.functional as F
from testbed.base.sample_selector import BaseSampleSelector


class SSFSampleSelector(BaseSampleSelector):
    """Selects representative new samples via KL-divergence mask optimisation.

    FROM: SSF-Strategic-Selection-and-Forgetting/utils.py
    """

    def __init__(self,
                 mask_threshold: float = 0.5,
                 num_bins: int = 10,
                 opt_steps: int = 100,
                 new_lr: float = 50.0,
                 old_lr: float = 1.0):
        """
        Args:
            mask_threshold: Mask value >= this counts as "selected" (default 0.5).
            num_bins: Number of histogram bins for KL divergence (default 10).
            opt_steps: Gradient-descent steps for mask optimisation (default 100).
            new_lr: Learning rate for new-data mask M_t (default 50.0).
            old_lr: Learning rate for old-data mask M_c (default 1.0).
        """
        self.mask_threshold = mask_threshold
        self.num_bins = num_bins
        self.opt_steps = opt_steps
        self.new_lr = new_lr
        self.old_lr = old_lr

    def select(self, new_data: torch.Tensor,
               new_labels: torch.Tensor,
               label_budget: int,
               drift_score: float = 0.0) -> List[int]:
        """Select up to label_budget representative new-data indices.

        Uses M_t mask (optimised to align new distribution with reference).
        drift_score > 0 triggers _select_with_drift strategy (currently the
        same algorithm; override to differentiate).

        Args:
            new_data: Incoming samples, shape (N, D) or (N,) score vector.
            new_labels: Labels, shape (N,).
            label_budget: Maximum number of samples to select.
            drift_score: KS statistic or similar (> 0 = drift present).

        Returns:
            List of int indices into new_data, length <= label_budget.
        """
        # Reduce to 1-D score if multi-dim
        if new_data.dim() > 1:
            scores = new_data.mean(dim=1)
        else:
            scores = new_data.float()

        if drift_score > 0:
            return self._select_with_drift(scores, label_budget)
        return self._select_without_drift(scores, label_budget)

    # FROM: utils.py::optimize_new_mask() — KL-div mask optimisation
    def _optimise_mask(self, scores: torch.Tensor, lr: float,
                       init_range: str = '0-0.5') -> torch.Tensor:
        """Optimise a soft mask M so that M-weighted histogram ≈ target histogram.

        FROM: SSF-Strategic-Selection-and-Forgetting/utils.py::optimize_new_mask()
        """
        device = scores.device
        delta = 1e-4

        # Initialise mask in [0, 0.5]
        M = torch.nn.Parameter(torch.rand(scores.size(0), device=device) * 0.5,
                                requires_grad=True)
        optimizer = torch.optim.SGD([M], lr=lr)

        bin_edges = torch.linspace(0., 1., self.num_bins + 1, device=device)
        target_hist = torch.histc(scores, bins=self.num_bins, min=0., max=1.)
        target_hist = target_hist / target_hist.sum().clamp(min=1e-10)

        for _ in range(self.opt_steps):
            with torch.no_grad():
                M.clamp_(delta, 1 - delta)
            optimizer.zero_grad()

            obs = torch.zeros(self.num_bins, device=device)
            for i in range(self.num_bins):
                mask_i = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
                obs[i] = (M * mask_i.float()).sum() / M.sum().clamp(min=1e-10)

            obs = obs / obs.sum().clamp(min=1e-10)
            tgt = target_hist.clamp(min=1e-10)
            loss = F.kl_div(obs.clamp(min=1e-10).log(), tgt, reduction='sum')
            loss.backward()
            optimizer.step()

        return M.detach()

    # FROM: utils.py::select_and_update_representative_samples() (line 192-257)
    def _select_without_drift(self, scores: torch.Tensor,
                               label_budget: int) -> List[int]:
        M = self._optimise_mask(scores, lr=self.new_lr)
        selected = (M >= self.mask_threshold).nonzero(as_tuple=True)[0]
        if len(selected) >= label_budget:
            # Take top-scored
            top_idx = M[selected].topk(label_budget).indices
            return selected[top_idx].tolist()
        # Fallback: random padding
        all_idx = torch.arange(len(scores))
        not_sel = all_idx[~torch.isin(all_idx, selected)]
        extra_n = min(label_budget - len(selected),
                      len(not_sel))
        if extra_n > 0:
            extra = not_sel[torch.randperm(len(not_sel))[:extra_n]]
            selected = torch.cat([selected, extra])
        return selected.tolist()

    # FROM: utils.py::select_and_update_representative_samples_when_drift() (line 259-388)
    def _select_with_drift(self, scores: torch.Tensor,
                            label_budget: int) -> List[int]:
        # Same logic as without drift; separate for future specialisation
        return self._select_without_drift(scores, label_budget)
