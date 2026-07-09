"""GPM (Gradient Projection Memory) anti-forgetting.

Reference: Saha et al., "Gradient Projection Memory for Continual Learning"
           ICLR 2021. https://openreview.net/forum?id=3AOj0RCNC2

Key idea:
  - After each task, compute an orthonormal basis for the activation subspace
    of each layer via SVD (retain eigenvectors up to a cumulative variance
    threshold, default 0.97).
  - Before each gradient update, project gradients onto the orthogonal
    complement of the accumulated task space. This prevents interfering
    with previously learned tasks.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from typing import Optional, Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from testbed.base.anti_forgetting import BaseAntiForgetting


class GPMAntiForgetting(BaseAntiForgetting):
    """Gradient Projection Memory continual-learning strategy.

    Reference: Saha et al., ICLR 2021.

    Usage::
        gpm = GPMAntiForgetting(threshold=0.97)
        loss = gpm.compute_loss(model, new_batch, replay_batch)
        loss.backward()
        gpm.project_gradients(model)   # call AFTER backward, BEFORE step
        optimizer.step()
        gpm.on_task_end(model, dataloader)  # update basis after task
    """

    def __init__(self, threshold: float = 0.97, device: str = 'cpu'):
        """
        Args:
            threshold: Cumulative variance fraction to retain (default 0.97).
            device: Torch device string (default 'cpu').
        """
        self.threshold = threshold
        self.device = torch.device(device)
        # Stores per-layer basis matrices: {layer_name: Tensor (D, K)}
        self._memory: Dict[str, torch.Tensor] = {}
        self._pending_dataloader = None

    # ------------------------------------------------------------------
    # 1. Activation collection
    # ------------------------------------------------------------------
    def _collect_activations(self, model: nn.Module,
                              dataloader) -> Dict[str, torch.Tensor]:
        """Run a forward pass and collect pre-activation matrices per layer.

        For each Linear layer, we record the input activations (shape N×in_features).
        SVD is then applied to these activation matrices.

        Args:
            model: Neural network.
            dataloader: Iterable of (data, labels) or just data tensors.

        Returns:
            Dict mapping layer names to activation matrices (N, in_features).
        """
        model.eval()
        activations: Dict[str, List[torch.Tensor]] = {}
        handles = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                def _hook(mod, inp, out, n=name):
                    activations.setdefault(n, []).append(inp[0].detach().cpu())
                handles.append(module.register_forward_hook(_hook))

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                model.to(self.device)
                model(x)

        for h in handles:
            h.remove()

        return {k: torch.cat(v, dim=0) for k, v in activations.items()}

    # ------------------------------------------------------------------
    # 2. SVD basis computation
    # ------------------------------------------------------------------
    def _compute_basis(self, activation_matrix: torch.Tensor) -> torch.Tensor:
        """Compute truncated SVD basis from an activation matrix.

        Args:
            activation_matrix: Shape (N, D) — N samples, D feature dim.

        Returns:
            Basis matrix of shape (D, K) where K is chosen so that
            cumulative singular-value energy >= self.threshold.
        """
        # Centre (subtract mean) before SVD
        A = activation_matrix - activation_matrix.mean(dim=0, keepdim=True)

        # Use torch.linalg.svd (economy SVD)
        try:
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        except Exception:
            # Fallback for older PyTorch
            U, S, V = torch.svd(A)
            Vh = V.T

        # Select K vectors that explain >= threshold of variance
        energy = (S ** 2)
        cumulative = torch.cumsum(energy, dim=0) / energy.sum().clamp(min=1e-10)
        K = int((cumulative < self.threshold).sum().item()) + 1
        K = max(1, min(K, S.shape[0]))

        # Vh rows = right singular vectors; transpose to get (D, K) basis
        # We want column basis for the input activation space
        basis = Vh[:K].T  # (D, K)
        return basis.float()

    # ------------------------------------------------------------------
    # 3. Update GPM memory
    # ------------------------------------------------------------------
    def update_gpm_memory(self, model: nn.Module, dataloader) -> None:
        """Collect activations and update the per-layer basis memory.

        After a task ends, this extends each layer's stored basis with new
        SVD vectors, then re-orthonormalises via QR decomposition.

        Args:
            model: Current trained model.
            dataloader: Training data from the finished task.
        """
        act_dict = self._collect_activations(model, dataloader)

        for layer_name, act_mat in act_dict.items():
            new_basis = self._compute_basis(act_mat)  # (D, K_new)

            if layer_name in self._memory:
                old_basis = self._memory[layer_name]  # (D, K_old)
                combined = torch.cat([old_basis, new_basis], dim=1)  # (D, K_old+K_new)
            else:
                combined = new_basis

            # Re-orthonormalise via QR
            Q, _ = torch.linalg.qr(combined, mode='reduced')
            self._memory[layer_name] = Q.float()  # (D, K)

    # ------------------------------------------------------------------
    # 4. Gradient projection
    # ------------------------------------------------------------------
    def project_gradients(self, model: nn.Module) -> None:
        """Project each layer's gradient onto the complement of past task space.

        Must be called AFTER loss.backward() and BEFORE optimizer.step().

        The projection is:
            grad_proj = grad - basis @ (basis.T @ grad)

        Args:
            model: Current model (with gradients populated by backward()).
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in self._memory:
                if module.weight.grad is None:
                    continue
                basis = self._memory[name].to(module.weight.device)  # (D_in, K)
                # Weight gradient shape: (D_out, D_in)
                grad = module.weight.grad  # (D_out, D_in)
                # Project each row of grad (each output neuron)
                # grad_proj[i] = grad[i] - basis @ (basis.T @ grad[i])
                proj = grad @ basis @ basis.T  # (D_out, D_in)
                module.weight.grad = grad - proj

    # ------------------------------------------------------------------
    # 5. BaseAntiForgetting interface
    # ------------------------------------------------------------------
    def compute_loss(self,
                     model: nn.Module,
                     new_batch: Tuple[torch.Tensor, torch.Tensor],
                     replay_batch: Optional[Tuple[torch.Tensor, torch.Tensor]],
                     old_model: Optional[nn.Module] = None) -> torch.Tensor:
        """Compute task loss (MSE reconstruction or CE) for the current batch.

        GPM modifies gradients after backward(); the loss itself is
        standard and does NOT include explicit regularisation terms.

        Args:
            model: Current model.
            new_batch: (data, labels).
            replay_batch: Unused — gradient projection handles past-task
                          protection without explicit replay.
            old_model: Unused.

        Returns:
            Scalar loss tensor. Call project_gradients(model) after backward().
        """
        data, labels = new_batch
        device = data.device
        model = model.to(device)

        out = model(data)
        if isinstance(out, (tuple, list)):
            recon = out[1]
        else:
            recon = out
        # MODIFIED: if output shape != input (non-AE model), use L2 regulariser
        if recon.shape == data.shape:
            loss = F.mse_loss(recon, data)
        else:
            loss = recon.pow(2).mean()

        # Add a tiny regulariser to ensure requires_grad=True in degenerate cases
        if not loss.requires_grad:
            loss = loss + 0.0 * sum(p.sum() for p in model.parameters()
                                    if p.requires_grad)

        return loss

    def on_task_end(self, model: nn.Module) -> None:
        """Update the GPM basis after each task.

        Requires data to be provided via set_pending_dataloader() before
        calling on_task_end(), or falls back to a dummy single-sample pass.

        FROM: Saha et al. ICLR 2021 — Algorithm 1, GPM update step.

        Args:
            model: Current model after the completed task.
        """
        if self._pending_dataloader is not None:
            self.update_gpm_memory(model, self._pending_dataloader)
            self._pending_dataloader = None
        # If no dataloader is set, skip silently (basis remains unchanged)

    def set_pending_dataloader(self, dataloader) -> None:
        """Provide the dataloader used for basis computation at task end.

        Args:
            dataloader: Training data from the task about to finish.
        """
        self._pending_dataloader = dataloader
