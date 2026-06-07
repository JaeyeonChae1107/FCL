"""CLClient — pluggable continual-learning client.

Orchestrates: Drift Detector → Sample Selector → Memory Manager
              → Anti-Forgetting → Anomaly Scorer
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from testbed.pipeline.component_registry import build


class CLClient:
    """Federated / continual learning client with swappable components.

    Config example::

        config = {
            "drift_detector":  {"name": "ssf", "drift_threshold": 0.05},
            "sample_selector": {"name": "ssf", "mask_threshold": 0.5},
            "memory_manager":  {"name": "ssf", "max_size": 1000},
            "anti_forgetting": {"name": "lwf_ssf", "lwf_lambda": 0.5},
            "anomaly_scorer":  {"name": "pca"},
            "label_budget": 50,
            "lr": 1e-3,
        }
        client = CLClient(model=my_model, config=config, device='cpu')
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any],
                 device: str = 'cpu'):
        """
        Args:
            model: PyTorch model. Expected forward: x → (z, recon) or z.
            config: Dict with keys for each component slot plus 'label_budget'
                    and 'lr'.
            device: Torch device string.
        """
        self.model = model.to(device)
        self.device = torch.device(device)
        self.config = config
        self.label_budget: int = config.get('label_budget', 50)
        self.lr: float = config.get('lr', 1e-3)

        # Build components from registry
        self.drift_detector = build(
            'drift_detector', **config.get('drift_detector', {'name': 'none'}))
        self.sample_selector = build(
            'sample_selector', **config.get('sample_selector', {'name': 'random'}))
        self.memory_manager = build(
            'memory_manager', **config.get('memory_manager', {'name': 'none'}))
        self.anti_forgetting = build(
            'anti_forgetting', **config.get('anti_forgetting', {'name': 'none'}))
        self.anomaly_scorer = build(
            'anomaly_scorer', **config.get('anomaly_scorer', {'name': 'pca'}))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self._anomaly_threshold: float = 0.5
        self._round: int = 0

    # ------------------------------------------------------------------
    def update(self, new_data: torch.Tensor,
               new_labels: torch.Tensor) -> Dict[str, Any]:
        """Run one continual-learning round.

        Steps:
          1. Drift detection (against memory buffer)
          2. Sample selection (label_budget)
          3. Memory update
          4. Replay batch retrieval
          5. Anti-forgetting loss + backward
          6. Gradient projection (GPM, if applicable)
          7. Optimizer step
          8. on_task_end() hook

        Args:
            new_data: Incoming data batch. Shape (N, D).
            new_labels: Corresponding labels. Shape (N,).

        Returns:
            Dict with keys 'loss', 'drift', 'drift_score', 'round'.
        """
        new_data = new_data.to(self.device)
        new_labels = new_labels.to(self.device)
        self._round += 1

        # 1. Drift detection
        buf_data, _ = self.memory_manager.get_buffer()
        buf_ref = buf_data.to(self.device) if buf_data is not None else None
        drift_score = self.drift_detector.get_drift_score(new_data, buf_ref)
        drift_detected = self.drift_detector.detect(new_data, buf_ref)

        # 2. Sample selection
        sel_idx = self.sample_selector.select(
            new_data, new_labels, self.label_budget, drift_score)
        if not sel_idx:
            sel_idx = list(range(min(self.label_budget, len(new_data))))
        sel_data = new_data[sel_idx]
        sel_labels = new_labels[sel_idx]

        # 3. Memory update
        self.memory_manager.update(sel_data, sel_labels, drift_detected)

        # 4. Replay batch
        replay_batch = None
        if self.memory_manager.size() > 0:
            r_data, r_labels = self.memory_manager.get_replay_batch(
                batch_size=self.label_budget)
            if r_data is not None:
                replay_batch = (r_data.to(self.device),
                                r_labels.to(self.device))

        # 5. Loss computation
        self.model.train()
        self.optimizer.zero_grad()
        new_batch = (sel_data, sel_labels)
        loss = self.anti_forgetting.compute_loss(
            self.model, new_batch, replay_batch)

        # 6. Backward + optional gradient projection
        loss.backward()
        if hasattr(self.anti_forgetting, 'project_gradients'):
            self.anti_forgetting.project_gradients(self.model)

        # 7. Optimizer step
        self.optimizer.step()

        # 8. Task-end hook
        self.anti_forgetting.on_task_end(self.model)

        return {
            'loss': loss.item(),
            'drift': drift_detected,
            'drift_score': drift_score,
            'round': self._round,
        }

    # ------------------------------------------------------------------
    def fit_anomaly_scorer(self, normal_data: torch.Tensor) -> None:
        """Fit the anomaly scorer on normal (inlier) data.

        Args:
            normal_data: Normal samples. Shape (N, D).
        """
        normal_data = normal_data.to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(normal_data)
            encoded = out[0] if isinstance(out, (tuple, list)) else out
        self.anomaly_scorer.fit(encoded.cpu())

    # ------------------------------------------------------------------
    def infer(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Score and classify a batch of samples.

        Args:
            data: Input samples. Shape (N, D).

        Returns:
            Dict with 'scores' (float, shape N) and 'predictions' (long, shape N).
        """
        data = data.to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(data)
            encoded = out[0] if isinstance(out, (tuple, list)) else out
        encoded = encoded.cpu()
        scores = self.anomaly_scorer.score(encoded)
        preds = self.anomaly_scorer.predict(encoded, self._anomaly_threshold)
        return {'scores': scores, 'predictions': preds}

    # ------------------------------------------------------------------
    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """Return model state dict for FL aggregation.

        Returns:
            state_dict compatible with nn.Module.load_state_dict().
        """
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def load_model_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load a global model received from the FL server.

        Args:
            state_dict: Model weights dict from the server.
        """
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in state_dict.items()})

    def set_anomaly_threshold(self, threshold: float) -> None:
        """Update the decision threshold for anomaly classification.

        Args:
            threshold: New threshold value.
        """
        self._anomaly_threshold = threshold
