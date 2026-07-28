"""모델 계약 — PRD 11절.

BaseCLModel.forward(x) -> (z, x_hat, logit):
  z     : (N, latent_dim)  잠재 표현. anomaly_scorer, drift_detector 입력
  x_hat : (N, input_dim)   재구성. Track B의 anti_forgetting(cndids)용
  logit : (N, 1)           이진 판별 로짓. Track A의 anti_forgetting/drift_detector(ssf) 입력
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class BaseCLModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...


class FCLAutoEncoder(BaseCLModel):
    """Track A/B 공용. Track A는 x_hat을 손실 계산에 쓰지 않을 뿐, 클래스는 공유한다."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.classifier = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        logit = self.classifier(z)
        return z, x_hat, logit
