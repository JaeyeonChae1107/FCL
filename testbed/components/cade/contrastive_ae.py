"""CADE Contrastive AutoEncoder — CADE 원 논문 근거.

CADE/cade/autoencoder.py:52-107 — 대칭 encoder/decoder, 마지막 encoder 레이어는
선형(활성함수 없음). autoencoder.py:210-232 contrastive loss:
  L_con = is_same*dist + (1-is_same)*relu(margin-dist), dist=L2 거리
  total = contrastive_lambda * L_con + MSE(recon, x)
기본 margin=10.0, contrastive_lambda=0.1 (CADE/cade/utils.py:68-73).

PRD 12.1절 — 이 클래스는 Track A 메인 분류기와 완전히 독립된 "사설(private)"
encoder다. CADEDriftDetector가 이를 내부에 소유하며, 메인 모델의 z/logit을
가져다 쓰지 않는다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveAutoEncoder(nn.Module):
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

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


def contrastive_loss(z: torch.Tensor, labels: torch.Tensor, margin: float = 10.0) -> torch.Tensor:
    """CADE autoencoder.py:210-232. 배치를 절반으로 나눠 anchor/비교 쌍을 만든다."""
    n = z.shape[0]
    half = n // 2
    if half == 0:
        return z.sum() * 0.0
    left, right = z[:half], z[half:2 * half]
    left_labels, right_labels = labels[:half], labels[half:2 * half]
    dist = torch.sqrt(((left - right) ** 2).sum(dim=1) + 1e-10)
    is_same = (left_labels == right_labels).float()
    loss = is_same * dist + (1 - is_same) * F.relu(margin - dist)
    return loss.mean()


def train_step(cae: ContrastiveAutoEncoder, optimizer: torch.optim.Optimizer,
               data: torch.Tensor, labels: torch.Tensor,
               margin: float = 10.0, lam: float = 0.1) -> float:
    cae.train()
    optimizer.zero_grad()
    z, x_hat = cae(data)
    recon_loss = F.mse_loss(x_hat, data)
    con_loss = contrastive_loss(z, labels, margin)
    loss = lam * con_loss + recon_loss
    loss.backward()
    optimizer.step()
    return float(loss.item())
