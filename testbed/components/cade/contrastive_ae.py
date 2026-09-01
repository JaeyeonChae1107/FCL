"""CADE Contrastive AutoEncoder — CADE 원 논문 근거.

CADE/cade/autoencoder.py:52-107 — 대칭 encoder/decoder, 마지막 encoder 레이어는
선형(활성함수 없음). autoencoder.py:210-232 contrastive loss:
  L_con = is_same*dist + (1-is_same)*relu(margin-dist), dist=L2 거리
  total = contrastive_lambda * L_con + MSE(recon, x)
기본 margin=10.0, contrastive_lambda=0.1 (CADE/cade/utils.py:68-73).

PRD 12.1절 — 이 클래스는 Track A 메인 분류기와 완전히 독립된 "사설(private)"
encoder다. CADEDriftDetector가 이를 내부에 소유하며, 메인 모델의 z/logit을
가져다 쓰지 않는다.

**2026-08-12 발견·수정 — class-aware pairing 누락**: `contrastive_loss()`는
이미 배치를 절반으로 나눠 위치별로 is_same을 계산하지만, 그 배치 자체가
"어떻게 구성되는가"는 별개 문제였다 — 이전에는 무작위 셔플 후 그냥 슬라이싱만
했다. CADE 원문(`cade/data.py:268-345`, `epoch_batches()`)은 배치를 "무작위
anchor 절반" + "각 anchor와 위치별로 짝지어진 비교 절반"으로 구성하고, 비교
절반 중 `similar_ratio`(기본 0.25, `cade/utils.py:70-71`) 비율만큼은 반드시
같은 클래스에서, 나머지는 반드시 다른 클래스에서 뽑는다(index_cls/index_no_cls
기반 강제 pairing) — 매 배치가 최소한의 dissimilar 쌍을 보장받는 설계다.
`_build_paired_batches()`가 이를 이식한다. 라벨 불균형이 심한 라운드(예:
class-incremental 분할에서 실측된 U2R 52건 단독 라운드)에서 무작위 슬라이싱만
쓰면 배치 안에 dissimilar 쌍이 우연히 하나도 없어 margin loss의 gradient
신호가 사라질 수 있는데, 이 pairing이 바로 그 상황을 막기 위한 원문의 핵심
설계였다. 원문은 이중 for-loop(batch × position)로 `np.random.choice`를
호출하지만, 이 테스트베드는 (label, similar 여부) 조합별로 그룹화해 한 번에
`torch.randint`+gather로 처리한다 — 표본 분포(각 위치가 해당 풀에서 복원추출로
독립적으로 뽑힘)는 동일하되 CICIDS2018 규모(라운드당 수십만 건)에서도
감당 가능하도록 벡터화했다. 이번 라운드 데이터에 클래스가 하나뿐이면(공격이
전혀 없는 라운드 등, class-incremental 분할에서 자연 발생) dissimilar 짝을
구성할 데이터 자체가 없으므로 similar 풀로 대체한다(원 논문이 다루지 않는
상황에 대한 테스트베드 자체의 안전한 폴백 — 크래시 대신 "이번 라운드는 전부
같은 클래스"라는 사실을 그대로 반영).
"""

from typing import Dict, Optional, Tuple

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


def build_paired_batches(data: torch.Tensor, labels: torch.Tensor, batch_size: int,
                          similar_ratio: float = 0.25
                          ) -> Tuple[int, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """CADE/cade/data.py:268-345 `epoch_batches()` 이식.

    각 배치는 무작위 anchor 절반(half_size=batch_size//2) + 위치별로 짝지어진
    비교 절반으로 구성된다. 비교 절반 중 앞 num_sim(=batch_size*similar_ratio)
    자리는 anchor와 반드시 같은 클래스에서, 나머지(half_size-num_sim)는 반드시
    다른 클래스에서 뽑는다(원문은 anchor 위치마다 np.random.choice를 개별
    호출하지만, 여기서는 (클래스, similar 여부) 조합별로 묶어 한 번에
    torch.randint+gather로 뽑는다 — 각 위치가 해당 풀에서 독립적으로 복원추출
    되는 분포는 동일하다). 이번 라운드 데이터에 클래스가 하나뿐이면(공격
    없는 라운드 등) dissimilar 풀이 비므로 similar 풀로 대체한다(원문이
    다루지 않는 상황에 대한 테스트베드 자체 폴백).

    Returns:
        (batch_count, paired_data, paired_labels). batch_count==0이면 이번
        라운드는 배치를 구성할 만큼 데이터가 없다는 뜻(원문처럼 그 epoch은
        조용히 건너뛴다 — data/labels는 None).
    """
    n = len(data)
    half_size = batch_size // 2
    if half_size < 1 or n < half_size:
        return 0, None, None
    batch_count = n // half_size
    num_sim = max(0, min(int(batch_size * similar_ratio), half_size))

    device = data.device
    perm = torch.randperm(n, device=device)
    anchor_idx = perm[: batch_count * half_size].view(batch_count, half_size)
    anchor_labels = labels[anchor_idx]

    want_same = torch.zeros(half_size, dtype=torch.bool, device=device)
    want_same[:num_sim] = True
    want_same = want_same.unsqueeze(0).expand(batch_count, half_size)

    unique_labels = [int(c.item()) for c in labels.unique()]
    multi_class = len(unique_labels) > 1
    pool_same: Dict[int, torch.Tensor] = {
        c: (labels == c).nonzero(as_tuple=True)[0] for c in unique_labels}
    pool_diff: Dict[int, torch.Tensor] = {
        c: ((labels != c).nonzero(as_tuple=True)[0] if multi_class else pool_same[c])
        for c in unique_labels}

    flat_labels = anchor_labels.reshape(-1)
    flat_want_same = want_same.reshape(-1)
    flat_partner = torch.empty(flat_labels.shape[0], dtype=torch.long, device=device)
    for c in unique_labels:
        for same, pool in ((True, pool_same[c]), (False, pool_diff[c])):
            mask = (flat_labels == c) & (flat_want_same == same)
            cnt = int(mask.sum().item())
            if cnt == 0:
                continue
            choice = pool[torch.randint(len(pool), (cnt,), device=device)]
            flat_partner[mask] = choice
    partner_idx = flat_partner.view(batch_count, half_size)

    batch_idx = torch.cat([anchor_idx, partner_idx], dim=1)
    return batch_count, data[batch_idx], labels[batch_idx]


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
