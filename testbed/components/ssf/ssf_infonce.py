"""SSF InfoNCE 재구성-대조 손실 — SSF 원 논문 근거.

SSF-Strategic-Selection-and-Forgetting/utils.py:458-492 (`InfoNCELoss`). SSF는
이 손실을 latent feature가 아니라 디코더 출력(recon_vec)에 적용한다
(`ssf.py:139,310,326` — `criterion(recon_vec, labels)`; `criterion =
InfoNCELoss(device, tem)`, `tem=0.02`는 `ssf.py:47,76`의 고정값) — 재구성
표현 공간에서 정상표본끼리는 서로 가깝게(분자), 정상-공격 쌍은 InfoNCE
분모로 밀어내는 대조 손실이다. 정상(label=0)인 행만 anchor로 쓴다 — SSF가
"정상 재구성이 서로 뭉치고 공격 재구성과는 멀어지도록" 유도하는 방식이다.

원문은 reduction 없이 (정상표본수 x 정상표본수) 행렬을 반환하고 호출부가
new_sample_weight로 가중한 뒤 `.mean()`한다(`ssf.py:311,317` 참고) — 이
함수도 동일하게 raw 행렬을 반환하고, 가중·reduction은 `ssf_anti_forgetting.py`
쪽에서 한다.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def ssf_infonce_loss(recon: torch.Tensor, labels: torch.Tensor,
                      temperature: float = 0.02) -> Optional[torch.Tensor]:
    """utils.py:465-492을 그대로 이식.

    배치에 정상(label=0) 표본이 하나도 없으면 anchor 자체가 없어 원문이
    다루지 않는 상황이므로 None을 반환한다(호출부가 그 라운드는 이 항을
    생략하도록) — 정상 표본이 1개뿐이거나 공격 표본이 0개인 경우는 원문의
    실제 수식(자기 자신과의 유사도는 대각선 마스킹으로 이미 제거됨, 공격
    표본 합은 빈 합=0)이 그대로 안전하게 처리하므로 별도 분기 없이 둔다.
    """
    normal_mask = labels == 0
    n_normal = int(normal_mask.sum().item())
    if n_normal == 0:
        return None

    features = F.normalize(recon, p=2, dim=1)
    batch_size = features.shape[0]
    logits = (features @ features.T) / temperature
    logits_mask = torch.ones_like(logits) - torch.eye(batch_size, device=features.device)
    logits_without_ii = logits * logits_mask

    logits_normal = logits_without_ii[normal_mask]
    logits_normal_normal = logits_normal[:, normal_mask]
    abnormal_mask = ~normal_mask
    if abnormal_mask.any():
        sum_of_vium = torch.exp(logits_normal[:, abnormal_mask]).sum(dim=1, keepdim=True)
    else:
        sum_of_vium = torch.zeros(n_normal, 1, device=features.device)

    denominator = torch.exp(logits_normal_normal) + sum_of_vium
    log_probs = logits_normal_normal - torch.log(denominator)
    return -log_probs * temperature
