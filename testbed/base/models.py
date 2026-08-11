"""모델 계약 — PRD 11절.

BaseCLModel.forward(x) -> (z, x_hat, logit):
  z     : (N, latent_dim)  잠재 표현. anomaly_scorer, drift_detector 입력
  x_hat : (N, input_dim)   재구성. Track B의 anti_forgetting(cndids)용
  logit : (N, 1)           이진 판별 로짓. Track A의 anti_forgetting/drift_detector(ssf) 입력

**분류기가 z에서 바로 나온다는 것은 SSF 원문과 다른 의도적 단순화다**
(2026-07-30 재검토, `docs/metric_justification.md` "전체 코드베이스 최종
라인별 재검토" 참고). SSF 원문(`utils.py:59-61`)은 `classifier(decoder(z))`
— 재구성 x_hat을 분류기 입력으로 쓴다. CADE는 분류기와 오토인코더가 아예
분리된 별도 모델이고, CND-IDS는 지도 분류기 헤드가 없다 — 세 논문의
"분류기가 뭘 보는가"가 전부 다르므로 하나의 공유 backbone에서는 가장 단순한
절충(z에서 바로 선형 분류, sigmoid 없이 raw logit)을 택했다.
"""

import math
from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


def ssf_backbone_dims(input_dim: int) -> Tuple[int, int]:
    """SSF 원 논문 공식(utils.py:32-36) 그대로: nearest_pow2 = 2**round(log2(input_dim));
    hidden = nearest_pow2 // 2; latent = nearest_pow2 // 4.

    SSF 저장소는 이 공식을 데이터셋을 로드할 때마다 매번 새로 계산한다
    (ssf.py:51-54,116-120) — 한 데이터셋에서 계산한 값을 고정 상수로 굳혀
    다른 데이터셋에도 강제하지 않는다. 2026-08-11 재검토로 이 테스트베드도
    동일하게 맞춘다(이전에는 UNSW-NB15 하나에서 계산한 값을 3개 데이터셋
    전부에 강제했고, NSL-KDD에는 이미 틀렸다는 걸 스스로 인정하고 있었다 —
    docs/metric_justification.md 참고). 호출부(grid_runner.py, smoke_test.py)가
    매 데이터셋의 input_dim으로 이 함수를 다시 호출해 global_hparams의
    hidden_dim/latent_dim을 그때그때 덮어쓴다."""
    nearest_pow2 = 2 ** round(math.log2(input_dim))
    hidden_dim = nearest_pow2 // 2
    latent_dim = nearest_pow2 // 4
    return hidden_dim, latent_dim


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
