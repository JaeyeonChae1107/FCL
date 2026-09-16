"""모델 계약 — PRD 11절.

BaseCLModel.forward(x) -> (z, x_hat, logit):
  z     : (N, latent_dim)  잠재 표현. anomaly_scorer, drift_detector 입력
  x_hat : (N, input_dim)   재구성. Track B의 anti_forgetting(cndids)용
  logit : (N, 1)           이진 판별 로짓. Track A의 anti_forgetting/drift_detector(ssf) 입력

**분류기는 SSF 원문대로 x_hat(재구성)에서 나온다**: SSF 원문(`utils.py:
59-61`)은 `classifier(decoder(z))` — 재구성 x_hat을 분류기 입력으로 쓴다.
CADE는 분류기와 오토인코더가 아예 분리된 별도 모델이고, CND-IDS는 지도
분류기 헤드가 없어 이 결정과 무관하다 — 셋 중 유일하게 분류기를 실제로
쓰는 SSF의 방식을 그대로 따랐다(raw logit, sigmoid 없음은 유지).

**decoder/classifier 입력 ReLU 누락(2026-09-14 재검토로 발견, 수정)**:
SSF 원문(`utils.py:28-57`, `AE_classifier`)의 decoder/classifier는 둘 다
"입력에 먼저 ReLU를 적용한 뒤 첫 Linear를 통과"하는 구조다 —
`decoder = Sequential(ReLU(), Linear(latent,hidden), ReLU(), Linear(hidden,
input))`, `classifier = Sequential(ReLU(), Linear(input,1), Sigmoid())`.
이 테스트베드는 그동안 두 곳 다 이 선행 ReLU를 빠뜨리고 z/x_hat을 그대로
첫 Linear에 통과시켰다 — 음수 성분을 0으로 클리핑하는 비선형 변환 자체가
빠진 것이라 단순 스케일 차이가 아니다. 78개 조합 전체가 공유하는 backbone
이라 x_hat(재구성)과 logit(SSF 계열의 판정·InfoNCE·LwF가 전부 의존) 양쪽에
영향을 준다 — "구조적 충실도 버그"로 분류해 A/B 방향과 무관하게 채택한다
(`design_decisions.md` 6절 원칙). Sigmoid 생략은 `BCEWithLogitsLoss`와
수학적으로 동치라 그대로 유지한다(raw logit). sanity 확인(NSL-KDD, epoch=5,
붕괴 없음): naive 기준선 f1=0.1060/bwt=-0.6399, SSF 풀조합 f1=0.6542/
bwt=+0.0102, CADE f1=0.4493/bwt=-0.1513 — 전부 정상 범위.
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
    다른 데이터셋에도 강제하지 않는다. 이 테스트베드도 동일하게 맞춘다
    (이전에는 UNSW-NB15 하나에서 계산한 값을 3개 데이터셋
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
    """Track A/B 공용. Track A 대부분은 x_hat을 손실 계산에 쓰지 않지만
    SSFAntiForgetting(af=lwf_ssf)은 x_hat에 InfoNCE 재구성-대조 손실을
    적용한다(ssf_infonce.py 참고) — 클래스 자체는 공유한다."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        logit = self.classifier(x_hat)
        return z, x_hat, logit
