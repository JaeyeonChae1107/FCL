"""BaseAnomalyScorer — PRD 12.6절.

재보정(refit)과 threshold 결정 요구사항:
- refit_on_update()는 encoder가 갱신될 때마다(매 experience 종료 시) 호출되어
  scorer의 내부 파라미터(PCA 기저, centroid 등)를 최신 encoder 기준으로
  갱신한다. threshold 결정과는 별개 절차다.
- compute_threshold()는 scorer마다 완전히 다른 원 논문 방식을 그대로
  구현한다(PRD 3.4/3.5절):
    pca: eval_scores=test 데이터 전체의 score, eval_labels 필수
        (Best-F, CND-IDS Algorithm 1).
    cade_mad/none: eval_scores=정상 참조 데이터의 score(s_ref), eval_labels
        불필요 (median + T_MAD*MAD — 이 이중 정규화 자체는 CADE 원 논문이
        아니라 이 테스트베드의 재보정 절차에 대응한 것, 근거는
        `cade_anomaly_scorer.py` 모듈 docstring 참고 / 고정 0.5, SSF/SPIDER
        원 논문).

**threshold 보정 방식은 Track이 아니라 scorer 자체의 속성**: 이전에는
`CLClient`가 `self.track == "A"`로 분기해 위 두 방식을 선택했다. 진짜
기준은 트랙이 아니라 "이 scorer의 compute_threshold()가 정상 참조(s_ref)
로 충분한가, 아니면 라벨이 붙은 pooled eval 표본이 필요한가"라는 scorer
고유의 계약이다(트랙은 `pca`=Track B, `cade_mad`/`none`=Track A로
겹쳤을 뿐 논리적으로 같은 것은 아니었다). `dd=none`+`as=cade_mad`가
"CADE 없이 MAD 채점만 쓰는" 정당한 재조합으로 남아 있는 것과 대칭으로,
"CND-IDS 표현 위에서 CADE MAD 채점만 쓰는" Track B 조합도 구조적으로
막을 이유가 없어(`common/compatibility.py` TRACK_B_GRID 참고)
`threshold_needs_labels` 플래그를 도입했다 — `CLClient`는 track이 아니라
이 플래그로 분기한다(`pipeline/cl_client.py` Step 7 참고).
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch


class BaseAnomalyScorer(ABC):
    required_backbone: str
    # drift_detector.uses_shared_representation과 같은 개념. 기본값
    # True(공유 backbone의 z를 입력으로 받음). CADEMADScorer만 dd=cade와
    # 함께 선택됐을 때 False로 전환해, 공유 z 대신 CADEDriftDetector
    # 사설 대조학습 인코더로 직접 인코딩한 표현을 쓴다(CLClient 참고) — CADE의
    # 실제 설계(대조학습 latent space 위에서 MAD 판정)를 온전히 재현하기 위함.
    uses_shared_representation: bool = True
    # compute_threshold()가 라벨이 붙은 pooled eval 표본을 반드시 필요로
    # 하는지(Best-F류, True) 아니면 정상 참조(s_ref)만으로 충분한지
    # (median+MAD류/고정값, False) 나타낸다. 기본값 False. PCAScorer만
    # True로 override한다(모듈 docstring 참고).
    threshold_needs_labels: bool = False
    # Step 6가 어떤 "정상 참조"를 넘길지 고르는 두 계약 — 둘 다 이 base
    # 클래스에 기본값 False로 선언해(2026-09-14 명시 — 이전엔 PCAScorer/
    # CADEMADScorer 등 개별 서브클래스에만 있어 `cl_client.py`가 getattr(...,
    # False)로 방어해야 했다. 런타임 동작은 항상 이미 옳았지만, 계약이 "이
    # 베이스 클래스가 보장하는 것"이 아니라 "각자 알아서 없으면 False일
    # 것"에 의존해 명시성이 떨어졌다) `hasattr` 대신 `getattr(obj, name)`을
    # 안전하게 쓸 수 있게 한다.
    # consumes_held_out_reference=True인 scorer(PCAScorer)는 CND-IDS의
    # 고정 held-out 정상 참조(init_normal)를 받는다.
    consumes_held_out_reference: bool = False
    # consumes_full_round_data=True인 scorer(CADEMADScorer)는 라벨 예산
    # 서브셋 대신 그 라운드 전체 정상 표본을 받는다(2026-09-12, CADE 라벨
    # 예산 면제).
    consumes_full_round_data: bool = False

    @abstractmethod
    def fit(self, normal_data: torch.Tensor) -> None:
        ...

    @abstractmethod
    def score(self, data: torch.Tensor) -> torch.Tensor:
        ...

    def predict(self, data: torch.Tensor, threshold: float) -> torch.Tensor:
        return (self.score(data) > threshold).long()

    def refit_on_update(self, normal_data: torch.Tensor) -> None:
        """encoder가 갱신될 때마다 재계산. 기본 구현은 fit() 재호출."""
        self.fit(normal_data)

    @abstractmethod
    def compute_threshold(self, eval_scores: torch.Tensor,
                           eval_labels: Optional[torch.Tensor]) -> float:
        ...
