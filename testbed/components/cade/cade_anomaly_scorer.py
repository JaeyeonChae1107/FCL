"""CADEMADScorer — CADE MAD 정규화 거리 기반 anomaly scorer (PRD 12.6절).

12.1절 참고 — 이 클래스는 CADEDriftDetector와 이름만 CADE를 공유할 뿐 완전히
별개의 클래스이며 상태를 공유하지 않는다. 이 스코어러는 (A) 공유 표현
소비자로, 메인 분류기(BaseCLModel)의 z 공간에서 직접 centroid/MAD를 계산한다.
fit()은 항상 정상(label=0) 데이터만 받으므로 centroid는 단일(정상) centroid다.

`score()`가 CADE 원문의 `A(x,i)=|dist-median|/mad`(`CADE/cade/detect.py:91,
150-158`)를 그대로 구현한다 — 이 값 자체가 이미 "몇 MAD만큼 벗어났는가"라는
단위다. 원문의 실제 판정(`detect.py:99`)은 이 값을 **상수 `mad_threshold`
(기본 3.5)와 직접 비교**한다(`if min_anomaly_score > mad_threshold`) — 점수
위에 다시 median/MAD를 씌우는 단계가 없다.

**2026-08-14 재검토 — 이중 MAD 정규화, 인용은 정정하되 코드는 유지**: 이
`compute_threshold()`는 `median(eval_scores)+t_mad*MAD(eval_scores)`를
계산하는데, `eval_scores` 자체가 이미 `score()`로 1차 MAD-정규화된 값이라
정규화를 두 번 적용하는 셈이다(구조 전수 감사에서 발견 — 예전에 인용했던
`detect.py:91,150-158`은 점수 공식이지 이런 임계값 유도 공식의 근거가 아니었다,
인용 정정). A/B 실측(NSL-KDD, 순정 CADE 콤보 `dd=cade/ss=random/mm=none/
af=none/as=cade_mad`)으로 원문처럼 상수 `t_mad`만 쓰도록 바꿔봤더니 f1
0.6482→0.5746, bwt -0.1403→-0.1896로 **오히려 악화**됐다(pr_auc/roc_auc는
0.7492/0.7415로 불변 — score() 자체는 안 바뀌고 threshold만 바뀌었다는 뜻).
원인으로 보이는 것: CADE 원문은 크고 안정된 단일 코퍼스로 한 번만 median/mad를
보정하는 정적 설계인데, 이 테스트베드는 라벨 예산(10%)만큼의 작은 정상
참조 표본으로 **매 라운드** 다시 보정한다(`cl_client.py` Step 6) — 그
결과 라운드마다 절대 스케일이 흔들릴 수 있는데, 이중 MAD 정규화가 그
라운드별 스케일 잡음을 흡수하는 적응적 보정 역할을 하는 것으로 보인다.
`normal_reference 재설계`(위 절 참고)와 같은 종류의 구조적 차이 — "정적
1회 보정 vs 반복 라운드 보정"에서 비롯된 필요한 적응이라 판단해 이중 MAD를
그대로 유지한다.

**2026-08-14 발견·수정 — "순정 CADE" 조합에서 대조학습 인코더와 MAD 판정이
서로 연결되지 않던 문제**: CADE의 실제 발명은 "대조학습으로 만든 latent
space 위에서 MAD 거리로 판정"하는 것 하나인데, 이 두 부품이 이 테스트베드의
5-슬롯 분해(drift_detector/anomaly_scorer 독립 축) 때문에 서로 분리돼
있었다 — `CADEDriftDetector`가 소유한 사설 `ContrastiveAutoEncoder`는
원문 그대로 학습되지만, `ss=random`+`mm=none` 같은 "비활성" 조합에서는
그 출력이 drift_detector 슬롯 밖으로 전혀 안 나갔고(이미 알려진 문제),
`CADEMADScorer`는 그와 무관하게 **공유 backbone**의 z에 MAD 공식만
적용하고 있었다(구조 전수 감사에서 발견) — 즉 "순정 CADE" 조합에서도
CADE의 핵심 아이디어가 한 번도 통합되어 실행되지 않았다.

이제 `dd=cade`와 `as=cade_mad`가 **함께** 선택된 콤보에서만
`CLClient`가 `set_private_encoder()`로 이 스코어러를 CADEDriftDetector에
연결한다(`uses_shared_representation`를 False로 전환,
`pipeline/cl_client.py` 참고) — `drift_detector.uses_shared_representation`
과 대칭인 설계다. `dd=cade`가 아닌 조합에서 `as=cade_mad`를 쓰면(예:
`dd=none`+`as=cade_mad`) 여전히 공유 backbone의 z를 쓴다 — 이는 버그가
아니라 "MAD 통계 판정 방식이 대조학습 없이도 통하는가"를 보는 정당한
재조합 실험으로 남긴다.

**2026-08-25 확장 — 연결 시 centroid 계산 자체를 CADEDriftDetector에
위임**: 처음 연결했을 때는(위 "2026-08-14" 절) 인코더만 공유하고 centroid/
median/MAD는 이 클래스가 (정상 데이터만으로) 별도로 다시 계산했다. 그런데
CADE 원문은 애초에 "인코더 + family별 centroid + MAD 판정"이 하나의
파이프라인이지 두 벌이 아니다(`fit_with_category()` 도입 배경은
`cade_drift_detector.py` 모듈 docstring 참고) — 특히 CADEDriftDetector가
이제 다중 family centroid(정상 포함)를 유지하는데, 이 클래스가 그와
별개로 "정상 전용 단일 centroid"를 또 계산하면 두 계산이 서로 다른
그룹핑을 쓰게 되어 "연결됐다"는 말이 무색해진다. 그래서 연결된 경우
`fit()`은 아무것도 하지 않고(centroid는 이미 CADEDriftDetector.fit_with_
category()가 이번 라운드분으로 갱신해 둔 상태), `score()`는
`CADEDriftDetector.min_anomaly_score()`(family별 거리의 min, CADE 원문
`detect.py:91-104`와 동일한 판정)를 그대로 위임한다. 연결 안 된 경우
(`dd=cade`가 아닌 조합)의 기존 단일-centroid 방식은 그대로 유지한다.
"""

from typing import Optional

import torch

from testbed.base.anomaly_scorer import BaseAnomalyScorer


class CADEMADScorer(BaseAnomalyScorer):
    required_backbone = "classifier"

    def __init__(self, t_mad: float = 3.5):
        self.t_mad = t_mad
        self._centroid: Optional[torch.Tensor] = None
        self._median: Optional[torch.Tensor] = None
        self._mad: Optional[torch.Tensor] = None
        self._private_detector = None

    def set_private_encoder(self, detector) -> None:
        """CLClient 전용 훅 — dd=cade와 함께 선택됐을 때만 호출된다(위 모듈
        docstring "2026-08-14"/"2026-08-25" 절 참고). `detector`는
        CADEDriftDetector 객체 자체(인코더뿐 아니라 family centroid까지
        같이 참조하기 위함, 2026-08-25 변경 — 이전엔 인코더 nn.Module만
        받았다)이며, 매 라운드 CADEDriftDetector.fit_with_category()가
        갱신하는 상태를 별도 재전달 없이 그대로 반영한다."""
        self._private_detector = detector
        self.uses_shared_representation = False

    def fit(self, normal_data: torch.Tensor) -> None:
        if self._private_detector is not None:
            return
        if len(normal_data) == 0:
            return
        self._centroid = normal_data.mean(dim=0)
        dist = torch.norm(normal_data - self._centroid, dim=1)
        self._median = dist.median()
        mad = 1.4826 * (dist - self._median).abs().median()
        # 2026-08-26 발견·수정(재감사, cade_drift_detector.py의 같은 절 참고)
        # — 절대 상수 floor는 참조 표본이 서로 거의 중복이라 MAD가 0에
        # 가까워지면 점수를 물리적 의미 없는 값까지 폭발시킬 수 있다.
        # 거리 스케일(median)에 비례한 floor를 절대 floor와 함께 쓴다.
        mad_floor = torch.clamp(0.01 * self._median.abs(), min=1e-8)
        self._mad = torch.clamp(mad, min=mad_floor)

    def score(self, data: torch.Tensor) -> torch.Tensor:
        if self._private_detector is not None:
            scores = self._private_detector.min_anomaly_score(data)
            if scores is None:
                return torch.zeros(len(data), device=data.device)
            return scores
        if self._centroid is None:
            return torch.zeros(len(data), device=data.device)
        dist = torch.norm(data - self._centroid, dim=1)
        return (dist - self._median).abs() / self._mad

    def compute_threshold(self, eval_scores: torch.Tensor,
                           eval_labels: Optional[torch.Tensor]) -> float:
        # Track A(cade_mad): eval_scores는 정상 참조 데이터의 score(s_ref).
        # eval_labels는 쓰지 않는다(통계적 threshold, 라벨 불필요 — PRD 3.5절).
        # 위 모듈 docstring "2026-08-14" 절 참고 — 원문처럼 상수 t_mad를
        # 그대로 쓰는 게 아니라, s_ref 위에 다시 median/MAD를 씌운
        # "이중 정규화"를 의도적으로 유지한다(A/B 실측 근거).
        median = eval_scores.median()
        mad = 1.4826 * (eval_scores - median).abs().median()
        # 2026-08-26 발견·수정(재감사) — 절대 floor 대신 스케일 비례 floor
        # (cade_drift_detector.py의 같은 절 참고).
        mad_floor = torch.clamp(0.01 * median.abs(), min=1e-8)
        mad = torch.clamp(mad, min=mad_floor)
        threshold = median + self.t_mad * mad

        # 2026-08-25 발견·수정 — ss=ssf(균일 커버리지 선택, "대표성" 표본이
        # 분포 전 구간에 고르게 퍼지도록 강제) + dd=cade의 다중 family
        # min-centroid 점수(다중클래스 연결, 위 모듈 docstring "2026-08-25"
        # 절 참고)가 만나면, eval_scores(정상 참조) 자체가 이질적으로 넓게
        # 퍼져(일부 "대표" 정상 표본이 우연히 어떤 공격 family centroid에
        # 더 가까워 점수가 낮게 나오고, 나머지는 정상 centroid 기준으로
        # 정상적인 값이 나옴) MAD가 과대해지고 threshold가 eval_scores
        # 자기 자신의 최댓값조차 넘어버리는 경우를 스모크 테스트로 실측
        # 확인했다(NSL-KDD, threshold=33.81, eval_scores range=[0,14.03] —
        # 이러면 어떤 표본도 이상으로 판정될 수 없어 예측이 완전히 정상
        # 한 클래스로 퇴화한다). threshold가 자신의 계산 근거였던 eval_scores
        # 범위조차 벗어나면 이미 "그 계산 근거로 보증되는 범위"를 넘어선
        # 것이므로, eval_scores 최댓값으로 clamp한다 — t_mad를 낮추는 식의
        # 전역 재보정 대신, 이 이중 MAD 공식이 정상 범위에서 계산될 때는
        # 전혀 손대지 않고 이런 병적인 경우만 막는 최소 안전장치다.
        threshold = torch.clamp(threshold, max=eval_scores.max())
        return float(threshold)
