"""CADEMADScorer — CADE MAD 정규화 거리 기반 anomaly scorer (PRD 12.6절).

CADEDriftDetector와 이름만 공유하는 완전히 별개 클래스, 상태 비공유. 기본은
공유 표현 소비자 — 메인 분류기(BaseCLModel)의 z에서 centroid/MAD를 계산.
fit()은 정상(label=0)만 받아 단일 centroid.

`score()` = CADE 원문 `A(x,i)=|dist-median|/mad`(`CADE/cade/detect.py:91,
150-158`). 원문 판정(`detect.py:99`)은 이 값을 상수 `mad_threshold`(기본
3.5)와 직접 비교한다.

`compute_threshold()`는 `median(eval_scores)+t_mad*MAD(eval_scores)` —
eval_scores가 이미 1차 MAD-정규화된 값이라 이중 정규화다(원문 `detect.py:91,
150-158`은 점수 공식이지 이 임계값 유도식의 근거는 아님). A/B(NSL-KDD, `dd=cade/ss=random/mm=none/af=none/as=cade_mad`):
원문처럼 상수 t_mad만 쓰면 f1 0.6482→0.5746, bwt -0.1403→-0.1896(pr_auc/
roc_auc는 0.7492/0.7415로 불변 — score()는 그대로, threshold만 변경). 이중
MAD 유지. 원인 추정: 원문은 정적 1회 보정이지만 이 테스트베드는 라벨 예산
(10%) 정상 표본으로 매 라운드 재보정(`cl_client.py` Step 6)해 라운드별
스케일이 흔들리고, 이중 MAD가 그 잡음을 흡수하는 것으로 보임.

`dd=cade`와 `as=cade_mad`가 함께 선택된 콤보에서만 `CLClient`가
`set_private_encoder()`로 이 스코어러를 CADEDriftDetector에 연결한다
(`uses_shared_representation`를 False로 전환, `pipeline/cl_client.py`).
연결 전에는(이전) `CADEDriftDetector`가 학습시키는 사설 대조학습
인코더의 출력이 어디에도 쓰이지 않았고, `CADEMADScorer`는 무관한 공유
backbone의 z에 MAD 공식만 적용하고 있었다. `dd=cade`가 아닌 조합에서
`as=cade_mad`를 쓰면(예: `dd=none`+`as=cade_mad`) 공유 backbone의 z를
그대로 쓴다.

**연결 시 centroid 계산을 CADEDriftDetector에 위임**: 이전엔
인코더만 공유하고 centroid/median/MAD는 이 클래스가 정상 데이터만으로 별도
계산했으나(`CADEDriftDetector`가 다중 family centroid를 유지하는 것과
다른 그룹핑), 연결된 경우 `fit()`은 아무것도 하지 않고(centroid는
CADEDriftDetector.fit_with_category()가 갱신), `score()`는
`CADEDriftDetector.min_anomaly_score()`(family별 거리의 min, `detect.py:
91-104`)로 위임한다. 미연결(`dd=cade`가 아닌 조합)은 기존 단일-centroid
방식 유지.
"""

from typing import Optional

import torch

from testbed.base.anomaly_scorer import BaseAnomalyScorer


class CADEMADScorer(BaseAnomalyScorer):
    required_backbone = "classifier"
    # 2026-09-12 추가(docs/design_decisions.md 5절) — CADEDriftDetector와 같은 이유
    # (원 논문에 라벨 예산 개념 없음). `_private_detector`에 연결된 경우
    # (dd=cade와 짝지어짐) fit()은 이미 no-op이라(centroid는 CADEDriftDetector가
    # 관리) 이 플래그가 실질적 영향이 없고, 오직 `as=cade_mad`가 `dd=cade`
    # 없이 단독 선택된 경우(공유 backbone z에 CADE의 median/MAD 공식만
    # 적용)에만 실제로 `pipeline/cl_client.py` Step 6의 `ref_raw`를
    # `new_data`의 정상 서브셋 전체로 넓힌다 — CADE 자신의 median/MAD
    # 통계는 원래 "그 시점 라벨 있는 정상 데이터 전체"로 계산하는 것이
    # 맞기 때문.
    consumes_full_round_data = True

    def __init__(self, t_mad: float = 3.5):
        self.t_mad = t_mad
        self._centroid: Optional[torch.Tensor] = None
        self._median: Optional[torch.Tensor] = None
        self._mad: Optional[torch.Tensor] = None
        self._private_detector = None

    def set_private_encoder(self, detector) -> None:
        """CLClient 전용 훅 — dd=cade와 함께 선택됐을 때만 호출된다. `detector`는
        CADEDriftDetector 객체 자체(인코더 + family centroid 참조).
        CADEDriftDetector.fit_with_category()가 매 라운드 갱신하는 상태를
        그대로 반영한다."""
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
        # 절대 floor(1e-8)만이면 참조 표본이 거의 중복이라 MAD→0일 때 점수가
        # 폭발할 수 있다 — 거리 스케일(median) 비례 floor를 함께 적용
        # (cade_drift_detector.py 동일 절).
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
        # eval_scores = 정상 참조 데이터의 score(s_ref). eval_labels는 쓰지
        # 않음(통계적 threshold, PRD 3.5절). 이중 정규화 유지 근거는 모듈
        # docstring 참고.
        median = eval_scores.median()
        mad = 1.4826 * (eval_scores - median).abs().median()
        mad_floor = torch.clamp(0.01 * median.abs(), min=1e-8)
        mad = torch.clamp(mad, min=mad_floor)
        threshold = median + self.t_mad * mad

        # ss=ssf + dd=cade(다중 family min-centroid)가 만나면 eval_scores가
        # 이질적으로 넓게 퍼져 threshold가 eval_scores 자체 최댓값을 넘는
        # 경우가 있다(실측: NSL-KDD threshold=33.81, range=[0,14.03] — 전
        # 표본이 정상으로 퇴화). eval_scores 최댓값으로 clamp.
        threshold = torch.clamp(threshold, max=eval_scores.max())
        return float(threshold)
