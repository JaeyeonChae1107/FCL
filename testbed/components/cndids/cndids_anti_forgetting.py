"""CNDIDSAntiForgetting — 결합 손실 anti-forgetting (PRD 4절/12.5절).

CND-IDS 원 논문 근거 (CND-IDS/FeatureExtractors/CND_IDS.py:161-166):
  loss = metric_loss(L_CS) + reg_strength * reconstruction_loss(L_R)
         + LwF_strength * lossLwF(L_CL)
  기본 reg_strength=0.1(lambda_r), LwF_strength=0.1(lambda_cl) (CND_IDS.py:43,45).
  metric_loss는 TripletMarginLoss(margin=2, semihard mining)를 pseudo-label
  클러스터(0=정상, 1=신규) 간 거리에 적용한다(CND_IDS.py:76-78).

**Pseudo-label 생성 방식 (수정됨 — 원본과 대조 후 재작성)**: 처음에는 z에
대한 2-means로 근사했으나, CND-IDS 원본(`FeatureExtractors/CND_IDS.py:105-115`,
실제 clusterer는 `FeatureExtractors/modules/K_Means.py`)을 다시 대조한 결과
실제 메커니즘은 다음과 같음을 확인했다:
  1. `cluster_labels = self.labeler.fit_transform(x)` — experience의 **원본
     입력 x**(인코딩 전)에 elbow-선택 K-Means를 **한 번** 적용 (후보
     K=[100,300,500,1000,2000], `modules/K_Means.py:18`).
  2. `init_normal_labels = self.labeler.transform(init_normal)` — **알려진
     정상 참조 데이터**(`datastream.init_normal`)가 속하는 클러스터 ID 집합을
     구한다.
  3. `y = [0 if i in init_normal_labels else 1 for i in cluster_labels]` —
     그 클러스터 ID 집합에 속하면 정상(0), 아니면 신규/이상(1)으로 pseudo-label
     을 부여한다. **공격 라벨은 전혀 쓰지 않는다** — 오직 "이 클러스터에 알려진
     정상 참조 데이터가 있는가"만 본다.
  (참고: 같은 폴더의 `AnomolyDetectors/K_Means.py`는 이름은 같지만 별개의
  standalone anomaly-scorer 베이스라인이며, 그건 실제 라벨이 섞인 캘리브레이션
  서브셋을 쓴다 — CND_IDS.py의 pseudo-labeling과는 무관하다. 처음에 이 둘을
  혼동해 "CND-IDS가 공격 라벨을 쓴다"고 잘못 판단했었다.)

이 구현은 위 메커니즘을 그대로 따른다. K 후보 [100,300,500,1000,2000]도 원문
그대로 데이터셋 무관 고정값이다(2026-08-11 재검토 — 한때 label_budget
서브샘플링에 맞춰 [5,10,20,30,50,80]으로 축소했었지만, Track B가 experience
전체를 그대로 쓰도록 바뀌면서 축소해야 했던 이유 자체가 사라졌다. 자세한
경위는 아래 `_CLUSTER_K_CANDIDATES` 정의부 주석 참고). 다만 아래 한 가지는
이 테스트베드 고유의 조정이다:
  - "알려진 정상 참조 데이터"로, 이번 라운드 라벨 예산 안에서 이미 선택된
    데이터 중 label=0인 것만 걸러 쓴다(`cl_client.py`의 `normal_subset`,
    2026-07-30 재설계 — 별도 고정 표본을 만들지 않는다) — CND-IDS의
    `datastream.init_normal`과 같은 역할이다.
  - 클러스터링은 experience(라운드)당 한 번만 수행하고(`on_experience_start`),
    이후 각 미니배치에서는 이미 학습된 K-Means로 predict만 한다 — 원본이
    `fit()` 진입 시 한 번 클러스터링하고 그 결과를 epoch 전체에서 재사용하는
    것과 동일한 절차다.

`pytorch_metric_learning`은 이 환경에 없는 의존성을 새로 설치하는 위험을
피하기 위해(이전 `deepod` 사고 참고, testbed/docs/metric_justification.md)
TripletMarginLoss를 CADE의 contrastive pairing과 동일한 형태(배치를 반으로
나눠 쌍을 구성, margin 기반)로 직접 구현했다.

**2026-08-14 정정 — "알고리즘의 본질은 동일"이라는 서술이 과장이었음**:
구조 전수 감사에서 재확인한 결과, 원문(`CND_IDS.py:38-39,76-78`)은
`distances.LpDistance` + `TripletMarginLoss(margin=2)` +
`TripletMarginMiner(type_of_triplets="semihard")`로 **상대 마진 트리플릿**
(`d(anchor,positive) - d(anchor,negative) + margin`에 relu, semihard
마이닝으로 "적당히 어려운" 삼중쌍만 선택)을 쓰는데, `_metric_loss()`는
배치를 반으로 나눈 **절대 마진 페어와이즈**(같은 클러스터면 거리 자체를
그대로, 다른 클러스터면 `relu(margin-dist)`)를 쓴다. 둘 다 L2 거리를 쓰고
"같은 클러스터는 가깝게, 다른 클러스터는 margin 이상 멀게"라는 방향은
같지만, 손실의 수학적 형태(상대 마진 트리플릿 vs 절대 마진 페어)가 달라
어떤 쌍이 "hard"로 취급되는지가 다르다 — "본질적으로 동일"이 아니라
"같은 목표를 다른 손실 형태로 근사"한 것이다. 새 의존성 설치 위험을
피한다는 이유 자체는 여전히 타당하므로 구현은 유지하되, 서술만 정정한다.

**라벨-프리(label-free) 준수**: `new_batch`의 `selected_labels`는 손실 계산에
전혀 쓰지 않는다(PRD 12.5절 명시 사항) — 위 확인대로 CND-IDS 원본도 pseudo-
label 생성에 공격 라벨을 쓰지 않으므로 이 원칙과 실제로 상충하지 않는다.

**2026-08-12 발견·수정 — teacher를 직전 1개만 유지하던 문제**: CND-IDS 원문
(`CND_IDS.py:42,54-69,195`)은 `self.old_models`라는 리스트에 매 experience
종료 시(`fit()` 마지막 줄) 그 시점 모델 전체를 `deepclone`해 **계속 추가만
하고 절대 비우지 않는다**. `LwFloss()`는 이 리스트의 **모든** 과거 모델에 대해
개별적으로 `MSE(현재 encoder 출력, old_model(현재 배치))`를 계산해 합산한다
(`CND_IDS.py:57-66`). 즉 experience 3에서는 experience 0·1·2의 스냅샷
**세 개 전부와** 거리를 재는 것이지, 직전 하나만 보는 게 아니다 — 이전
구현은 `self._teacher`를 매 `on_task_end`마다 덮어써서 가장 최근 스냅샷하고만
비교했으므로, 라운드가 진행될수록 원문 대비 망각방지 압력이 점점 약해지고
있었다(n_experiences=5라 최대 4개까지 누적되므로 비용 부담은 작다).

또한 원문의 가중치 적용 방식도 다시 대조해 함께 고쳤다: `LwFloss()` 내부에서
개별 old-model 항마다 이미 `reg_strength`(=lambda_r)를 곱하고(`:66`), 그
합계에 호출부가 **다시** `LwF_strength`(=lambda_cl)를 곱한다(`:159,180`) —
즉 항 하나당 실효 가중치는 `lambda_r * lambda_cl`이지 `lambda_cl` 단독이
아니다. 이전 구현은 `lambda_cl`만 곱하고 있었다(이 값이 우연히 CND-IDS
기본값에서 lambda_r과 같은 0.1이라 결과가 크게 갈리진 않았지만, 두 값이
달라지면 원문과 어긋난다).

**2026-08-26 발견·수정 — "정상 참조"가 매 라운드 그 라운드 자신의 라벨
구성에서 다시 정의되어, 공격이 희귀한 라운드에서 참조가 거의 전체를
뒤덮는 문제**: 4개 논문 컴포넌트 전수 재감사에서, `on_experience_start`가
받는 `normal_subset`(=`cl_client.py`의 `selected_data[selected_labels==0]`,
Track B는 experience 전체가 selected_data)이 원문의 `datastream.init_normal`
(스트림 시작 전 **한 번** 확정되는 고정 참조, 이후 라운드 구성과 무관)과
근본적으로 다른 성질을 가진다는 걸 재확인했다 — 이 참조의 크기가 **이번
라운드 자신의 공격 비율**에 그대로 좌우된다. R2L 라운드(실제 공격 6.9%)는
`normal_subset`이 라운드의 93.1%, U2R 라운드(실제 공격 0.38%)는 99.6%까지
차지한다. 참조 데이터가 그 라운드 데이터 거의 전체와 겹치면, K-Means가
만든 클러스터 대부분이 참조 데이터를 최소 하나는 포함하게 되어(`is_normal
[cluster] = True` 판정 기준이 "그 클러스터에 참조 데이터가 하나라도
있는가") **거의 모든 클러스터가 "정상"으로 판정**된다 — 실측(NSL-KDD 5라운드
전체)으로 pseudo-label 다수 클래스 비율이 R2L 0.9844, U2R 1.0000까지
치솟았고, U2R 라운드 직후 0.870이었던 그 category의 정확도가 다음 라운드엔
0.707로 떨어지는 실제 망각 효과까지 확인했다(`smoke_test.py` 모듈 docstring
"2026-08-26" 절, `docs/metric_justification.md` 참고).

원문의 실제 설계 의도(참조는 "지금 이 라운드가 어떻게 생겼는지"와 무관하게
안정적으로 "알려진 정상이 무엇인지"만 나타내야 함)에 맞춰, `normal_subset`을
**인스턴스 수명 전체에 걸쳐 누적**하는 별도 풀(`_normal_ref_pool`, 최근
`max_normal_ref`개 캡)로 바꿨다. 클러스터링 자체(원문처럼 그 라운드 원본
입력으로 매 라운드 새로 fit)는 그대로 두고, "이 클러스터가 정상인가"
판정에만 그 라운드 하나가 아니라 누적 풀을 쓴다.

**2026-08-26 재감사에서 발견 — 위 "누적"이 처음엔 사실상 전혀 누적되지
않고 있었다**: 처음 구현(`combined_ref[-max_normal_ref:]`, 꼬리 슬라이싱)
은 CADE의 `_category_refs`와 같은 패턴을 그대로 흉내 낸 것이었는데, CADE는
공격 family별로 나눠 담아 한 family의 라운드당 기여가 보통 cap보다
작다는 전제가 성립하지만, 여기는 "정상"이라는 단일 통합 버킷이라 그
전제가 전혀 안 맞았다 — Track B는 label_budget 없이 experience 전체를
쓰므로 `normal_subset` 자체가 이미 매 라운드 cap(5000)보다 훨씬 크다
(NSL-KDD 13,468~59,396건). 그 결과 `combined_ref`의 마지막 `cap`개는
**항상 100% 이번 라운드 자신의 데이터**였다(실측: 5라운드 전체에서 이전
라운드 표본 생존율 0%, provenance 추적으로 확인) — "누적"이라는 이름과
달리 실제로는 매 라운드 참조를 통째로 교체하고 있었던 것이다.

그런데도 위에서 인용한 개선(f1→0.889 등)은 실제로 관찰됐다 — 원인을
분리 실험(같은 K-Means 클러스터링에 대해 참조를 (A) 라운드 전체 vs
(B) 5000건 무작위 부분표본으로만 바꿔서 비교)으로 추적한 결과, "라운드를
넘어 기억한다"가 아니라 **"참조 표본 수 자체가 줄어들면 그만큼 K-Means
클러스터가 덜 뒤덮여 '정상'으로 오판되는 클러스터가 줄어든다"**는 전혀
다른, 원 논문과 무관한 우연한 부작용이었음이 확인됐다(같은 클러스터링,
참조만 전체 vs 5000건 무작위: R2L 라운드 pseudo_ratio 0.9720→0.9634,
U2R 라운드 0.9998→0.9962 — 표본 수 축소 자체가 효과의 전부였다).

이제 꼬리 슬라이싱 대신 **무작위 표본**(`torch.randperm`)으로 캡을
적용한다 — `combined_ref`(누적 풀 + 이번 라운드) 전체에서 균등하게
`max_normal_ref`개를 뽑으므로, 이전 라운드의 표본도 실제로 (비율만큼)
살아남는다. 이러면 "우연한 표본 수 축소 효과"와 "원래 의도했던 진짜
누적 효과"가 함께 작동하게 된다 — 다만 완벽한 CND-IDS 원문의
`datastream.init_normal`(스트림 시작 전 한 번 고정되는 참조)과는 여전히
다르다는 점은 정직하게 남긴다: 이건 그 근사(라운드가 지날수록 갱신되는
근사적 저수지 표본)이지 문자 그대로 "한 번 고정"은 아니다. A/B 실측
결과는 `docs/metric_justification.md` 참고.
"""

import copy
from typing import List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from testbed.base.anti_forgetting import BaseAntiForgetting
from testbed.base.models import BaseCLModel

# **2026-08-11 재검토로 원상 복귀**: CND-IDS 원본(modules/K_Means.py:18)은
# [100,300,500,1000,2000]을 데이터셋 무관하게(7개 평가 데이터셋 전부 동일)
# 그대로 쓴다 — 스케일링 공식 자체가 없다. 이전(2026-08-04)에 이 값을
# [5,10,20,30,50,80]으로 축소했던 이유는 "라운드당 선택 데이터가 수천 건"
# 이라는, 이 테스트베드가 자체적으로 추가했던 label_budget 서브샘플링(~10%)
# 때문이었다(코드 주석에 명시돼 있었음). 그런데 그 서브샘플링은 Track B가
# 원 논문처럼 experience 전체를 그대로 쓰도록 바뀌면서 이미 제거됐다 —
# 즉 "작은 리스트로 축소해야 했던 이유" 자체가 사라졌으므로, sqrt(n)
# 스케일링(내가 임의로 고안한 공식)이 아니라 원 논문 리스트를 그대로
# 채택하는 게 맞다. CND-IDS 저장소가 평가하는 7개 데이터셋은 전부 이
# 테스트베드의 NSL-KDD보다 훨씬 큰 규모라, 원문 리스트가 NSL-KDD처럼
# 작은 데이터에서도 잘 맞는다는 보장은 없다 — 실측(A/B, 기존 축소 리스트
# 대비)으로 확인 후에도 품질이 유지되는지가 관건이며, 확인 결과는
# docs/metric_justification.md에 기록한다.
_CLUSTER_K_CANDIDATES = (100, 300, 500, 1000, 2000)


def _elbow_kmeans_fit(data_np: np.ndarray, candidates: List[int], seed: int = 42,
                       fit_sample_size: Optional[int] = None):
    """CND-IDS modules/K_Means.py:fit()과 동일한 elbow 선택 절차.

    원본(`KMeans(n_clusters=i, random_state=42)`)은 n_init을 명시하지 않아
    설치된 sklearn(1.2.1)의 기본값인 n_init=10이 그대로 적용된다. 이 테스트베드는
    원본과 달리 experience(라운드)마다 이 elbow 탐색을 반복하므로(원본은 한
    학습 세션당 한 번), 후보 6개 × n_init=10 조합이 라운드마다 반복되면 비용이
    크다. 그래서 **elbow 탐색 단계**(어떤 K가 좋은지 WCSS 추세만 보면 되는
    단계)는 n_init=3으로 줄이고, **실제 pseudo-label에 쓰이는 최종 fit**만
    원본과 동일하게 n_init=10(기본값)을 유지한다 — 클러스터 배정 품질에
    영향을 주는 부분은 원본 그대로 두고, 탐색용 반복만 줄인 것이다.

    **2026-08-11 추가(fit_sample_size)**: K 후보가 원 논문 리스트(최대 2000)로
    돌아가면서, CICIDS2018처럼 라운드당 선택 데이터가 수십만 건인 경우
    `KMeans(n_clusters=2000).fit()`을 experience당 7회(elbow 6 + 최종 1) 반복
    하는 비용이 감당 불가능해진다(GPM의 `activation_sample_size` — 정확히
    같은 종류의 문제를 같은 방식으로 이미 해결한 전례). `fit_sample_size`가
    주어지고 데이터가 그보다 크면, **elbow 탐색과 최종 fit 모두** 무작위
    부분표본에서 수행한다 — 이렇게 찾은 중심점(cluster_centers_)은 이후
    `_pseudo_labels_for_batch`가 `torch.cdist`로 전체 데이터를 배정하는 데
    그대로 쓰이므로(그 경로는 이미 GPU 벡터화·검증 완료), 부분표본으로 찾은
    중심점이 합리적이기만 하면 최종 라벨링 자체는 전체 데이터에 적용된다.
    합성 데이터 검증과 실콤보 A/B는 docs/metric_justification.md 참고.
    """
    from sklearn.cluster import KMeans
    from kneed import KneeLocator

    n = len(data_np)
    if fit_sample_size is not None and n > fit_sample_size:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(n, size=fit_sample_size, replace=False)
        fit_data = data_np[sample_idx]
    else:
        fit_data = data_np
    n_fit = len(fit_data)

    valid = [k for k in candidates if k < n_fit]
    if not valid:
        valid = [max(2, min(n_fit, 2))]

    wcss = []
    for k in valid:
        km = KMeans(n_clusters=k, random_state=seed, n_init=3)
        km.fit(fit_data)
        wcss.append(km.inertia_)

    optimal_k = valid[-1]
    if len(valid) > 2:
        kneedle = KneeLocator(valid, wcss, curve="convex", direction="decreasing")
        if kneedle.elbow is not None:
            optimal_k = kneedle.elbow

    final_km = KMeans(n_clusters=optimal_k, random_state=seed)  # n_init 기본값(10) 유지
    final_km.fit(fit_data)
    return final_km


def _metric_loss(z: torch.Tensor, pseudo_labels: torch.Tensor, margin: float = 2.0) -> torch.Tensor:
    """2026-08-26 발견·수정(4개 논문 컴포넌트 전수 재감사 — CONFIRMED,
    단위 테스트로 실측 확인) — pseudo_labels가 이 배치 안에서 전부 같은
    값이면(같은 클러스터 판정, `same`이 전부 1) 이 손실은 무조건적으로
    거리를 좁히기만 한다(`same*dist` 항만 활성화, saturate 없음) — 실측:
    동질 배치에서 loss=4.43, gradient norm 0.5로 실제 임베딩을 뭉갠다.
    원문의 실제 미이너(`TripletMarginMiner(type_of_triplets="semihard")`,
    `CND_IDS.py:38-39,76-78`)는 anchor당 양성/음성이 둘 다 있어야 triplet을
    만들 수 있어, pseudo-label이 배치 전체에서 동질적이면 유효 triplet이
    0개가 되어 손실이 조용히 0이 된다 — 이 구현은 같은 상황에서 오히려
    적극적으로 임베딩을 뭉개 원문과 질적으로 다르게 반응하고 있었다.
    `on_experience_start`의 정상 참조 풀 수정(위 모듈 docstring 참고)
    이후에도 R2L/U2R류 라운드는 pseudo_ratio가 여전히 0.96~0.99대라 이
    경로가 실제로 자주 발동함을 재확인했다. `pytorch_metric_learning`
    (새 의존성 도입 위험 — 이 프로젝트의 기존 정책)을 설치하지 않고도,
    배치 전체의 pseudo_labels가 단일 값이면(유효 triplet이 있을 수 없는
    경우) 원문처럼 손실을 0으로 만든다 — 새 의존성 없이 원문의 "0개
    triplet → 0 loss" 의미만 가져온 것이다."""
    n = z.shape[0]
    half = n // 2
    if half == 0:
        return z.sum() * 0.0
    if len(pseudo_labels.unique()) < 2:
        return z.sum() * 0.0
    left, right = z[:half], z[half:2 * half]
    left_l, right_l = pseudo_labels[:half], pseudo_labels[half:2 * half]
    dist = torch.norm(left - right, dim=1)
    same = (left_l == right_l).float()
    loss = same * dist + (1 - same) * F.relu(margin - dist)
    return loss.mean()


class CNDIDSAntiForgetting(BaseAntiForgetting):
    backbone_type = "autoencoder"

    def __init__(self, lambda_r: float = 0.1, lambda_cl: float = 0.1,
                 triplet_margin: float = 2.0,
                 cluster_fit_sample_size: Optional[int] = None,
                 max_normal_ref: int = 5000):
        self.lambda_r = lambda_r
        self.lambda_cl = lambda_cl
        self.margin = triplet_margin
        self.cluster_fit_sample_size = cluster_fit_sample_size
        self.max_normal_ref = max_normal_ref
        self._teachers: List[BaseCLModel] = []
        self._kmeans = None
        self._normal_cluster_ids: Set[int] = set()
        # 2026-08-26 추가 — normal_subset을 인스턴스 수명 전체에 걸쳐 누적한
        # 정상 참조 풀(위 모듈 docstring "2026-08-26" 절 참고). 매 라운드
        # 그 라운드 자신의 normal_subset "만"으로 참조를 다시 정의하면
        # 공격이 희귀한 라운드에서 참조가 라운드 전체를 뒤덮는 문제가 있어,
        # CADE의 `_category_refs`와 같은 패턴으로 누적한다.
        self._normal_ref_pool: Optional[torch.Tensor] = None
        # _pseudo_labels_for_batch()가 매 미니배치마다 다시 계산할 수 있도록
        # 캐시해두는 값들 — on_experience_start()에서 한 번만 채운다.
        self._centers: Optional[torch.Tensor] = None
        self._is_normal_cluster: Optional[torch.Tensor] = None
        # PRD 15.4절 — Track B pseudo-label 균형 확인(경고 전용)을 위해 마지막
        # compute_loss 호출에서 생성된 pseudo-label의 다수 클래스 비율을 기록한다.
        self.last_pseudo_label_ratio: Optional[float] = None

    def on_experience_start(self, selected_data: torch.Tensor,
                             normal_subset: torch.Tensor) -> None:
        """CND-IDS CND_IDS.py:fit() 진입부와 동일 — experience(라운드) 시작 시
        원본 입력 공간에서 K-Means를 한 번 학습하고, 정상 참조 데이터가 속하는
        클러스터 ID 집합을 구해둔다. CLClient가 학습 루프(step 4) 이전에
        호출한다. `normal_subset`은 이번 라운드 라벨 예산 안에서 선택된
        데이터 중 label=0인 것만 걸러낸 것이다(비어있지 않을 때만 호출됨 —
        cl_client.py 참고). 클러스터링 자체는 원문처럼 이번 라운드 데이터로
        새로 fit하지만, "어떤 클러스터가 정상인가" 판정은 이번 라운드
        normal_subset이 아니라 누적 정상 참조 풀로 한다(위 모듈 docstring
        "2026-08-26" 절 참고)."""
        data_np = selected_data.detach().cpu().numpy()
        self._kmeans = _elbow_kmeans_fit(
            data_np, list(_CLUSTER_K_CANDIDATES),
            fit_sample_size=self.cluster_fit_sample_size)

        if self._normal_ref_pool is not None:
            combined_ref = torch.cat([self._normal_ref_pool, normal_subset], dim=0)
        else:
            combined_ref = normal_subset
        if len(combined_ref) > self.max_normal_ref:
            # 2026-08-26 발견·수정(재감사) — `combined_ref[-cap:]`(꼬리
            # 슬라이싱)는 사실상 "누적"이 전혀 아니었다: Track B는
            # label_budget 없이 experience 전체를 쓰므로 `normal_subset`
            # 자체가 매 라운드 이미 cap(5000)보다 크다(NSL-KDD 13468~59396,
            # UNSW-NB15 11200~130541건) — 즉 `combined_ref`의 마지막
            # `cap`개는 100% 이번 라운드 자신의 데이터였다(실측: 5라운드
            # 전체에서 이전 라운드 표본 생존율 0%). 처음 관찰했던 개선
            # (f1 0.889 등)은 "라운드를 넘어 기억한다"가 아니라 "정상
            # 참조 표본 수가 우연히 줄어들어 K-Means 클러스터가 덜
            # 뒤덮인다"는 전혀 다른, 원 논문과 무관한 부작용이었다.
            # 무작위 표본으로 바꿔 combined_ref(이전 누적 + 이번 라운드)
            # 전체에서 균등하게 뽑는다 — 이러면 이전 라운드 표본도 실제로
            # (비율만큼) 살아남아 원래 의도한 "과거 정상도 계속 기억한다"
            # 는 성질이 실제로 성립한다.
            perm = torch.randperm(len(combined_ref), device=combined_ref.device)
            combined_ref = combined_ref[perm[:self.max_normal_ref]]
        self._normal_ref_pool = combined_ref.detach()

        ref_np = self._normal_ref_pool.cpu().numpy()
        ref_clusters = self._kmeans.predict(ref_np)
        self._normal_cluster_ids = set(ref_clusters.tolist())

        # 실측 발견(2026-08-05): 클러스터 배정을 매 미니배치마다
        # sklearn.predict()로 다시 계산하면(CPU 왕복 + 호출 오버헤드) 라운드당
        # 수만~수십만 번 호출이 반복되어 CICIDS2018 규모에서 감당 불가능할
        # 정도로 느려진다(K-means 스케일링 수정과 label_budget 제거가 겹치며
        # 처음으로 드러남). KMeans.predict()는 정의상 "유클리드 거리로 가장
        # 가까운 클러스터 중심 찾기"이므로, 중심점(cluster_centers_)만 라운드당
        # 한 번 GPU 텐서로 캐시해두면 torch.cdist(...).argmin(dim=1)로
        # 수학적으로 동일한 결과를 GPU에서 직접 계산할 수 있다(로컬에서
        # sklearn.predict() 대 이 방식을 float32/float64 양쪽으로 대조해 완전히
        # 일치함을 확인함, docs/metric_justification.md 참고). elbow 탐색·
        # 최종 fit 자체는 원본과 동일하게 sklearn/CPU를 그대로 쓴다 — 반복
        # 호출되는 predict()만 대체한다.
        device = selected_data.device
        self._centers = torch.tensor(
            self._kmeans.cluster_centers_, dtype=selected_data.dtype, device=device)
        n_clusters = self._centers.shape[0]
        is_normal = torch.zeros(n_clusters, dtype=torch.bool, device=device)
        normal_idx = [c for c in self._normal_cluster_ids if 0 <= c < n_clusters]
        if normal_idx:
            is_normal[torch.tensor(normal_idx, dtype=torch.long, device=device)] = True
        self._is_normal_cluster = is_normal

    def _pseudo_labels_for_batch(self, data: torch.Tensor) -> torch.Tensor:
        if self._centers is None:
            # on_experience_start이 호출되지 않은 경우(단위 테스트 등)의 안전
            # 폴백 — 전부 "정상"으로 간주(원본의 "정상 참조와 다른 클러스터가
            # 없으면 전부 정상" 극단 상황과 동일하게 처리).
            return torch.zeros(len(data), dtype=torch.long, device=data.device)
        cluster_ids = torch.cdist(data, self._centers).argmin(dim=1)
        pseudo = (~self._is_normal_cluster[cluster_ids]).long()
        return pseudo

    def compute_loss(self, model: BaseCLModel,
                      new_batch: Tuple[torch.Tensor, torch.Tensor],
                      replay_batch: Optional[Tuple[torch.Tensor, torch.Tensor]]
                      ) -> torch.Tensor:
        data, _labels = new_batch  # _labels는 의도적으로 미사용 (라벨-프리)
        z, x_hat, _ = model(data)
        recon_loss = F.mse_loss(x_hat, data)

        pseudo = self._pseudo_labels_for_batch(data)
        ratio = pseudo.float().mean().item()
        self.last_pseudo_label_ratio = max(ratio, 1.0 - ratio)
        metric_loss = _metric_loss(z, pseudo, self.margin)

        loss = metric_loss + self.lambda_r * recon_loss

        if replay_batch is not None and replay_batch[0] is not None:
            r_data, _ = replay_batch
            _, r_x_hat, _ = model(r_data)
            loss = loss + self.lambda_r * F.mse_loss(r_x_hat, r_data)

        if self._teachers:
            # CND_IDS.py:54-69 LwFloss() — 누적된 과거 스냅샷 각각과 개별
            # MSE(가중치 lambda_r)를 구해 합산한 뒤, 그 합계에 다시 lambda_cl을
            # 곱한다(:159,180) — 항 하나당 실효 가중치는 lambda_r*lambda_cl.
            lwf_sum = z.new_zeros(())
            for teacher in self._teachers:
                with torch.no_grad():
                    teacher_z, _, _ = teacher(data)
                lwf_sum = lwf_sum + self.lambda_r * F.mse_loss(z, teacher_z)
            loss = loss + self.lambda_cl * lwf_sum

        return loss

    def on_task_end(self, model: BaseCLModel) -> None:
        # CND_IDS.py:195 self.old_models.append(deepclone(self)) — 매
        # experience 종료 시 누적만 하고 절대 비우지 않는다(직전 1개로
        # 덮어쓰지 않음, 2026-08-12 정정).
        teacher = copy.deepcopy(model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self._teachers.append(teacher)
