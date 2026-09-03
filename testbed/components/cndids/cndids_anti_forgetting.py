"""CNDIDSAntiForgetting — 결합 손실 anti-forgetting (PRD 4절/12.5절).

CND-IDS 원 논문 근거 (CND-IDS/FeatureExtractors/CND_IDS.py:161-166):
  loss = metric_loss(L_CS) + reg_strength * reconstruction_loss(L_R)
         + LwF_strength * lossLwF(L_CL)
  기본 reg_strength=0.1(lambda_r), LwF_strength=0.1(lambda_cl) (CND_IDS.py:43,45).
  metric_loss는 TripletMarginLoss(margin=2, semihard mining)를 pseudo-label
  클러스터(0=정상, 1=신규) 간 거리에 적용한다(CND_IDS.py:76-78).

**Pseudo-label 생성**: 원본(`FeatureExtractors/CND_IDS.py:105-115`, 실제
clusterer는 `FeatureExtractors/modules/K_Means.py`)의 메커니즘:
  1. `cluster_labels = self.labeler.fit_transform(x)` — experience의 원본
     입력 x(인코딩 전)에 elbow-선택 K-Means를 한 번 적용(후보
     K=[100,300,500,1000,2000], `modules/K_Means.py:18`).
  2. `init_normal_labels = self.labeler.transform(init_normal)` — 알려진
     정상 참조 데이터(`datastream.init_normal`)가 속하는 클러스터 ID 집합.
  3. `y = [0 if i in init_normal_labels else 1 for i in cluster_labels]` —
     그 클러스터 집합에 속하면 정상(0), 아니면 신규/이상(1). 공격 라벨은
     쓰지 않는다 — "이 클러스터에 알려진 정상 참조가 있는가"만 본다.
  (같은 폴더의 `AnomolyDetectors/K_Means.py`는 이름은 같지만 별개의
  standalone anomaly-scorer 베이스라인이며 실제 라벨이 섞인 캘리브레이션
  서브셋을 쓴다 — CND_IDS.py의 pseudo-labeling과는 무관.)

이 구현은 위 메커니즘을 그대로 따른다. K 후보 [100,300,500,1000,2000]도
원문 그대로 데이터셋 무관 고정값이다(`_CLUSTER_K_CANDIDATES` 참고). 이
테스트베드 고유의 조정 두 가지:
  - "알려진 정상 참조 데이터"로 이번 라운드 라벨 예산 안에서 선택된 데이터
    중 label=0인 것만 걸러 쓴다(`cl_client.py`의 `normal_subset`) —
    CND-IDS의 `datastream.init_normal`과 같은 역할.
  - 클러스터링은 experience(라운드)당 한 번만 수행하고
    (`on_experience_start`), 이후 각 미니배치에서는 이미 학습된 K-Means로
    predict만 한다 — 원본이 `fit()` 진입 시 한 번 클러스터링하고 그 결과를
    epoch 전체에서 재사용하는 것과 동일한 절차다.

`pytorch_metric_learning`은 새 의존성 설치 위험을 피하기 위해(이전 deepod
사고 참고, docs/metric_justification.md) TripletMarginLoss를 CADE의
contrastive pairing과 동일한 형태(배치를 반으로 나눠 쌍을 구성, margin
기반)로 직접 구현했다. 원문(`CND_IDS.py:38-39,76-78`)은
`TripletMarginMiner(type_of_triplets="semihard")`로 상대 마진 트리플릿을
쓰는데, `_metric_loss()`는 절대 마진 페어와이즈를 쓴다 — 방향은 같지만
손실의 수학적 형태가 달라 "같은 목표를 다른 손실 형태로 근사"한 것이다.

**라벨-프리(label-free) 준수**: `new_batch`의 `selected_labels`는 손실 계산에
쓰지 않는다(PRD 12.5절) — CND-IDS 원본도 pseudo-label 생성에 공격 라벨을
쓰지 않으므로 이 원칙과 상충하지 않는다.

**teacher를 직전 1개만 유지하던 결함**: CND-IDS 원문(`CND_IDS.py:42,54-69,
195`)은 `self.old_models` 리스트에 매 experience 종료 시 그 시점 모델
전체를 `deepclone`해 계속 추가만 하고 비우지 않는다. `LwFloss()`는 이
리스트의 모든 과거 모델에 대해 개별적으로 `MSE(현재 encoder 출력,
old_model(현재 배치))`를 계산해 합산한다(`CND_IDS.py:57-66`) — experience
3에서는 experience 0·1·2 스냅샷 세 개 전부와 거리를 잰다. 이전 구현은
`self._teacher`를 매 `on_task_end`마다 덮어써서 최근 스냅샷 하나만
비교했다 — 이제 리스트로 누적한다.

가중치 적용도 정정: `LwFloss()` 내부에서 개별 old-model 항마다 이미
`reg_strength`(=lambda_r)를 곱하고(`:66`), 그 합계에 호출부가 다시
`LwF_strength`(=lambda_cl)를 곱한다(`:159,180`) — 항 하나당 실효 가중치는
`lambda_r * lambda_cl`이지 `lambda_cl` 단독이 아니다.

**정상 참조가 희귀 라운드에서 라운드 전체를 뒤덮는 문제**: `on_experience_
start`가 받는 `normal_subset`(=`selected_data[selected_labels==0]`, Track
B는 experience 전체가 selected_data)은 원문의 `datastream.init_normal`
(스트림 시작 전 한 번 확정되는 고정 참조)과 달리 이번 라운드 자신의 공격
비율에 크기가 좌우된다. R2L 라운드(실제 공격 6.9%)는 `normal_subset`이
라운드의 93.1%, U2R 라운드(실제 공격 0.38%)는 99.6%까지 차지한다. 참조가
라운드 거의 전체와 겹치면 K-Means 클러스터 대부분이 참조 데이터를 하나는
포함해 거의 모든 클러스터가 "정상"으로 판정된다 — 실측(NSL-KDD 5라운드
전체): pseudo-label 다수 클래스 비율이 R2L 0.9844, U2R 1.0000까지 치솟고
U2R 라운드 직후 0.870이던 정확도가 다음 라운드 0.707로 하락했다.

`normal_subset`을 인스턴스 수명 전체에 걸쳐 누적하는 별도 풀
(`_normal_ref_pool`, 최근 `max_normal_ref`개 캡)로 바꿨다. 클러스터링
자체(원문처럼 그 라운드 원본 입력으로 매 라운드 새로 fit)는 그대로 두고,
"이 클러스터가 정상인가" 판정에만 누적 풀을 쓴다.

처음 구현(`combined_ref[-cap:]`, 꼬리 슬라이싱)은 실제로는 거의 누적되지
않았다 — Track B는 label_budget 없이 experience 전체를 쓰므로
`normal_subset` 자체가 이미 매 라운드 cap(5000)보다 크다(NSL-KDD
13,468~59,396건). `combined_ref`의 마지막 `cap`개는 항상 100% 이번 라운드
자신의 데이터였다(실측: 5라운드 전체에서 이전 라운드 표본 생존율 0%). 그런데
개선(f1→0.889 등)은 실제로 관찰됐는데, 원인은 "라운드를 넘어 기억한다"가
아니라 "참조 표본 수 자체가 줄면 K-Means 클러스터가 덜 뒤덮인다"는 우연한
부작용이었다(같은 클러스터링에 참조만 전체 vs 5000건 무작위로 비교: R2L
pseudo_ratio 0.9720→0.9634, U2R 0.9998→0.9962). 꼬리 슬라이싱을 무작위
표본(`torch.randperm`)으로 바꿔 이전 라운드 표본도 비율만큼 살아남게 했다
— 이러면 "우연한 표본 수 축소 효과"와 "원래 의도한 누적 효과"가 함께
작동한다. 다만 CND-IDS 원문의 `datastream.init_normal`(스트림 시작 전 한
번 고정)과는 여전히 다르다 — 이건 라운드가 지날수록 갱신되는 근사적
저수지 표본이다. A/B 실측 결과는 `docs/metric_justification.md` 참고.
"""

import copy
from typing import List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from testbed.base.anti_forgetting import BaseAntiForgetting
from testbed.base.models import BaseCLModel

# CND-IDS 원본(modules/K_Means.py:18)은 [100,300,500,1000,2000]을 데이터셋
# 무관하게 그대로 쓴다 — 스케일링 공식 자체가 없다. Track B가 원 논문처럼
# experience 전체를 그대로 쓰도록 바뀐 뒤로는 축소할 이유가 없어 원 논문
# 리스트를 그대로 채택한다.
_CLUSTER_K_CANDIDATES = (100, 300, 500, 1000, 2000)


def _fit_one_candidate(fit_data: np.ndarray, k: int, seed: int) -> float:
    """`_elbow_kmeans_fit()`의 elbow 탐색 루프 한 스텝 — 후보 K 하나를 fit해
    WCSS(inertia_)만 반환한다. 후보끼리 서로 완전히 독립된 계산이라
    `joblib.Parallel`로 병렬 실행해도 **elbow가 어떤 K를 선택하는지**(선택된
    optimal_k, WCSS 추세)는 순차 실행과 동일하다 — 실측 확인(같은 데이터,
    elbow_n_jobs=1 vs 4, n_clusters/inertia 완전 일치).

    **정정(2026-09-04, 실측 후)**: 다만 이 정정은 "cluster_centers_ 좌표
    자체까지 완전히 동일하다"는 뜻은 아니다 — 실측해보니 `elbow_n_jobs=1`만
    써서 **병렬화가 전혀 관여하지 않는 상태**로 이 함수를 두 번 순차
    호출해도(`_elbow_kmeans_fit(..., elbow_n_jobs=1)` 두 번) `n_clusters`/
    `inertia_`는 완전히 같은데 `cluster_centers_` 좌표는 다르게 나왔다 —
    이건 이 병렬화 변경과 무관하게 sklearn KMeans 자체가 가진 기존
    비결정성이다(멀티스레드 Lloyd 알고리즘의 부동소수점 합산 순서가
    `random_state` 고정과 별개로 실행마다 미세하게 달라질 수 있음, sklearn에
    알려진 특성). 즉 이 병렬화가 새로운 비결정성을 만드는 게 아니라, 이미
    있던 비결정성 위에 얹히는 것뿐이다."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=seed, n_init=3)
    km.fit(fit_data)
    return float(km.inertia_)


def _elbow_kmeans_fit(data_np: np.ndarray, candidates: List[int], seed: int = 42,
                       fit_sample_size: Optional[int] = None, elbow_n_jobs: int = 1):
    """CND-IDS modules/K_Means.py:fit()과 동일한 elbow 선택 절차.

    원본(`KMeans(n_clusters=i, random_state=42)`)은 n_init을 명시하지 않아
    설치된 sklearn(1.2.1)의 기본값인 n_init=10이 그대로 적용된다. 이 테스트베드는
    원본과 달리 experience(라운드)마다 이 elbow 탐색을 반복하므로(원본은 한
    학습 세션당 한 번), 후보 6개 × n_init=10 조합이 라운드마다 반복되면 비용이
    크다. elbow 탐색 단계(어떤 K가 좋은지 WCSS 추세만 보면 되는 단계)는
    n_init=3으로 줄이고, 실제 pseudo-label에 쓰이는 최종 fit만 원본과 동일하게
    n_init=10(기본값)을 유지한다.

    `fit_sample_size`: K 후보가 원 논문 리스트(최대 2000)로 돌아가면서,
    CICIDS2018처럼 라운드당 선택 데이터가 수십만 건인 경우
    `KMeans(n_clusters=2000).fit()`을 experience당 7회(elbow 6 + 최종 1)
    반복하는 비용이 감당 불가능해진다(GPM의 `activation_sample_size`와
    같은 종류의 문제). 주어지고 데이터가 그보다 크면 elbow 탐색과 최종 fit
    모두 무작위 부분표본에서 수행한다 — 찾은 중심점(cluster_centers_)은
    이후 `_pseudo_labels_for_batch`가 `torch.cdist`로 전체 데이터를
    배정하는 데 쓰이므로, 중심점이 합리적이기만 하면 최종 라벨링은 전체
    데이터에 적용된다.

    `elbow_n_jobs`(2026-09-04 추가): elbow 탐색의 K 후보들(최대 5개)은
    서로 완전히 독립적인 계산인데(`_fit_one_candidate` 참고) 지금까지는
    Python for-loop로 하나씩 순차 실행했다 — `cndids.yaml` 주석의 실측치
    (K=2000 기준 라운드당 약 150초)가 이 순차 실행 시간이다. 이 값이
    1보다 크면 `joblib.Parallel`로 후보들을 동시에 fit한다 — 각 후보는
    서로 다른 K로 완전히 독립적인 KMeans 인스턴스라 병렬로 돌려도 **어떤
    K가 최종 선택되는지**(optimal_k, WCSS 추세)는 순차 실행과 동일함을
    실측 확인했다. 단, 최종 `cluster_centers_`의 정확한 좌표는
    `elbow_n_jobs` 값과 무관하게(`elbow_n_jobs=1`끼리 비교해도) 실행마다
    미세하게 달라질 수 있다 — sklearn KMeans 자체의 기존 비결정성이지
    이 병렬화가 새로 만드는 문제가 아니다(`_fit_one_candidate` docstring
    "2026-09-04 정정" 절 참고). 기본값 1(순차, 기존과 동일)로 둔 이유는
    `grid_runner.py --shard`로 여러 프로세스를 동시에 띄우는 경우와
    합쳐지면 코어를 과다 점유(oversubscription)할 수 있어서다 — 이 값은
    `component_hparams/cndids.yaml`에서 명시적으로 켜야 한다(그 파일의
    주석에 `--shard`와 함께 쓸 때의 배분 기준을 남겨둔다).
    """
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

    if elbow_n_jobs > 1 and len(valid) > 1:
        from joblib import Parallel, delayed
        wcss = Parallel(n_jobs=min(elbow_n_jobs, len(valid)))(
            delayed(_fit_one_candidate)(fit_data, k, seed) for k in valid)
    else:
        wcss = [_fit_one_candidate(fit_data, k, seed) for k in valid]

    optimal_k = valid[-1]
    if len(valid) > 2:
        kneedle = KneeLocator(valid, wcss, curve="convex", direction="decreasing")
        if kneedle.elbow is not None:
            optimal_k = kneedle.elbow

    from sklearn.cluster import KMeans
    final_km = KMeans(n_clusters=optimal_k, random_state=seed)  # n_init 기본값(10) 유지
    final_km.fit(fit_data)
    return final_km


def _metric_loss(z: torch.Tensor, pseudo_labels: torch.Tensor, margin: float = 2.0) -> torch.Tensor:
    """pseudo_labels가 배치 안에서 전부 같은 값이면(같은 클러스터 판정)
    이 손실은 거리를 좁히기만 한다(`same*dist` 항만 활성화, saturate
    없음) — 실측: 동질 배치에서 loss=4.43, gradient norm 0.5로 실제
    임베딩을 뭉갠다. 원문의 미이너(`TripletMarginMiner(type_of_triplets=
    "semihard")`, `CND_IDS.py:38-39,76-78`)는 anchor당 양성/음성이 둘 다
    있어야 triplet을 만들 수 있어, pseudo-label이 배치 전체에서 동질적이면
    유효 triplet이 0개가 되어 손실이 조용히 0이 된다. `on_experience_
    start`의 정상 참조 풀 수정 이후에도 R2L/U2R류 라운드는 pseudo_ratio가
    여전히 0.96~0.99대라 이 경로가 자주 발동함을 재확인했다.
    `pytorch_metric_learning`을 설치하지 않고도, 배치 전체의 pseudo_labels가
    단일 값이면(유효 triplet이 있을 수 없는 경우) 원문처럼 손실을 0으로
    만든다."""
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
                 max_normal_ref: int = 5000, elbow_n_jobs: int = 1):
        self.lambda_r = lambda_r
        self.lambda_cl = lambda_cl
        self.margin = triplet_margin
        self.cluster_fit_sample_size = cluster_fit_sample_size
        self.max_normal_ref = max_normal_ref
        # 2026-09-04 추가 — elbow K 후보 탐색을 joblib으로 병렬화할 때 쓸
        # 워커 수. 기본값 1(순차, 기존과 동일) — _elbow_kmeans_fit() 모듈
        # docstring 참고.
        self.elbow_n_jobs = elbow_n_jobs
        self._teachers: List[BaseCLModel] = []
        self._kmeans = None
        self._normal_cluster_ids: Set[int] = set()
        # normal_subset을 인스턴스 수명 전체에 걸쳐 누적한 정상 참조 풀
        # (모듈 docstring 참고) — CADE의 `_category_refs`와 같은 패턴.
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
        normal_subset이 아니라 누적 정상 참조 풀로 한다(모듈 docstring
        참고)."""
        data_np = selected_data.detach().cpu().numpy()
        self._kmeans = _elbow_kmeans_fit(
            data_np, list(_CLUSTER_K_CANDIDATES),
            fit_sample_size=self.cluster_fit_sample_size,
            elbow_n_jobs=self.elbow_n_jobs)

        if self._normal_ref_pool is not None:
            combined_ref = torch.cat([self._normal_ref_pool, normal_subset], dim=0)
        else:
            combined_ref = normal_subset
        if len(combined_ref) > self.max_normal_ref:
            # 꼬리 슬라이싱 대신 무작위 표본으로 캡을 적용한다(모듈
            # docstring 참고) — combined_ref(이전 누적 + 이번 라운드)
            # 전체에서 균등하게 뽑으므로 이전 라운드 표본도 비율만큼
            # 살아남는다.
            perm = torch.randperm(len(combined_ref), device=combined_ref.device)
            combined_ref = combined_ref[perm[:self.max_normal_ref]]
        self._normal_ref_pool = combined_ref.detach()

        ref_np = self._normal_ref_pool.cpu().numpy()
        ref_clusters = self._kmeans.predict(ref_np)
        self._normal_cluster_ids = set(ref_clusters.tolist())

        # 클러스터 배정을 매 미니배치마다 sklearn.predict()로 다시 계산하면
        # (CPU 왕복 + 호출 오버헤드) 라운드당 수만~수십만 번 호출이 반복되어
        # CICIDS2018 규모에서 감당 불가능하게 느려진다. KMeans.predict()는
        # 정의상 "유클리드 거리로 가장 가까운 클러스터 중심 찾기"이므로,
        # 중심점(cluster_centers_)만 라운드당 한 번 GPU 텐서로 캐시해두면
        # torch.cdist(...).argmin(dim=1)로 수학적으로 동일한 결과를 GPU에서
        # 직접 계산할 수 있다(sklearn.predict() 대비 float32/float64 양쪽
        # 완전 일치 확인, docs/metric_justification.md 참고). elbow 탐색·
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
        # experience 종료 시 누적만 하고 절대 비우지 않는다.
        teacher = copy.deepcopy(model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self._teachers.append(teacher)
