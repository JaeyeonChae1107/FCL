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
나눠 쌍을 구성, margin 기반)로 직접 구현했다 — 알고리즘의 본질(같은
클러스터는 가깝게, 다른 클러스터는 margin 이상 멀게)은 동일하게 유지한다.

**라벨-프리(label-free) 준수**: `new_batch`의 `selected_labels`는 손실 계산에
전혀 쓰지 않는다(PRD 12.5절 명시 사항) — 위 확인대로 CND-IDS 원본도 pseudo-
label 생성에 공격 라벨을 쓰지 않으므로 이 원칙과 실제로 상충하지 않는다.
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
    n = z.shape[0]
    half = n // 2
    if half == 0:
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
                 cluster_fit_sample_size: Optional[int] = None):
        self.lambda_r = lambda_r
        self.lambda_cl = lambda_cl
        self.margin = triplet_margin
        self.cluster_fit_sample_size = cluster_fit_sample_size
        self._teacher: Optional[BaseCLModel] = None
        self._kmeans = None
        self._normal_cluster_ids: Set[int] = set()
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
        cl_client.py 참고)."""
        data_np = selected_data.detach().cpu().numpy()
        self._kmeans = _elbow_kmeans_fit(
            data_np, list(_CLUSTER_K_CANDIDATES),
            fit_sample_size=self.cluster_fit_sample_size)

        ref_np = normal_subset.detach().cpu().numpy()
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

        if self._teacher is not None:
            with torch.no_grad():
                teacher_z, _, _ = self._teacher(data)
            lwf_loss = F.mse_loss(z, teacher_z)
            loss = loss + self.lambda_cl * lwf_loss

        return loss

    def on_task_end(self, model: BaseCLModel) -> None:
        self._teacher = copy.deepcopy(model)
        self._teacher.eval()
        for p in self._teacher.parameters():
            p.requires_grad_(False)
