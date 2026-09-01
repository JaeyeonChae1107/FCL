"""GPM (Gradient Projection Memory) anti-forgetting.

SPIDER 저장소 코드를 확보하지 못해(testbed/docs/metric_justification.md 참고),
GPM 원 논문(Saha, Garg, Roy, "Gradient Projection Memory for Continual
Learning", ICLR 2021)을 근거로 이 테스트베드의 BaseAntiForgetting 계약
(PRD 12.5절)에 맞춰 작성했다. 2026-07-30 최종 재검토에서 GPM 저자 공식
코드(https://github.com/sahagobinda/GPM, `main_pmnist.py`)를 실제로 찾아
대조했고, 한 번은 문자 그대로 맞췄다가 실측 회귀를 발견해 다시 되돌렸다 —
`_compute_basis()`의 주석에 그 경위가 자세히 남아있다. 최종적으로 아래
1·4번은 공식 코드와 일치하고, 2·3번은 의도적으로 다르다:

핵심 알고리즘 (공식 코드 `main_pmnist.py` 대조 완료):
  1. 태스크(experience) 종료 시(on_task_end), 그 태스크에서 실제로 학습에
     쓰인 selected_data로 각 Linear 레이어의 입력 activation 행렬을 수집한다.
  2. 레이어별 activation 행렬에 SVD를 적용해 누적 에너지 비율이
     activation_threshold 이상이 되는 최소 개수(+1 여유)의 우특이벡터를
     이번 태스크의 기저로 채택한다. **공식 코드는 평균을 빼지 않은 원본
     activation에 바로 SVD를 적용하고 `+1` 여유도 없다.** 문자 그대로
     맞춰봤더니 NSL-KDD에서 `af=gpm+as=cade_mad` 일부 조합(ss=ssf 계열)이
     exp1에서 분류기가 완전히 한 클래스로 퇴화하는 회귀가 실측으로
     확인됐다(30개 gpm 조합 전수검사: nsl-kdd 2/30 실패, unsw-nb15 0/30) —
     평균 중심화와 `+1` 여유 중 어느 하나만 복원해도 통과했다. 이 얕은
     은닉층 1개짜리 공유 아키텍처에서는 공식 코드의 문자 그대로의 계산이
     기저를 지나치게 작게(또는 ReLU의 항상-양수 편향 방향으로) 만들어
     분류기 학습에 필요한 그래디언트까지 사영으로 지워버리는 경계 사례를
     만든다 — 그래서 평균 중심화와 `+1` 여유 둘 다 원래대로 유지한다.
  3. 기존에 저장된 기저와 새 기저를 concat 후 QR 분해로 재직교화해 누적
     기저(GPM memory)를 갱신한다. **공식 코드는 `np.hstack`으로 이어붙이기만
     하고 QR 재직교화를 하지 않는다** — 이 테스트베드는 5개 experience에
     걸쳐 기저가 계속 누적되는데, 서로 다른 태스크의 SVD 결과를 단순히
     이어붙이기만 하면(태스크 간 직교성이 보장되지 않아) `basis @ basis.T`가
     더 이상 참된 직교 사영행렬이 되지 못하는 문제가 생길 수 있어 QR
     재직교화를 의도적으로 유지한다.
  4. 이후 태스크 학습 시 backward() 직후 project_gradients()가 각 레이어
     가중치의 gradient에서 누적 기저 방향 성분을 제거한다:
       grad_proj = grad - grad @ basis @ basis.T
     (공식 코드: `Uf = feature_list[i] @ feature_list[i].T`을 미리 계산해
     동일한 사영을 수행 — 이 부분은 일치한다.)

**요약**: 이 테스트베드의 목적(0절 — 논문 재현이 아니라 재조합·비교)상,
"공식 코드와 문자 그대로 같다"보다 "조합이 실제로 붕괴하지 않고 작동한다"를
우선했다. 공식 코드와 다른 지점(2·3번)은 전부 실측 근거와 함께 정직하게
기록해뒀다 — 몰라서 다른 게 아니라 알고도 다르게 한 것이다.

레지스트리 키는 "gpm" (PRD 4.1절 — 폴더는 components/spider_gpm/).
GPM은 명시적 replay/정규화 항 없이 gradient projection만으로 이전 태스크를
보호하므로(원 논문 설계), compute_loss는 task loss만 계산한다.

**2026-08-14 발견·수정 — 기저가 라운드를 거듭하며 풀랭크에 도달**: 구조
전수 감사에서 실측(NSL-KDD, af=gpm, 5 experience)으로 `encoder.0`(121차원)이
exp4에서 정확히 121/121(풀랭크)에 도달해 그 레이어의 그래디언트가 마지막
라운드에 완전히 0이 되는 걸 확인했다. GPM 원 논문 Algorithm 2의 residual
projection(이미 누적된 기저 방향을 먼저 제거하고 남은 residual에만 SVD
적용) 단계가 이 구현엔 없어서일 거라 보고 **두 차례** 추가해봤다 — 1차는
잔차 자체의 에너지로 정규화(자체 구현 결함 가능성 있음), 2차는 이후 재감사
에이전트가 WebFetch로 공식 저장소(`sahagobinda/GPM`, `main_pmnist.py`)를
직접 가져와 확인해준 진짜 Eq-9 정규화(잔차를 빼기 전 원본 activation의
총 에너지로 정규화 + 누적치가 "기존 기저가 이미 설명한 비율"에서 시작)
그대로 재구현. **둘 다 A/B 실측으로 오히려 악화**됐다(1차: f1
0.7006→0.6440, bwt -0.108→-0.142; 2차, 즉 공식 그대로: f1 0.7006→0.5538,
bwt -0.108→-0.170 — 2차가 더 나쁨). 2차는 구현 결함 가능성을 배제한
채로도 나빠졌으므로, "구현이 미묘하게 달랐다"가 아니라 **residual
projection 자체가 이 아키텍처(얕은 공유 backbone, latent_dim=32, 5
experience만)에 안 맞는다**는 결론이 신뢰할 만하다 — 상세 원인 분석은
`_compute_basis()` 주석 참고. 대신 `max_basis_ratio` 상한만 추가해 어떤
레이어도 ambient dimension의 10%(기본값) 미만으로는 학습 가능 공간이
줄어들지 않도록 했다 — "풀랭크로 완전히 죽는" 치명적 실패만 막고, 그 외
기저 성장 방식은 실측으로 이미 검증된 기존(residual 없는) 로직을 그대로
유지한다.

**2026-08-26 발견·수정 — bias가 projection 대상에서 완전히 빠져 있던 치명적
결함**: 4개 논문 컴포넌트 전수 재감사(병렬 에이전트)에서, 이 테스트베드의
`project_gradients()`가 `module.weight.grad`만 사영하고 `module.bias.grad`는
전혀 건드리지 않는다는 걸 재확인했다. 공식 GPM 저장소(`main_pmnist.py:29-31`)
는 `nn.Linear(..., bias=False)`만 쓰는 아키텍처라 이 문제 자체가 없었는데,
이 테스트베드의 공유 `FCLAutoEncoder`(`base/models.py`)는 기본 `bias=True`를
쓴다 — GPM이 보호하려는 weight 방향의 그래디언트가, 보호받지 않는 bias
(특히 classifier의 판정 임계값을 직접 결정하는 스칼라 하나)로 우회해
빠져나갈 길이 열려 있었던 것이다.

**실측(NSL-KDD, 200 epoch/experience, 5라운드 전체 — 이전엔 스모크
테스트가 앞 2라운드만 봐서 발견되지 않았다, `smoke_test.py` 모듈 docstring
"2026-08-26" 절 참고)**: `dd=none/ss=random/mm=spider/af=gpm/as=none`이
f1=0.0053, recall=0.26%, pr_auc=0.5191(거의 무작위), bwt=-0.6280까지
붕괴 — 같은 조합에서 `af=none`(아무 망각방지도 안 함)이 f1=0.3541,
pr_auc=0.8205, bwt=-0.3085인 것과 비교하면 **GPM이 있는데 오히려 훨씬
나쁜** 역설적 결과였다. 라운드별 추적 결과 매 라운드 학습 손실은 거의
0까지 수렴하는데(그 라운드 자신에게는 완벽히 과적합) 전체 pooled 예측
양성 수는 7753→9257→1959→510→37(22544건 중)으로 라운드를 거듭할수록
단조 감소 — weight가 갈수록 강하게 사영되어 학습 가능 공간이 줄어들수록
(encoder.0 기저가 46→108/121까지 성장), classifier.bias가 그 압력을
흡수하듯 매 라운드 계속 흔들리며(델타 -0.0225,-0.0038,-0.0173,-0.0086)
결국 거의 모든 표본을 "정상"으로 판정하는 방향으로 표류했다.

`project_gradients()`/`_update_basis()`를 수정해 bias를 "항상 1인 입력
차원"으로 취급, weight와 하나로 증강한 행렬에 대해 동일한 기저로 사영한다
— GPM을 bias 있는 affine 레이어로 수학적으로 자연스럽게 확장한 것이다
(원 논문/공식 코드에는 없는, bias 아키텍처를 쓰는 이 테스트베드 고유의
확장). **다만 A/B 실측 결과 이 수정만으로는 위 붕괴가 전혀 해결되지
않았다**(f1 0.0053→0.0053, roc_auc는 오히려 0.265로 악화) — bias 미보호는
실재하는, 이론적으로 정당한 결함이지만 이 붕괴의 **원인이 아니었다**.
아래 "진짜 원인" 절 참고.

**2026-08-26 진짜 원인 확정 — GPM 자체 결함이 아니라 `as=none`(고정 0.5
임계값)과의 궁합 문제였다**: bias 수정이 안 통해서 계속 추적한 결과,
`af=gpm`이 만드는 gradient projection 압력 아래서 classifier의 출력
로짓 스케일이 라운드를 거듭하며 계속 커진다는 걸 확인했다(같은 콤보,
`clf.weight_norm`이 exp0→exp4에 걸쳐 3.08→6.16까지 거의 2배 성장,
`as=cade_mad`가 매 라운드 다시 계산하는 threshold도 1.76→2.67로 그에
맞춰 같이 커짐 — 스케일이 실제로 움직이고 있다는 직접 증거). `as=none`은
SSF/SPIDER 원 논문 그대로 **고정 0.5** sigmoid 임계값을 쓰는데(재보정
없음, `common_baselines.py` 참고), 이렇게 계속 커지는 로짓 스케일을
고정 임계값이 전혀 못 따라가면서 거의 모든 표본이 "정상"으로 판정되는
것이었다 — GPM의 gradient projection 메커니즘 자체가 고장난 게 아니다.

**결정적 A/B(NSL-KDD, `dd=none/ss=random/mm=none`, 200 epoch·5라운드
전체)**:
  - `af=gpm + as=none`(고정 임계값): f1=0.0054, recall=0.27%, pr_auc=0.711,
    roc_auc=0.603, bwt=-0.507
  - `af=gpm + as=cade_mad`(매 라운드 재보정 임계값): **f1=0.6655,
    recall=51.9%, pr_auc=0.884, roc_auc=0.852, bwt=-0.1405**
  - 비교: `af=none`(아무 망각방지 없음) + `as=none`: f1=0.2445, bwt=-0.4331
같은 데이터로 판정 방식만 바꿨을 뿐인데 f1이 0.0054→0.6655로 완전히
달라진다 — `as=cade_mad`로 재보정하면 GPM은 오히려 naive fine-tuning보다
**훨씬 덜 잊는다**(bwt -0.14 vs -0.43) — 이게 GPM이 원래 주장하는 효과다.

**결론과 조치**: `af=gpm`+`as=none` 조합은 두 컴포넌트 각각은 자기 원
논문대로 충실히 구현됐지만, **이 둘을 같이 쓰면 서로 안 맞는다**는 진짜
구조적 비호환성이 발견된 것이다(성능 문제가 아니라 "이 재조합이 목적에
부합하는가"의 문제, 사용자 확정 기준) — GPM 코드를 더 고쳐서 이 조합을
억지로 "고치지" 않는다. 대신 `smoke_test.py`(2026-08-26 절 참고)의
강화된 15.2 게이트(다수 클래스 비율 ≥0.97 실패 처리)와 신규 15.2b
게이트(roc_auc 역전 감지)가 이 조합을 정확히 실패로 잡아내 그리드에서
자동으로 제외하도록 했다 — 이게 정답이다: 안 맞는 조합은 억지로 성능을
끌어올리는 게 아니라 정직하게 무효로 걸러내는 것. `af=gpm`+`as=cade_mad`/
`as=none` 둘 다 `TRACK_A_GRID`에 남겨두되(현재 config는 그대로 유지),
전체 그리드 재실행 시 스모크 테스트가 `af=gpm`+`as=none` 조합들을 실제로
걸러내는지 확인한다.

**2026-07-30 CICIDS2018 전체 데이터 실행 중 CUDA OOM으로 발견한 버그**:
`compute_loss()`가 매 미니배치 스텝마다 그 배치를 `_pending_data`에 계속
누적했는데, `epochs_per_experience`(Track A 200)만큼 반복되는 각 epoch은
사실 같은 `selected_data`를 다시 섞어 도는 것뿐이라 200배 중복 누적이었다.
NSL-KDD/UNSW-NB15 규모에서는 문제없었지만 CICIDS2018 전체(선택 샘플 약
19만개) × 200 epoch에서 약 3860만 행(~11.8GB)까지 쌓여 실제로 CUDA OOM이
났다. `activation_sample_size`(기본 2000)로 상한을 둬서, experience당
최대 그만큼만 모으고 이후로는 더 안 쌓도록 고쳤다 — GPM 논문의 실제 취지
(활성화의 "대표 표본"으로 SVD 기저 계산)와도 일치한다.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from testbed.base.anti_forgetting import BaseAntiForgetting
from testbed.base.models import BaseCLModel


class GPMAntiForgetting(BaseAntiForgetting):
    backbone_type = "classifier"

    def __init__(self, activation_threshold: float = 0.97,
                 activation_sample_size: int = 2000,
                 max_basis_ratio: float = 0.9):
        self.activation_threshold = activation_threshold
        # 2026-08-14 추가 — residual projection을 넣어도(아래 _compute_basis
        # 참고) 레이어별로 기저가 결국 ambient dimension을 다 채우는 경우가
        # 실측으로 확인됐다(예: NSL-KDD classifier/decoder.0는 32/32 풀랭크
        # 도달, exp4). 기저가 전체 공간을 덮으면 project_gradients()의
        # `grad - grad@basis@basis.T`가 그 레이어 그래디언트를 완전히
        # 0으로 만들어버려 학습이 조용히 멈춘다 — GPM 원 논문이 상정한
        # "은닉층이 충분히 큰" 아키텍처와 달리, 이 테스트베드의 공유 백본은
        # latent_dim(NSL-KDD 기준 32)이 작아 5 experience 안에 이 경계
        # 상황에 실제로 도달한다. 항상 ambient dimension의
        # (1-max_basis_ratio) 비율만큼은 학습 가능하게 남겨둔다 — 이 값
        # 자체는 원 논문에 없는 이 테스트베드 고유의 안전장치다.
        self.max_basis_ratio = max_basis_ratio
        # 2026-07-30 추가: on_task_end에서 activation SVD 기저를 계산할 때 쓸
        # "대표 표본" 상한. 이전에는 compute_loss()가 매 미니배치 스텝마다
        # (epochs_per_experience=200 전체에 걸쳐) 그 배치를 그냥 계속
        # 누적했다 — 매 epoch은 같은 selected_data를 다시 섞어 도는 것뿐이라
        # 200배 중복 누적이었다. NSL-KDD/UNSW-NB15 규모(선택 샘플 2500~3500개)
        # 에서는 누적해도 수백MB라 안 터졌지만, CICIDS2018 전체 데이터(선택
        # 샘플 약 19만개) × 200 epoch에서는 약 3860만 행(~11.8GB)까지 쌓여
        # CUDA OOM으로 실측 확인됐다. GPM 논문의 실제 취지(활성화의 "대표
        # 표본"으로 SVD 기저 계산)에 맞춰, 한 번의 experience당 최대
        # activation_sample_size개까지만 모으고 그 이후로는 더 안 쌓는다.
        self.activation_sample_size = activation_sample_size
        self._basis: Dict[str, torch.Tensor] = {}
        self._pending_data: List[torch.Tensor] = []
        self._pending_count = 0

    def compute_loss(self, model: BaseCLModel,
                      new_batch: Tuple[torch.Tensor, torch.Tensor],
                      replay_batch: Optional[Tuple[torch.Tensor, torch.Tensor]]
                      ) -> torch.Tensor:
        data, labels = new_batch
        # SVD 기저(GPM 고유 메커니즘)는 원문 그대로 "이번 태스크 자신의
        # 데이터"만으로 계산한다 — replay_batch를 여기 섞으면 "이 태스크의
        # 대표 activation"이라는 의미 자체가 흐려진다.
        if self._pending_count < self.activation_sample_size:
            self._pending_data.append(data.detach())
            self._pending_count += len(data)
        _, _, logit = model(data)
        loss = F.binary_cross_entropy_with_logits(logit.squeeze(-1), labels.float())

        # 2026-08-26 발견·수정 — replay_batch를 지금까지 완전히 무시하고
        # 있었다. GPM 원 논문 자체는 리플레이가 필요 없다는 게 핵심 주장
        # (gradient projection이 리플레이를 대체)이라 mm=none과 결합됐을
        # 때(=원 논문 그대로의 "순정 GPM")는 이 무시가 맞다. 그런데
        # SPIDER 논문은 바로 그 GPM에 **별도의 유한 버퍼 리플레이**를
        # 추가한 것이 핵심 기여이므로(`spider_memory_manager.py` 참고,
        # mm=spider가 그 버퍼), af=gpm이 mm=spider와 결합됐을 때 그 버퍼를
        # 학습에 전혀 안 쓰면 "GPM+버퍼"가 아니라 "버퍼를 만들기만 하고
        # 버리는 GPM"이 되어 SPIDER를 재현하지 못한다(4개 논문 컴포넌트
        # 전수 재감사에서 발견 — `NoAntiForgetting`/`SSFAntiForgetting`은
        # 이미 replay_batch를 쓰고 있었는데 GPM만 빠져 있었다). replay_batch가
        # 있을 때만(즉 mm=spider 등 실제로 버퍼를 채워주는 memory_manager와
        # 결합됐을 때만) 그 위에도 같은 task loss를 계산해 더한다 — mm=none과
        # 결합되면 replay_batch가 항상 None이라 원 논문 그대로(리플레이 없음)
        # 동작이 자동으로 보존된다. `mm=ssf`(SSFMemoryManager)와 결합돼도
        # 같은 방식으로 동작한다 — 그쪽은 진짜 라벨을 담고 있어(아래
        # "mm=spider 자기학습 루프" 절과 달리) 라벨 출처 걱정이 없다.
        #
        # **재감사에서 발견 — `mm=spider`와 결합되면 이 replay_batch가
        # 진짜 라벨이 아니라 SPIDERMemoryManager의 스냅샷 모델이 만든
        # pseudo-label이다**(`spider_memory_manager.py`의 `_pseudo_label()`
        # 참고) — 즉 `af=gpm+mm=spider`도 모델이 최근에 낸 예측을 스스로
        # 다시 학습하는 자기학습(self-training) 피드백 루프에 해당한다.
        # `af=lwf_ssf`/`af=none`에 대해서만 분석됐던 것과 같은 종류의
        # 우려이지, 완전히 별개의 통제군이 아니다 — 다만 실측(전체 SPIDER
        # 콤보, f1≈0.84, bwt≈+0.04~0.10)에서 자기학습으로 인한 뚜렷한
        # 붕괴는 관찰되지 않았다.
        if replay_batch is not None and replay_batch[0] is not None:
            r_data, r_labels = replay_batch
            _, _, r_logit = model(r_data)
            loss = loss + F.binary_cross_entropy_with_logits(
                r_logit.squeeze(-1), r_labels.float())
        return loss

    def project_gradients(self, model: BaseCLModel) -> None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in self._basis:
                if module.weight.grad is None:
                    continue
                basis = self._basis[name].to(module.weight.device)
                grad = module.weight.grad
                has_bias = module.bias is not None and module.bias.grad is not None
                # 2026-08-26 발견·수정 — bias는 지금까지 사영 대상에서 완전히
                # 빠져 있었다(아래 _update_basis "2026-08-26" 절 참고). 실측
                # (NSL-KDD, af=gpm, 5라운드 전체)으로 f1 0.35(af=none, 아무
                # 망각방지도 안 함)→0.0053(af=gpm)로 GPM이 있는데 오히려
                # 훨씬 나빠지는 것을 확인했다 — bias(특히 classifier의 판정
                # 임계값을 직접 결정하는 스칼라)가 GPM이 막는 weight 방향
                # 대신 그래디언트가 빠져나가는 우회로 역할을 한 것으로
                # 보인다. bias를 "항상 1인 입력 차원"으로 취급해 weight와
                # 함께 증강된 하나의 행렬로 다뤄 동일하게 사영한다(공식 GPM
                # 저장소는 bias=False 아키텍처라 이 문제 자체가 없었다 —
                # `main_pmnist.py` 확인 완료, 원문에 없는 이 테스트베드
                # 고유의 보정).
                if has_bias:
                    grad_aug = torch.cat([grad, module.bias.grad.unsqueeze(1)], dim=1)
                else:
                    grad_aug = grad
                proj = grad_aug @ basis @ basis.T
                grad_aug = grad_aug - proj
                if has_bias:
                    module.weight.grad = grad_aug[:, :-1]
                    module.bias.grad = grad_aug[:, -1]
                else:
                    module.weight.grad = grad_aug

    def on_task_end(self, model: BaseCLModel) -> None:
        if not self._pending_data:
            return
        all_data = torch.cat(self._pending_data, dim=0)
        self._update_basis(model, all_data)
        self._pending_data = []
        self._pending_count = 0

    def _update_basis(self, model: BaseCLModel, data: torch.Tensor) -> None:
        activations: Dict[str, torch.Tensor] = {}
        handles = []

        def make_hook(name):
            def hook(module, inp, out):
                activations[name] = inp[0].detach()
            return hook

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(make_hook(name)))

        # 2026-08-26 발견 — forward가 예외를 던지면(입력 shape 오류 등)
        # 훅이 안 지워진 채 남아 이후 라운드의 정상적인 forward 호출마다
        # 계속 같이 발동해 activations 딕셔너리를 조용히 오염시킬 수
        # 있었다(4개 논문 컴포넌트 전수 재감사에서 발견, 지금까지 실제로
        # 발동한 적은 없음). try/finally로 무슨 일이 있어도 훅은 반드시
        # 해제되게 한다.
        try:
            model.eval()
            with torch.no_grad():
                model(data)
        finally:
            for h in handles:
                h.remove()

        # 2026-08-26 발견·수정 — bias를 project_gradients()에서 weight와
        # 함께 사영하려면(위 project_gradients "2026-08-26" 절 참고),
        # 기저 자체가 "항상 1인 입력 차원"까지 포함한 증강 activation으로
        # 계산되어야 project 시 basis 차원이 grad_aug(weight+bias 결합)
        # 차원과 맞는다 — bias는 그 "항상 1" 방향의 gradient 성분이라는
        # 수학적으로 정확한 대응이다.
        modules_by_name = dict(model.named_modules())
        for name, act in activations.items():
            module = modules_by_name[name]
            if module.bias is not None:
                ones = torch.ones(act.shape[0], 1, device=act.device, dtype=act.dtype)
                act = torch.cat([act, ones], dim=1)
            new_basis = self._compute_basis(act)
            if name in self._basis:
                combined = torch.cat([self._basis[name], new_basis], dim=1)
            else:
                combined = new_basis
            Q, _ = torch.linalg.qr(combined, mode="reduced")
            # max_basis_ratio 상한 적용 — Q의 앞쪽 열은 기존(더 오래된) 기저를
            # 직교화한 것이라, 자르더라도 최근 experience의 신규 방향부터
            # 밀려나고 과거 보호는 우선 유지된다(GPM의 "오래된 태스크를 더
            # 우선 보호"라는 취지와 부합).
            ambient_dim = Q.shape[0]
            cap = max(1, int(ambient_dim * self.max_basis_ratio))
            self._basis[name] = Q[:, :cap]

    def _compute_basis(self, activation: torch.Tensor) -> torch.Tensor:
        # 2026-07-30: 공식 코드(main_pmnist.py, 평균 미중심화 + `+1` 없는
        # r=sum(cumsum<threshold))에 정확히 맞췄다가, 실측으로 회귀를
        # 발견해 되돌렸다 — NSL-KDD에서 af=gpm+as=cade_mad 조합 중 일부
        # (ss=ssf/mm=ssf 또는 spider, dd=ssf 또는 cade)가 exp1에서 예측이
        # 완전히 한 클래스로 퇴화했다(30개 gpm 조합 전수 검사로 확인,
        # unsw-nb15는 0/30, nsl-kdd는 2/30 실패). 원인 분리 실험 결과 두
        # 변경(중심화 제거, +1 제거) 중 **어느 한쪽만 되돌려도** 통과했다 —
        # 즉 공식 코드가 이 테스트베드의 (얕은 은닉층 1개 + 작은 분류기
        # head) 아키텍처에서는 기저가 지나치게 작아지거나(또는 ReLU 출력의
        # 항상-양수 편향 방향을 그대로 기저로 채택해) 분류기 학습에 필요한
        # 그래디언트 방향까지 사영으로 제거해버리는 경계 사례를 만든다.
        # 공식 코드를 문자 그대로 재현하는 것보다 "조합이 실제로 붕괴하지
        # 않고 작동하는 것"이 이 테스트베드의 목적(0절 — 재현이 아니라
        # 재조합·비교)에 더 부합하므로, 평균 중심화 + `+1` 여유 둘 다
        # 원래대로 유지한다(공식 코드와 다르다는 점만 정직하게 기록).
        #
        # 2026-08-14 residual projection을 두 차례 시도했다가 둘 다
        # 되돌림 — 자세한 경위는 이 클래스 docstring의 "2026-08-14" 절
        # 참고. 1차는 잔차 자체의 에너지로 정규화(자체 결함 가능성),
        # 2차는 WebFetch로 확인한 공식 Eq-9 그대로(잔차를 빼기 전 원본
        # activation 총 에너지로 정규화 + 누적치가 "기존 기저가 이미
        # 설명한 비율"에서 시작) — 둘 다 residual 없는 버전보다 A/B
        # 실측으로 더 나빴다(1차: f1 0.7006→0.6440, 2차: f1
        # 0.7006→0.5538, bwt도 둘 다 악화). 즉 "구현이 미묘하게 달랐다"가
        # 아니라 residual projection 자체가(정확히 구현해도) 이
        # 아키텍처(얕은 공유 backbone, latent_dim=32, 5 experience)에
        # 안 맞는다는 결론이 더 신뢰할 만하다 — 원 논문은 훨씬 큰 은닉층과
        # 훨씬 많은 태스크 수를 전제하는데, 그 전제에서 residual
        # projection이 주는 이점(레이어별 점진적 수렴)이 이 테스트베드
        # 규모에서는 오히려 "매 라운드 새로 채택하는 방향 수를 원래보다
        # 더 줄여 forgetting 방지 압력 자체를 약화시키는" 역효과로
        # 나타난 것으로 보인다(2차 시도의 실측 basis 크기는 모든 레이어에서
        # residual 없는 버전보다 작게 수렴했는데도 bwt는 더 나빴다 —
        # "더 작은 기저 = 더 약한 보호 = 더 심한 망각"과 부합).
        # residual 없이 max_basis_ratio 상한만 두는 현재 방식을 최종
        # 채택한다.
        centered = activation - activation.mean(dim=0, keepdim=True)
        _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        energy = S ** 2
        cumulative = torch.cumsum(energy, dim=0) / energy.sum().clamp(min=1e-10)
        k = int((cumulative < self.activation_threshold).sum().item()) + 1
        k = max(1, min(k, Vh.shape[0]))
        return Vh[:k].T
