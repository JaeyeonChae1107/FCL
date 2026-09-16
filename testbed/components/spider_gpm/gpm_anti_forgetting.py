"""GPM (Gradient Projection Memory) anti-forgetting.

SPIDER 저장소 코드를 확보하지 못해(testbed/docs/metric_justification.md 참고)
GPM 논문(Saha, Garg, Roy, ICLR 2021) 설명 기반으로 작성했었으나, GPM 자신의
공식 코드는 공개 GitHub(https://github.com/sahagobinda/GPM)에 있어
2026-09-14에 `main_pmnist.py`(366줄) 전체를 직접 받아 실제 코드 대 코드로
대조했다(로컬 클론은 없지만 웹에서 확보 — "코드 대조 불가능"은 아니었다).
아래 1·4번은 공식 코드와 일치, 2·3번은 의도적으로 다르다. 이번 대조로
새로 찾은 차이 2가지는 "GPM 기저 계산 — 공식 코드 재대조" 절 참고.

핵심 알고리즘:
  1. 태스크(experience) 종료 시(on_task_end), 그 태스크에서 실제로 학습에
     쓰인 selected_data로 각 Linear 레이어의 입력 activation 행렬을 수집한다.
  2. 레이어별 activation 행렬에 SVD를 적용해 누적 에너지 비율이
     activation_threshold 이상이 되는 최소 개수(+1 여유)의 우특이벡터를
     이번 태스크의 기저로 채택한다. 공식 코드는 평균을 빼지 않은 원본
     activation에 바로 SVD를 적용하고 `+1` 여유도 없다 — 문자 그대로
     맞춰봤더니 NSL-KDD `af=gpm+as=cade_mad` 일부 조합(ss=ssf 계열)이
     exp1에서 분류기가 완전히 한 클래스로 퇴화하는 회귀가 실측으로
     확인됐다(30개 gpm 조합 전수검사: nsl-kdd 2/30 실패, unsw-nb15 0/30).
     평균 중심화와 `+1` 여유 둘 다 원래대로 유지한다.
  3. 기존에 저장된 기저와 새 기저를 concat 후 QR 분해로 재직교화해 누적
     기저(GPM memory)를 갱신한다. 공식 코드는 `np.hstack`으로 이어붙이기만
     하고 QR 재직교화를 하지 않는다 — 이 테스트베드는 5개 experience에
     걸쳐 기저가 누적되는데, 서로 다른 태스크의 SVD 결과를 단순히
     이어붙이면 `basis @ basis.T`가 참된 직교 사영행렬이 아니게 될 수
     있어 QR 재직교화를 유지한다.
  4. 이후 태스크 학습 시 backward() 직후 project_gradients()가 각 레이어
     가중치의 gradient에서 누적 기저 방향 성분을 제거한다:
       grad_proj = grad - grad @ basis @ basis.T
     (공식 코드의 `Uf = feature_list[i] @ feature_list[i].T` 사영과 일치.)

이 테스트베드의 목적(0절 — 논문 재현이 아니라 재조합·비교)상, "공식 코드와
문자 그대로 같다"보다 "조합이 실제로 붕괴하지 않고 작동한다"를 우선했다.
공식 코드와 다른 지점(2·3번)은 실측 근거와 함께 기록한다.

레지스트리 키는 "gpm"(폴더는 components/spider_gpm/). GPM은 명시적
replay/정규화 항 없이 gradient projection만으로 이전 태스크를 보호하므로
(원 논문 설계), compute_loss는 task loss만 계산한다.

**기저 풀랭크 문제**: NSL-KDD 실측(af=gpm, 5 experience)에서 `encoder.0`
(121차원)이 exp4에서 121/121(풀랭크)에 도달해 그 레이어의 그래디언트가
마지막 라운드에 0이 됐다. GPM Algorithm 2의 residual projection(누적 기저
방향을 먼저 제거하고 남은 residual에만 SVD 적용)을 두 방식으로 추가해
시도했으나(1차: 잔차 에너지 정규화, 2차: 공식 Eq-9 그대로) 둘 다 A/B
실측으로 악화됐다(1차: f1 0.7006→0.6440, bwt -0.108→-0.142; 2차: f1
0.7006→0.5538, bwt -0.108→-0.170). residual projection 자체가 이
아키텍처(얕은 공유 backbone, latent_dim=32, 5 experience)에 안 맞는다는
결론이다 — 상세는 `_compute_basis()` 참고. 대신 `max_basis_ratio` 상한만
추가해 어떤 레이어도 ambient dimension의 (1-max_basis_ratio) 미만으로는
학습 가능 공간이 줄어들지 않게 했다.

**bias가 projection 대상에서 빠져 있던 결함**: `project_gradients()`가
`module.weight.grad`만 사영하고 `module.bias.grad`는 건드리지 않았다.
공식 GPM은 `nn.Linear(..., bias=False)`만 쓰는 아키텍처라 이 문제가 없었지만
(`main_pmnist.py:29-31`), 이 테스트베드의 공유 `FCLAutoEncoder`는
`bias=True`를 쓴다 — GPM이 보호하려는 weight 방향의 그래디언트가 보호받지
않는 bias(classifier의 판정 임계값을 결정하는 스칼라)로 우회할 수 있었다.

실측(NSL-KDD, 200 epoch/experience, 5라운드 전체): `dd=none/ss=random/
mm=spider/af=gpm/as=none`이 f1=0.0053, recall=0.26%, pr_auc=0.5191,
bwt=-0.6280까지 붕괴 — 같은 조합에서 `af=none`(망각방지 없음)은 f1=0.3541,
pr_auc=0.8205, bwt=-0.3085로 오히려 나았다. 라운드별 pooled 양성 예측 수가
7753→9257→1959→510→37로 단조 감소했다(encoder.0 기저가 46→108/121까지
성장하며 classifier.bias가 계속 흔들려 거의 모든 표본을 "정상"으로 판정).
bias를 "항상 1인 입력 차원"으로 취급해 weight와 증강한 행렬에 동일한
기저로 사영하도록 수정했으나, A/B 실측 결과 이 수정만으로는 붕괴가 전혀
해결되지 않았다(f1 0.0053→0.0053, roc_auc는 오히려 0.265로 악화) — bias
미보호는 실재하는 결함이지만 이 붕괴의 원인은 아니었다.

**진짜 원인 — GPM 결함이 아니라 `as=none`(고정 0.5 임계값)과의 궁합**:
`af=gpm`의 gradient projection 압력 아래서 classifier 로짓 스케일이
라운드를 거듭하며 계속 커진다(`clf.weight_norm`이 exp0→exp4에 걸쳐
3.08→6.16). `as=none`은 SSF/SPIDER 원 논문 그대로 고정 0.5 sigmoid
임계값을 쓰는데(재보정 없음), 커지는 로짓 스케일을 고정 임계값이 못
따라가며 거의 모든 표본이 "정상"으로 판정됐다.

결정적 A/B(NSL-KDD, `dd=none/ss=random/mm=none`, 200 epoch·5라운드 전체):
  - `af=gpm + as=none`(고정 임계값): f1=0.0054, recall=0.27%, pr_auc=0.711,
    roc_auc=0.603, bwt=-0.507
  - `af=gpm + as=cade_mad`(매 라운드 재보정 임계값): f1=0.6655,
    recall=51.9%, pr_auc=0.884, roc_auc=0.852, bwt=-0.1405
  - 비교: `af=none`(망각방지 없음) + `as=none`: f1=0.2445, bwt=-0.4331

같은 데이터로 판정 방식만 바꿨을 뿐인데 f1이 0.0054→0.6655로 달라진다 —
`as=cade_mad`로 재보정하면 GPM은 naive fine-tuning보다 훨씬 덜 잊는다
(bwt -0.14 vs -0.43, GPM이 원래 주장하는 효과).

**결론**: `af=gpm`+`as=none`은 두 컴포넌트 각각 자기 원 논문대로 충실히
구현됐지만 같이 쓰면 안 맞는 구조적 비호환 조합이다. GPM 코드로 이 조합을
억지로 고치지 않는다 — 대신 `smoke_test.py`의 15.2 게이트(다수 클래스
비율≥0.97 실패)와 15.2b 게이트(roc_auc 역전 감지)가 이 조합을 실패로 잡아
그리드에서 제외한다. `af=gpm`+`as=cade_mad`/`as=none` 둘 다 `TRACK_A_GRID`
에는 남겨둔다.

**CUDA OOM 버그**: `compute_loss()`가 매 미니배치 스텝마다 배치를
`_pending_data`에 누적했는데, `epochs_per_experience`(Track A 200)만큼
반복되는 각 epoch은 같은 `selected_data`를 다시 섞어 도는 것뿐이라 200배
중복 누적이었다. 라운드당 선택 샘플이 많은 경우 수천만 행까지 쌓여 CUDA
OOM이 났다. `activation_sample_size`
(기본 2000)로 상한을 둬서 experience당 최대 그만큼만 모은다 — GPM 논문의
취지(활성화의 "대표 표본"으로 SVD 기저 계산)와도 일치한다.

**클래스 균형 강제를 시도했다가 되돌림**: SPIDER 논문이 명시하는 "class당
n_s개, 클래스 균형" 요구사항을 `compute_loss()`가 지키지 않는다(`labels`를
받으면서도 쓰지 않고 순서대로만 `_pending_data`에 채운다) — 이걸 클래스당
목표치(`activation_sample_size // 2`)로 균형을 강제하도록 고쳐 NSL-KDD
3개 조합(mm∈{none,spider,ssf}, af=gpm/as=cade_mad)으로 A/B 실측했다.
`mm=spider` 조합만 bwt가 개선됐고(-0.011→+0.123, f1은 거의 그대로
0.793→0.795) 나머지 둘은 뚜렷이 악화됐다 — `mm=none`: f1 0.739→0.624,
bwt -0.043→-0.094; `dd=ssf/ss=ssf/mm=ssf`: f1 0.815→0.575, bwt
+0.070→-0.318(3개 중 가장 큰 악화). 원인 추정: 이 데이터셔은
class-incremental이라 매 라운드 새로운 공격 카테고리가 등장하는데, 공격
activation의 비중을 인위적으로 50%까지 끌어올리면 SVD 기저가 "지금까지
등장한 공격 방향"을 더 넓게 덮게 되고, 그 방향이 project_gradients()로
보호되면서 다음 라운드의 새로운(아직 안 본) 공격 카테고리를 학습하는 데
필요한 그래디언트까지 함께 억눌린다 — 순서대로 누적하는 원래 방식(자연
발생 비율, 대개 정상이 다수)은 공격 방향을 상대적으로 덜 보호해 새 공격
카테고리 학습 여지를 더 남긴다는 가설이다. 다른 문제(2·3번, "기저 풀랭크",
"bias 미보호")와 같은 "원문에 더 충실하지만 이 테스트베드의
class-incremental 구조와 안 맞는" 패턴이라 순서대로 누적하는 원래 방식을
유지한다.

**Item 8b — SPIDER self-training(Algorithm 1의 (c) 요소)**: `mm=spider`
조합에서 `CLClient`가 Step 3에 이번 라운드 라벨 예산 밖 잔여 데이터를
`set_self_training_pool()`로 넘겨주면, Step 4의 매 미니배치마다
`get_self_training_batch()`가 거기서 무작위 표본을 뽑아 **현재 학습
중인 모델**로 pseudo-label해 `compute_loss()`에서 다른 소스와 합쳐진다.
`set_self_training_pool()`이 애초에 `mm=spider`일 때만 호출되므로 "풀이
채워져 있는가"만으로 이미 게이팅이 되어 별도 플래그는 불필요하다 —
`mm=none`/`mm=ssf`와 결합되면 풀이 항상 `None`이라 `get_self_training_
batch()`가 매번 `None`을 반환해 기존 동작이 그대로 보존된다.
`last_self_training_pseudo_ratio`(다수 클래스 비율)로 자기학습 피드백
루프 붕괴 여부를 진단한다.

**2026-09-12 재설계 — SPIDER 원문 Algorithm 1과 재대조,
NSL-KDD smoke A/B로 3가지를 각각 검증**: 사용자가 SPIDER 논문 전문을
제공해 Algorithm 1을 정확히 확인한 결과, 1차 구현(2026-09-11)은 원문과
두 지점에서 달랐다 — (a) `B_u`를 현재 모델로 pseudo-label(원문은 직전
태스크 스냅샷 f_θ^{t-1} 요구), (b) `B_l ∪ B̂_u ∪ B̂_m^u`를 별도 loss 항
합산(원문은 하나의 배치로 합쳐 그래디언트 1회). GPM 기저의 클래스 균형
(Algorithm 1 line 10)은 예전에 self-training이 아직 (a)(b) 방식일 때
단독으로 시도해 mm=spider만 개선되고(-0.011→+0.123) mm=none/ssf는
악화됐던 전력이 있어(모듈 docstring "클래스 균형 강제를 시도했다가
되돌림" 절), 이번에 self-training도 함께 고친 뒤 재검증했다.

기준선(`dd=none/ss=random/mm=spider/af=gpm/as=cade_mad`, smoke epoch 10):
자기학습 없음(mm=none) f1=0.7085/bwt=-0.0723. 1차 구현(현재모델+별도
loss항, 클래스균형 없음) f1=0.8689/bwt=-0.0073 — 이 smoke 스케일에서는
기존 A/B 기록(f1=0.818/bwt=+0.0312, 스케일 불명)과 다른 수치지만 방향은
같다(자기학습 없음보다 개선).

(a)(b)(클래스균형)을 전부 원문대로 바꾼 버전: f1=0.7480/bwt=+0.0082 —
1차 구현보다 f1이 뚜렷이 나빠짐(라운드별 자체 성능 diag
[0.906,0.907,0.263,0.365] — round2/3 붕괴). 원인 진단: 이 테스트베드는
"라운드 하나 = 새 공격 유형 하나"인데, 스냅샷(직전 라운드 모델)은 이번
라운드에 처음 등장하는 공격을 한 번도 본 적이 없어, 비선택 잔여(대부분
이번 라운드의 새 카테고리)를 pseudo-label하면 거의 다 "정상"으로 잘못
판정한다 — 이번 라운드가 배워야 할 신호를 자기 자신이 억누르는 꼴이다.
SPIDER 원 논문 벤치마크는 태스크당 여러 클래스가 섞여 있어 이 문제가
덜 드러났을 가능성이 높다.

(a)만 되돌리고(현재모델) (b)(클래스균형)은 유지한 버전:
f1=0.8462/bwt=+0.0285 — diag [0.906,0.875,0.748,0.814]로 건강. 1차
구현보다 bwt가 더 좋고(−0.0073→+0.0285) f1은 근소하게만 낮다
(0.8689→0.8462, -2.3%p). **최종 채택**: (a) 스냅샷은 폐기하고 현재모델
유지(`set_snapshot_model`/`_snapshot_model` 전부 제거, `_has_prior_round`
로 대체 — round 0에는 self-training을 안 한다는 성질만 유지, Algorithm 1
순수성 근거는 폐기), (b) concat 방식은 채택하되 **mm=spider에만** 적용
(처음에 replay_batch가 있기만 하면 항상 concat했더니 `dd=ssf/ss=ssf/
mm=ssf`가 f1 0.7058→0.6855, bwt -0.0737→-0.1352로 뜻하지 않게 나빠져 —
mm=ssf의 replay_batch는 SPIDER의 버퍼가 아니라 SSFMemoryManager의
것이라 Algorithm 1과 무관하므로 예전 방식으로 되돌림), (c) 클래스균형은
채택(같은 게이팅).

(b)(concat) 없이 (a)(현재모델)+클래스균형만: f1=0.7679/bwt=+0.0052 —
최종 채택안보다 둘 다 나쁨. 클래스균형이 self-training을 제대로 고친
뒤에는(1차 구현 때와 달리) 명확히 기여한다는 뜻.

**한계**: 단일 시드·NSL-KDD 단일 데이터셋·smoke 스케일(epoch 10) 스윕
— 6.8절 멀티시드 실행 시 재검증 대상. `no_selftrain_ref(mm=none)`와
`ssf_combo(mm=ssf)`는 최종 코드에서 정확히 1차 구현 이전과 동일한
숫자로 확인됐다(회귀 없음 확인 완료).

**GPM 기저 계산 — 공식 코드 재대조(2026-09-14, `main_pmnist.py` 366줄
전체 확보 후 재확인)**: 기존에 알던 차이(평균중심화 없음/`+1` 없음/
QR 재직교화 없음, 전부 실측으로 되돌린 상태)에 더해 2가지를 새로 찾았다:

1. **대표 표본 수**: 공식 코드는 태스크당 정확히 `b=r[0:300]` — **300개
   고정**(`get_representation_matrix`, `main_pmnist.py:119-145`). 이
   테스트베드 기본값(`activation_sample_size=2000`)은 CUDA OOM 방지
   목적으로 정한 것이라(모듈 docstring 위쪽 참고) 애초에 논문 근거가
   아니었다. NSL-KDD smoke A/B(`dd=none/ss=random/mm=spider/af=gpm/
   as=cade_mad`)로 300 직접 테스트: f1 0.8462→0.7162, bwt
   +0.0285→+0.0046, pr_auc 0.8762→0.8522 — **뚜렷이 나빠져 2000 유지**.
   원인 추정: 공식 코드는 Permuted MNIST(784차원 픽셀, 10-클래스 단순
   분류)용이라 300개로도 대표성이 충분하지만, 이 테스트베드는 훨씬
   고차원·복잡한 NIDS 트래픽 특징을 다루므로 같은 표본 수로는 SVD 기저의
   대표성이 부족한 것으로 보인다.
2. **레이어별 threshold**: 공식 코드는 `threshold = [0.95, 0.99, 0.99]`로
   레이어마다 다른 값(깊을수록 엄격)을 쓰는데(`main_pmnist.py:216`), 이
   테스트베드는 모든 레이어에 `activation_threshold=0.97` 하나를 적용한다.
   이건 테스트하지 않고 현행 유지로 결정했다 — 공식 값은 GPM 논문 고유의
   3-레이어 MLP(인코더 2개+분류기 1개, 디코더 없음)에 맞춘 것이라 레이어
   수·역할 자체가 다른 이 테스트베드의 공유 오토인코더+분류기 구조(레이어
   수가 데이터셋마다 다르고 디코더까지 포함)에 그대로 대응시킬 원칙적
   근거가 없다 — "몇 번째 레이어에 0.95를, 몇 번째에 0.99를 줄지"를
   결정할 원문 기준 자체가 없어 억지로 매핑하면 오히려 근거 없는 임의
   선택이 된다. 균일 0.97을 문서화된 근사로 유지한다.

**`mm=spider` 결합배치가 Algorithm 1의 `r`(라벨 비율) 비례 크기를 구현하지
않는다(2026-09-14, Round-4 교차-슬롯 감사로 발견)**: `compute_loss()`의
concat 분기(`B_l ∪ B̂_u ∪ B̂_m^u`)는 세 소스(`data`=new_batch, `replay_batch`
=SPIDER 버퍼, `self_training_batch`=GPM 자기학습)를 **각각 독립적으로
최대 `batch_size`까지** 채운 뒤 단순히 이어붙인다 — SPIDER 원 논문
Algorithm 1이 실제로 명시하는 `b_rem = b - b_m; b_l = b_rem × r; b_u =
b_rem - b_l`(진짜 라벨 대 pseudo-label의 **비례** 크기를 `r`로 통제)은
구현돼 있지 않다. 세 소스가 전부 가득 차면 결과적으로 대략 1:1:1로
섞이는데, 이는 `r`이 어떤 값이든 하나로 고정된 것과 같다 — "b_l/b_rem"
비율을 다른 값으로 바꿔보는 실험 자체가 지금 구조로는 불가능하다.

**같은 배치 안에서 서로 다른 시점의 모델이 만든 pseudo-label이 동등한
가중치로 섞인다**: `replay_batch`(SPIDER)의 라벨은 **직전 라운드 종료
시점 스냅샷**(`_snapshot_model`)이, `self_training_batch`(GPM)의 라벨은
**지금 학습 중인 현재 모델**(위 "재설계" 절에서 스냅샷을 폐기하고 현재
모델을 채택하기로 실측 확정)이 만든다 — 하나의 BCE 손실 안에 서로 다른
두 시점의 모델이 낸 예측이 실측 라벨과 구분 없이 동등하게 섞이는 것이다.
이 구조 자체(concat 방식 채택, 자기학습에 현재 모델 채택, 클래스 균형
채택)는 각각 개별적으로 A/B 실측을 거쳤지만(위 "Item 8b" 절), **이
세 가지가 합쳐진 결과로 생기는 "두 pseudo-label 소스의 시점 불일치를
동등 가중치로 섞는다"는 조합 효과 자체는 별도로 분석된 적이 없다** —
`ssf_memory_manager.py`의 "자기학습 피드백 루프" 절과 같은 성격의,
정직하게 남겨두는 한계다. 크래시나 눈에 띄는 붕괴는 없다(위 최종 채택안
f1=0.8462/bwt=+0.0285가 이미 이 구조를 거쳐 나온 건강한 수치) — 다만
`r` 비례 구현과 스냅샷 통일(둘 다 재현하려면 GPM 자기학습도 스냅샷으로
되돌려야 하는데, 위 (a) 실측에서 스냅샷 자체가 뚜렷한 회귀임을 이미
확인했다) 중 어느 쪽을 택해도 지금 구조와는 다른 새로운 A/B가 필요해
이번엔 손대지 않는다. `design_decisions.md` 7절 참고.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from testbed.base.anti_forgetting import BaseAntiForgetting
from testbed.base.models import BaseCLModel
from testbed.common.rng_utils import derived_seed


class GPMAntiForgetting(BaseAntiForgetting):
    """**전역 RNG 오염(2026-09-14 재검토로 발견, 수정)**: `get_self_training_
    batch()`의 `torch.randperm`이 전역 RNG를 소비하는데, 이 메서드는
    `mm=spider`일 때만 호출된다(Step 4, 미니배치마다) — 즉 `mm=spider`
    on/off 하나로 그 뒤 메인 모델 학습이 보는 난수 시퀀스 전체가 갈라진다.
    `CADEDriftDetector`/`ssf_masks.py`/`SPIDERMemoryManager`에도 같은
    문제가 있어(각 파일 참고) 국지적 버그가 아니라 설계 패턴 자체의
    문제였다 — "슬롯 값만 바꿔 통제된 비교를 한다"는 이 벤치마크의 핵심
    전제를 위협해 우선 수정했다. `torch.random.fork_rng()`로 격리한다.
    파생 시드는 `rng_utils.derived_seed(seed, "gpm_self_training",
    self._self_training_call_count)`(해시 기반) — 이 카운터는 라운드가
    아니라 미니배치마다 증가해 Track A 실전 설정(epoch=200)에서 수천까지
    쉽게 도달하므로, 애초에 정수 오프셋(예: seed+90300+count) 방식을 썼을
    때 다른 컴포넌트의 오프셋 구간(100 간격)을 실제로 침범하는 문제가
    있었다 — 2026-09-14 재검토로 발견해 해시 기반으로 전면 교체했다
    (`rng_utils.py` docstring 참고)."""

    backbone_type = "classifier"

    # 2026-09-14 재검토로 발견·수정 — GPM 원 논문의 `get_representation_
    # matrix(net, device, x, y=None)`(공식 코드, WebFetch로 확인)는 라벨(y)
    # 조차 안 쓴다 — "라벨 예산"이라는 개념 자체가 없다. CADE(`consumes_
    # full_round_data`, 2026-09-12)와 SPIDERMemoryManager(Item 8a)는 정확히
    # 같은 근거("원 논문에 라벨 예산 개념 없음")로 면제를 받았는데 GPM만
    # 빠져 있었다 — `compute_loss()`가 라벨예산 서브셋(`selected_data`)의
    # 미니배치에서만 activation을 누적해, SVD 기저가 실제 논문보다 훨씬
    # 좁은 풀에서 계산되고 있었다. `set_full_round_data()`가 이 자리를
    # 대체한다 — `cl_client.py` Step 3이 이 플래그를 보고 라운드당 1회
    # `new_data`(라벨예산 밖 포함 전체)를 넘긴다. 분류기 학습 자체(task
    # loss)는 여전히 라벨예산 서브셋만 쓴다 — 바뀌는 건 "GPM이 기저 계산에
    # 쓰는 표본 풀"뿐이다. CADE/SPIDER와 같은 "원 논문에 없는 라벨예산
    # 제약을 없앤다"는 구조적 일관성 수정이라 A/B 방향과 무관하게 채택한다
    # (SSF 표본선택 수정과 같은 원칙) — sanity 확인(NSL-KDD, epoch=5):
    # `af=gpm/mm=none/as=cade_mad` f1=0.7465/bwt=-0.0611,
    # `af=gpm/mm=spider/as=cade_mad` f1=0.7986/bwt=-0.0054, 둘 다 붕괴
    # 없이 건강하게 작동함을 확인(정식 epoch=200 재검증은 그리드 실행 시).
    consumes_full_round_data = True

    def __init__(self, activation_threshold: float = 0.97,
                 activation_sample_size: int = 2000,
                 max_basis_ratio: float = 0.9, seed: int = 42):
        self.activation_threshold = activation_threshold
        # 전역 RNG 오염 격리(2026-09-14, 모듈 docstring "전역 RNG 오염" 참고).
        self._seed = seed
        self._self_training_call_count = 0
        # 기저가 ambient dimension을 다 채우면(풀랭크) project_gradients()의
        # `grad - grad@basis@basis.T`가 그 레이어 그래디언트를 완전히 0으로
        # 만들어 학습이 조용히 멈춘다 — 이 테스트베드의 작은 latent_dim(NSL-KDD
        # 기준 32)에서는 5 experience 안에 이 경계에 실제로 도달한다. 항상
        # ambient dimension의 (1-max_basis_ratio) 비율만큼은 학습 가능하게
        # 남겨둔다(원 논문에 없는 이 테스트베드 고유의 안전장치).
        self.max_basis_ratio = max_basis_ratio
        # on_task_end에서 activation SVD 기저를 계산할 "대표 표본" 상한.
        # experience당 최대 activation_sample_size개까지만 모은다(모듈
        # docstring "CUDA OOM 버그" 절 참고). 클래스 균형 강제는 시도했다가
        # 실측 회귀로 되돌렸다(모듈 docstring "클래스 균형 강제를 시도했다가
        # 되돌림" 절 참고) — 등장 순서 그대로(자연 발생 비율) 누적한다.
        self.activation_sample_size = activation_sample_size
        self._basis: Dict[str, torch.Tensor] = {}
        self._pending_data: List[torch.Tensor] = []
        self._pending_count = 0
        # 2026-09-12 추가 — mm=spider일 때만 쓰는 클래스별 누적 카운트
        # (compute_loss 참고). mm=none/ssf에서는 항상 빈 채로 남아
        # _pending_count(기존 순서대로 누적)만 쓰인다.
        self._pending_count_by_class: Dict[int, int] = {}
        # Item 8b(SPIDER self-training) — mm=spider일 때만 CLClient가
        # 채워준다(set_self_training_pool). af=gpm이 mm=none/ssf와
        # 결합되면 이 풀은 항상 비어 있어 get_self_training_batch()가
        # 매번 None을 반환한다 — 다른 조합의 동작은 바뀌지 않는다.
        self._self_training_pool: Optional[torch.Tensor] = None
        # PRD 15.4절과 같은 성격의 진단 필드 — 자기학습 배치의 pseudo-label
        # 다수 클래스 비율(마지막 미니배치 기준). 자기학습 피드백 루프가
        # 붕괴하면(모델이 한쪽 클래스로만 확신) 이 값이 1.0에 가까워진다.
        self.last_self_training_pseudo_ratio: Optional[float] = None
        # 2026-09-12 추가 — 최소 한 라운드는 끝나야
        # self-training을 시작한다(아래 get_self_training_batch 참고).
        # on_task_end()에서 True로 바뀐다.
        self._has_prior_round = False

    def set_full_round_data(self, new_data: torch.Tensor, new_labels: torch.Tensor) -> None:
        """`cl_client.py` Step 3가 라운드당 1회 호출(`consumes_full_round_data
        =True`) — 이전엔 `compute_loss()`의 미니배치(라벨예산 서브셋 유래,
        epoch마다 재셔플)에서만 채우던 `_pending_data`를 이제 라운드 전체
        (`new_data`, `dataset_loader.py`가 로드 시점에 1회 섞어둔 순서)에서
        직접 채운다. 분기 조건(`mm=spider`면 클래스 균형, 아니면 순서대로)
        은 `compute_loss()`가 하던 것과 같은 신호(`_self_training_pool`)로
        게이팅되지만, **표본의 출처·크기·셔플 방식 자체는 근본적으로
        바뀌었다**(라벨예산 서브셋의 미니배치 → 라운드 전체) — "정책을
        그대로 옮겼을 뿐"이라 말하는 건 부정확하다(2026-09-14 재검토로
        정정).

        **클래스 균형(`mm=spider`)의 새 데이터 출처 기준 재A/B(2026-09-14)**:
        NSL-KDD smoke(epoch=5, `af=gpm/mm=spider/as=cade_mad`) 균형 ON
        f1=0.8003/bwt=+0.0028/pr_auc=0.8988 vs OFF(순서대로 누적)
        f1=0.7950/bwt=+0.0045/pr_auc=0.8836 — 차이가 작고 방향도 지표마다
        갈린다(예전에 다른 데이터 출처로 봤던 것 같은 뚜렷한 개선/악화가
        아님). Algorithm 1 line 10이 클래스 균형을 명시하므로 구조적
        근거가 있는 쪽(균형 ON)을 유지하되, 이 판단이 최종적인 건
        아니다 — 정식 epoch(200)·멀티시드 단계에서 반드시 재확인이
        필요하다(`design_decisions.md` 7절에 추적 항목으로 남긴다)."""
        if self._self_training_pool is not None:
            target_per_class = self.activation_sample_size // 2
            for c in (0, 1):
                cur = self._pending_count_by_class.get(c, 0)
                if cur >= target_per_class:
                    continue
                mask = new_labels == c
                if not bool(mask.any()):
                    continue
                cand = new_data[mask].detach()
                take = min(len(cand), target_per_class - cur)
                self._pending_data.append(cand[:take])
                self._pending_count_by_class[c] = cur + take
        elif self._pending_count < self.activation_sample_size:
            take = min(len(new_data), self.activation_sample_size - self._pending_count)
            self._pending_data.append(new_data[:take].detach())
            self._pending_count += take

    def set_self_training_pool(self, unselected_data: torch.Tensor) -> None:
        """CLClient가 Step 3에서 라운드당 1회 호출(mm=spider 조합만) —
        이번 라운드 라벨 예산 밖에 남은 데이터(원 논문 Algorithm 1의
        (c) 요소가 소비하는 비라벨 잔여 풀)."""
        self._self_training_pool = unselected_data

    def get_self_training_batch(self, model: BaseCLModel, batch_size: int
                                 ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Step 4의 각 미니배치마다 호출 — 비라벨 잔여 풀에서 무작위로
        최대 batch_size개를 뽑아 **현재 학습 중인 모델**로 pseudo-label한다.
        풀이 없거나 비어 있거나, 아직 한 라운드도 끝나지 않았으면(round 0)
        None(호출부가 self_training_batch 인자 자체를 생략).

        **2026-09-12 재설계 — 스냅샷 대신 현재모델로 재확정**: Algorithm 1은
        직전 태스크의 스냅샷(f_θ^{t-1})으로 pseudo-label하라고 명시하지만,
        스냅샷 버전을 실제로 구현해 A/B했더니(NSL-KDD smoke,
        `dd=none/ss=random/mm=spider/af=gpm/as=cade_mad`) f1이 0.8689→0.7480로
        뚜렷이 나빠졌다(라운드별 자체 성능 diag: [0.906,0.907,0.263,0.365] —
        round2/3 붕괴). 원인: 이 테스트베드는 "라운드 하나 = 새 공격 유형
        하나"라 스냅샷(직전 라운드 모델)은 이번 라운드에 처음 등장하는
        공격 유형을 한 번도 본 적이 없다 — 그런 모델이 이번 라운드의
        비선택 잔여(대부분 이번 라운드의 새 카테고리)를 pseudo-label하면
        높은 확률로 전부 "정상"으로 잘못 판정해, 정작 이번 라운드가 배워야
        할 신호를 스스로 억누른다. 현재 모델로 되돌리자(스냅샷/set_snapshot_
        model 자체를 제거) f1=0.8462/bwt=+0.0285로 원래 기준선(f1=0.8689/
        bwt=-0.0073)보다 bwt가 개선되고 f1도 근접했다(diag:
        [0.906,0.875,0.748,0.814] — 훨씬 건강). SPIDER 논문의 벤치마크는
        태스크당 여러 클래스가 섞여 있어 이 문제가 덜 드러났을 가능성이
        높다 — "매 라운드 정확히 하나의 새 카테고리"라는 이 테스트베드
        고유의 스트림 구조와 원문의 스냅샷 요구가 안 맞는, 이 세션에서
        반복적으로 발견된 "원문에 더 충실하지만 이 구조와 안 맞는" 패턴이다.
        """
        if self._self_training_pool is None or len(self._self_training_pool) == 0:
            return None
        if not self._has_prior_round:
            return None
        pool = self._self_training_pool
        n = min(batch_size, len(pool))
        # 전역 RNG 오염 격리 — 이 메서드는 Step 4의 각 미니배치마다 호출되며
        # mm=spider일 때만 발동한다. 격리 없이 전역 RNG를 쓰면 mm=spider
        # on/off에 따라 그 뒤 이어지는 메인 모델 셔플·다음 라운드 전체의
        # 난수 시퀀스가 갈라진다(모듈 docstring "전역 RNG 오염" 참고).
        # 호출마다 다른 파생 시드(_self_training_call_count)를 써서 매번
        # 같은 표본만 뽑히지 않게 한다.
        with torch.random.fork_rng():
            torch.manual_seed(derived_seed(
                self._seed, "gpm_self_training", self._self_training_call_count))
            idx = torch.randperm(len(pool), device=pool.device)[:n]
        self._self_training_call_count += 1
        batch = pool[idx]
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                _, _, logit = model(batch)
        finally:
            if was_training:
                model.train()
        pseudo_labels = (torch.sigmoid(logit).squeeze(-1) > 0.5).long()
        ratio = pseudo_labels.float().mean().item()
        self.last_self_training_pseudo_ratio = max(ratio, 1.0 - ratio)
        return batch, pseudo_labels

    def compute_loss(self, model: BaseCLModel,
                      new_batch: Tuple[torch.Tensor, torch.Tensor],
                      replay_batch: Optional[Tuple[torch.Tensor, torch.Tensor]],
                      self_training_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                      ) -> torch.Tensor:
        data, labels = new_batch
        # SVD 기저(GPM 고유 메커니즘) 표본 누적은 2026-09-14부로 이 메서드가
        # 아니라 `set_full_round_data()`(라운드당 1회, `new_data` 전체)가
        # 담당한다 — 클래스 docstring "라벨예산 면제" 절 참고. 이전엔 여기서
        # `new_batch`(라벨예산 서브셋의 미니배치)마다 누적했었다.
        #
        # 2026-09-12 재설계 — Algorithm 1은
        # B_l(new_batch) ∪ B̂_u(self_training_batch) ∪ B̂_m^u(replay_batch)를
        # **하나의 배치**로 합쳐 그래디언트 1회를 계산한다. 이전엔 각각
        # 별도 BCE를 계산해 단순 합산했다(배치 크기 비율과 무관하게 각 항이
        # 동일 가중치).
        #
        # **범위(Q2): mm=spider일 때만** concat 방식으로 바꾼다 —
        # `_self_training_pool is not None`으로 게이팅(클래스 균형과 동일한
        # 신호). 처음에는 replay_batch가 있기만 하면(mm=ssf 포함) 항상
        # concat했는데, NSL-KDD smoke A/B로 `dd=ssf/ss=ssf/mm=ssf`가
        # f1 0.7058→0.6855, bwt -0.0737→-0.1352로 뜻하지 않게 나빠지는 걸
        # 확인했다 — Algorithm 1은 애초에 mm=spider(SPIDER 자신의 버퍼)를
        # 전제하는 절차라 mm=ssf의 replay_batch(SSFMemoryManager)에 적용할
        # 근거가 없었다. mm=ssf/mm=none은 예전 방식(별도 BCE 합산) 그대로
        # 보존한다.
        if self._self_training_pool is not None:
            all_data = [data]
            all_labels = [labels]
            if replay_batch is not None and replay_batch[0] is not None:
                all_data.append(replay_batch[0])
                all_labels.append(replay_batch[1])
            if self_training_batch is not None:
                all_data.append(self_training_batch[0])
                all_labels.append(self_training_batch[1])
            combined_data = torch.cat(all_data, dim=0)
            combined_labels = torch.cat(all_labels, dim=0)
            _, _, logit = model(combined_data)
            loss = F.binary_cross_entropy_with_logits(
                logit.squeeze(-1), combined_labels.float())
            return loss

        # mm=none/ssf — 예전 방식(별도 BCE 합산) 그대로. GPM 원 논문은
        # 리플레이가 필요 없다는 게 핵심 주장(gradient projection이 리플레이를
        # 대체)이라 mm=none(순정 GPM)은 replay_batch가 항상 None이라 이
        # 분기 자체가 no-op. mm=ssf는 SSFMemoryManager의 진짜 리플레이를
        # 받는데, 이건 Algorithm 1과 무관한 이 테스트베드의 별도 조합이므로
        # 기존 동작을 그대로 보존한다.
        _, _, logit = model(data)
        loss = F.binary_cross_entropy_with_logits(logit.squeeze(-1), labels.float())
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
                # bias를 "항상 1인 입력 차원"으로 취급해 weight와 함께
                # 증강된 행렬로 다뤄 동일하게 사영한다(공식 GPM은 bias=False
                # 아키텍처라 이 문제가 없었다 — 모듈 docstring 참고).
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
        # 2026-09-12 추가 — 이 라운드가 끝났다는 신호(get_self_training_batch()
        # 참고: round 0에는 self-training을 하지 않는다). _pending_data가
        # 비어 있어 아래에서 조기 반환하는 경우에도 반드시 실행되도록 맨
        # 앞에 둔다.
        self._has_prior_round = True
        if not self._pending_data:
            return
        all_data = torch.cat(self._pending_data, dim=0)
        self._update_basis(model, all_data)
        self._pending_data = []
        self._pending_count = 0
        self._pending_count_by_class = {}

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

        # forward가 예외를 던지면 훅이 안 지워진 채 남아 이후 라운드의
        # forward 호출마다 계속 발동해 activations를 오염시킬 수 있다 —
        # try/finally로 항상 해제되게 한다.
        try:
            model.eval()
            with torch.no_grad():
                model(data)
        finally:
            for h in handles:
                h.remove()

        # bias를 project_gradients()에서 weight와 함께 사영하려면, 기저
        # 자체가 "항상 1인 입력 차원"까지 포함한 증강 activation으로
        # 계산되어야 project 시 basis 차원이 grad_aug 차원과 맞는다 — bias는
        # 그 "항상 1" 방향의 gradient 성분이라는 수학적으로 정확한 대응이다.
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
            # 밀려나고 과거 보호는 우선 유지된다.
            ambient_dim = Q.shape[0]
            cap = max(1, int(ambient_dim * self.max_basis_ratio))
            self._basis[name] = Q[:, :cap]

    def _compute_basis(self, activation: torch.Tensor) -> torch.Tensor:
        # 공식 코드(평균 미중심화 + `+1` 없는 r=sum(cumsum<threshold))에
        # 문자 그대로 맞췄다가 실측으로 회귀를 발견해 되돌렸다 — NSL-KDD
        # af=gpm+as=cade_mad 일부 조합이 exp1에서 예측이 완전히 한 클래스로
        # 퇴화했다(30개 gpm 조합 전수검사: nsl-kdd 2/30 실패, unsw-nb15
        # 0/30). 중심화 제거·+1 제거 중 어느 한쪽만 되돌려도 통과했다 —
        # 공식 코드가 이 테스트베드의 얕은 아키텍처(은닉층 1개 + 작은
        # 분류기 head)에서는 기저가 지나치게 작아지거나 ReLU 출력의
        # 항상-양수 편향 방향을 그대로 기저로 채택해 분류기 학습에 필요한
        # 그래디언트까지 사영으로 제거하는 경계 사례를 만든다. 평균 중심화
        # + `+1` 여유 둘 다 원래대로 유지한다.
        #
        # residual projection(GPM Algorithm 2)을 두 차례 시도했다가 둘 다
        # 되돌림(클래스 docstring "기저 풀랭크 문제" 절 참고) — 1차는 잔차
        # 에너지 정규화, 2차는 공식 Eq-9 그대로. 둘 다 residual 없는
        # 버전보다 A/B 실측으로 나빴다(1차: f1 0.7006→0.6440, 2차: f1
        # 0.7006→0.5538, bwt도 둘 다 악화). 2차 시도는 basis 크기가 모든
        # 레이어에서 더 작게 수렴했는데도 bwt는 더 나빴다("더 작은 기저 =
        # 더 약한 보호 = 더 심한 망각"과 부합) — residual projection
        # 자체가 이 아키텍처(얕은 공유 backbone, latent_dim=32, 5
        # experience)에 안 맞는다는 결론이다. residual 없이 max_basis_ratio
        # 상한만 두는 현재 방식을 최종 채택한다.
        centered = activation - activation.mean(dim=0, keepdim=True)
        _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        energy = S ** 2
        cumulative = torch.cumsum(energy, dim=0) / energy.sum().clamp(min=1e-10)
        k = int((cumulative < self.activation_threshold).sum().item()) + 1
        k = max(1, min(k, Vh.shape[0]))
        return Vh[:k].T
