# 지표·구성요소·하이퍼파라미터 근거 기록

이 문서는 PRD(`CL-NIDS-Bench v2.5`) 0절/6절이 요구하는 "논문 근거 기록"을 위한
문서다. 코드에 반영된 모든 알고리즘적 판단은 아래에 근거를 남긴다.

## 사용자 지시로 추가된 변경 (PRD 4절 compatibility table 자체를 수정)

### A. anomaly_scorer='none' 추가 (Track A, 54개 조합 신규 → 전체 63→117개)

SSF/SPIDER는 원 논문에 별도 anomaly_scorer가 없다 — classifier head의
`sigmoid(logit) > 0.5`로 직접 판정한다. 기존 PRD 4.2절은 Track A의
anomaly_scorer를 `cade_mad` 하나로 고정했는데(SSF/SPIDER 원문 그대로의
판정 방식과 CADE의 MAD 채점 방식을 실제로 비교할 수 없는 구조), 사용자
지시로 `none`(분류기 자체 판정)을 두 번째 옵션으로 추가했다.

`NoAnomalyScorer`(`pipeline/common_baselines.py`)는 PRD 12.1절 "anomaly_scorer
전부 z 소비" 원칙의 유일한 예외다 — z가 아니라 classifier head가 낸 logit이
필요해서 `model.classifier(z)`를 다시 계산해야 하므로, `CLClient`가 표준 계약
밖의 선택적 훅 `set_model()`로 모델 참조를 한 번 넘겨준다(CNDIDSAntiForgetting.
on_experience_start와 같은 패턴). `compute_threshold()`는 SSF/SPIDER 원문 그대로
고정 0.5를 반환한다(라벨도 정상 참조 데이터도 쓰지 않음).

`TRACK_A_GRID["anomaly_scorer"] = ["cade_mad", "none"]`로 바뀌어 Track A가
54→108개, 전체가 63→117개로 늘었다(`common/compatibility.py`).

### B. NSL-KDD/UNSW-NB15 데이터 프로토콜 통일

기존에는 NSL-KDD만 `preserve_official_split=True`(원본 train/test 파일 분리
유지, 원본 행 순서 그대로 분할), UNSW-NB15는 `False`(병합 후 재분할)를 썼다 —
UNSW-NB15 원본 파일이 라벨 기준으로 사실상 정렬되어 있어(실측: 앞 20%는 전부
정상, 뒤 20%는 전부 공격), 행 순서를 그대로 유지한 채 나누면 일부 experience의
학습 데이터에 한쪽 클래스가 아예 없었기 때문이다.

SSF 원본(`ssf.py:104,155-159`)을 다시 대조한 결과, SSF도 원본 파일 행 순서를
그대로 쓰지 않는다는 것을 확인했다 — `train_test_split(x_train, ...)`(기본
shuffle=True)로 train을, `torch.randperm`으로 test를 각각 무작위로 섞은 뒤에야
스트리밍 윈도우로 처리한다. 그래서 "원본 행 순서 유지"는 SSF의 실제 방식보다
오히려 더 엄격했던 것으로 판단해, `dataset_loader.py`를 "각 파일(train/test)은
분리하되, 파일 내부는 고정 seed로 한 번 섞은 뒤 n_experiences 분할"로 바꾸고
(`_chunk_shuffled`), 이 방식을 NSL-KDD/UNSW-NB15 양쪽에 동일하게 적용해
프로토콜을 통일했다(`preserve_official_split` 기본값도 True로 변경). train과
test 데이터가 서로 섞이는 일은 없다는 핵심 원칙은 그대로 유지된다.

## Phase 0 — 셋업 관련 결정

### 1. SPIDER/GPM 원본 코드 부재

`spider/` 폴더는 README.md + LICENSE만 존재하며, 실제 SPIDER 저장소 코드
(GPM 구현 포함)는 클론되지 않았다. 따라서 `components/spider_gpm/`은 SPIDER
논문이 인용하는 일반 GPM 메커니즘의 원 논문(Saha, Garg, Roy, "Gradient
Projection Memory for Continual Learning", ICLR 2021,
https://openreview.net/forum?id=3AOj0RCNC2)의 알고리즘 설명을 직접 근거로
새로 작성했다. SPIDER 저장소의 실제 코드와 대조하지 못했다는 한계를 명시한다.

### 2. 원본 저장소(CADE/CND-IDS/SSF) 스모크 테스트 대체

PRD Phase 0 체크리스트는 "CADE/CND-IDS/SSF 원본 그대로 실행되는지 확인"을
요구하지만, 세 저장소의 `requirements.txt`가 서로 다른/충돌하는 torch 버전을
고정하고 있음을 확인했다(CADE: TensorFlow 1.x 스타일 API(`tf.train.
AdamOptimizer`), CND-IDS: `torch==2.5.1`, SSF: `torch==1.13.1`). 이 테스트베드
환경은 `torch==2.12.0`(CPU)을 사용 중이며, 실제로 `deepod` 패키지 설치
시도에서 torch가 1.13.0으로 강제 다운그레이드되어 환경이 깨지는 사고가
있었다(즉시 원복함). 동일한 위험을 반복하지 않기 위해, 원본 저장소를
직접 실행하는 대신 **소스 코드를 직접 읽어 파일:라인 단위로 알고리즘·수식·
하이퍼파라미터를 대조하는 방식**으로 Phase 0의 검증 의도(원 논문의 알고리즘이
올바르게 코드로 옮겨졌는지 확인)를 충족했다. 아래 절들이 그 결과다.

### 3. DIF(Deep Isolation Forest) — 자체 구현 채택

CND-IDS의 `AnomolyDetectors/DIF.py`는 `deepod.models.tabular.dif.
DeepIsolationForest`를 그대로 래핑한다. `deepod` 설치 시 위 2번의 환경 파손이
발생했고, 사용자에게 확인한 결과 자체 구현을 채택하기로 했다. Xu et al.,
"Deep Isolation Forest for Anomaly Detection" (2023) 원 논문 방식대로
**무작위 초기화된 여러 개의 얕은 신경망 표현(random representation ensemble)에
IsolationForest를 적용**하는 방식으로 직접 구현한다 (`components/novelty_baselines/dif_scorer.py`).
CND-IDS 저장소가 이 알고리즘을 `deepod` 라이브러리로 그대로 가져다 쓴 것이므로,
알고리즘 자체는 동일한 메커니즘이다.

## CADE 원문 대조 (서브에이전트 조사, file:line 확인)

- Contrastive AE: 대칭 encoder/decoder, 마지막 encoder 레이어는 활성함수 없음
  (`CADE/cade/autoencoder.py:52-107`).
- Contrastive loss: `L_con = is_same*dist + (1-is_same)*relu(margin-dist)`,
  `dist`=L2 거리 (`autoencoder.py:210-232`). 기본 `margin=10.0`
  (`cade/utils.py:72-73`).
- 총 손실: `loss = lambda_1 * L_con + L_AE(MSE)` (`autoencoder.py:232`), 기본
  `lambda_1=0.1` (`utils.py:68-69`).
- 클래스별 centroid = 학습셋 latent 평균 (`cade/detect.py:62`).
- MAD: `mad = 1.4826 * median(|d - median(d)|)` (`detect.py:150-158`,
  1.4826은 정규분포 일치 상수).
- MAD 정규화 거리: `A(x,i) = |‖z_x-centroid_i‖ - median(dis_i)| / mad_i`
  (`detect.py:91`). **`T_MAD=3.5`** 기본값 (`utils.py:77-78`).

→ 이 테스트베드 채택값: `components/cade/` — contrastive AE margin=10.0,
lambda_1=0.1 (CADEDriftDetector 내부 사설 encoder 학습에 사용),
`t_mad=3.5`(CADEMADScorer의 `compute_threshold` MAD 승수, `configs/component_hparams/cade.yaml`).

## CND-IDS 원문 대조

- Encoder/decoder: 128→256→128→latent(default 30) (`CND_IDS.py:17-35`).
- 손실 = `LwF_MSE * 0.1(LwF_strength) + Recon_MSE * 0.1(reg_strength) +
  TripletMarginLoss(margin=2, semihard mining)` (`CND_IDS.py:43,45,71-78,161-166`).
- PCA scorer: `pca_dim='auto'`(누적분산 95% 이상 최소 성분수), score=원공간
  재구성오차 절대값 평균 (`AnomolyDetectors/PCA.py:7-16,34`).
- LOF: 별도 모듈 없이 `sklearn.neighbors.LocalOutlierFactor(novelty=True,
  contamination=0.001)` 인라인 사용 (`main.py:149-151`).
- Best-F: `sklearn.metrics.precision_recall_curve` 기반 F1 최댓값 threshold
  선택 (`metrics.py:278-291`).
- `n_experiences=5` 기본값 확인 (`utils.py:62,92,126,156,186,215,244`).

→ 이 테스트베드 채택값: `cndids.lambda_r=0.1`(reconstruction 가중치),
`cndids.lambda_cl=0.1`(LwF/continual 가중치 — CND-IDS의 `LwF_strength`를
PRD의 `lambda_cl` 필드에 대응시킴; metric/triplet 항은 원 논문처럼 가중치 1
암묵 적용). PCA/LOF/DIF 모두 Best-F thresholding(3.5절) 사용.

## SSF 원문 대조

- AE: 은닉층 크기 `2**round(log2(input_dim))`, latent=`h/4` (`utils.py:427-456`,
  UNSW용 분류기 헤드 포함 버전은 `utils.py:39-63`).
- KL-mask 최적화: `optimize_old_mask`/`optimize_new_mask`, 고정 `steps=100`
  (CLI 노출 없음) (`utils.py:109-190`, `ssf.py:230-231`).
- K-S 검정: `scipy.stats.ks_2samp`, `drift_threshold=0.05` 기본값
  (`utils.py:646-658`, `ssf.py:49`).
- Strategic forgetting: 대표성 점수(M_c) 낮은 샘플 우선 제거
  (`utils.py:192-257,259-388`).
- LwF: MSE 기반 distillation(reconstruction/classifier 출력에 직접, 온도 없음),
  `lwf_lambda=0.5` (`ssf.py:45,292-334`).
- NSL-KDD 121차원 / UNSW-NB15 196차원 확인 (`ssf.py:52,54`).

→ 이 테스트베드 채택값: `ssf.kl_max_iter=100`(원 코드의 `steps` 기본값을
그대로 채택), K-S 검정 `drift_threshold=0.05`, LwF `lwf_lambda=0.5`.

## CND-IDS pseudo-label 생성 방식 정정 (사용자 피드백으로 재조사 후 수정)

최초 구현은 `CNDIDSAntiForgetting`의 pseudo-label을 z에 대한 단순 2-means로
근사했다. 사용자가 "기존 방식과 일치해야 의미가 있다"고 지적해 원본을 다시
정밀 대조한 결과, 실제 메커니즘이 다음과 같음을 확인했다:

- `CND_IDS.py:105-115`(실제 clusterer는 `FeatureExtractors/modules/K_Means.py`,
  **`AnomolyDetectors/K_Means.py`와는 다른 별개 클래스**)는 experience의 원본
  입력 x에 elbow-선택 K-Means(후보 [100,300,500,1000,2000])를 **한 번** 적용해
  `cluster_labels`를 얻고, `datastream.init_normal`(알려진 정상 참조 데이터)이
  속하는 클러스터 ID 집합을 구해, 그 집합에 속하면 0(정상)·아니면 1(신규)로
  pseudo-label을 부여한다. **공격 라벨은 전혀 쓰지 않는다.**
- 최초 조사 때 `AnomolyDetectors/K_Means.py`(실제 라벨이 섞인 캘리브레이션
  서브셋을 쓰는 별개의 standalone anomaly-scorer 베이스라인)와 혼동해 "CND-IDS가
  라벨을 쓴다"고 잘못 판단했었다. 두 파일은 이름만 같고 무관한 클래스다.

수정된 구현(`components/cndids/cndids_anti_forgetting.py`)은 원본 메커니즘을
그대로 따르되 두 가지를 테스트베드 스케일에 맞춰 조정했다:
- K 후보를 [100,300,500,1000,2000] → [5,10,20,30,50,80]로 축소 — 원본은
  label_budget 없이 experience 전체(수만 건)에 클러스터링하지만, 이 테스트베드는
  PRD 9.2절에 따라 Track B도 label_budget(기본 10%)만큼만 접근 가능해 스케일이
  훨씬 작기 때문. "여러 K를 elbow로 선택한다"는 메커니즘 자체는 동일.
- "알려진 정상 참조 데이터"로 이미 있는 `_normal_reference_raw`(12.6절)를
  그대로 재사용 — CND-IDS의 `datastream.init_normal`과 같은 역할.
- elbow 탐색 단계의 KMeans n_init을 3으로 줄임(원본/최종 fit은 sklearn 기본값
  10 유지) — 이 테스트베드는 experience마다 elbow 탐색을 반복하므로(원본은
  학습 세션당 한 번) 그대로 두면 라운드당 과도하게 느려짐. 클러스터 배정
  품질에 직접 영향을 주는 최종 fit은 그대로 두고 탐색 단계만 줄였다.

이 수정으로 Track B(9개 조합) 결과를 재실행했다(Track A는 영향 없음).

## GPM (Saha et al., ICLR 2021) — 논문 기반 신규 구현

SPIDER 저장소 코드가 없으므로 GPM 원 논문의 알고리즘 설명(Algorithm 1: 태스크
종료 시 레이어별 activation을 수집해 SVD, 누적 에너지 비율
`activation_threshold` 이상을 만족하는 최소 개수의 우특이벡터를 기저로 채택,
이후 태스크에서는 gradient를 이 기저의 직교여집합에 투영)을 직접 구현했다.
`activation_threshold` 기본값은 원 논문이 실험에서 사용한 범위(0.95~0.99,
데이터셋에 따라 조정)의 중간값인 **0.97**로 설정했다 — 특정 논문 실험의
정확한 재현이 아니라 이 테스트베드의 기본값이라는 점을 명시한다.
`components/spider_gpm/`는 이전 testbed 코드(`components/gpm/`)를 참고하지
않고 이 PRD의 `BaseAntiForgetting`(§12.5) 계약에 맞춰 처음부터 작성했다
(사용자 지시 — GPM 재사용 명시적 거부).

## memory_manager 재구성: fifo 삭제 → SPIDER 유한 버퍼 메모리로 대체, lof/dif 삭제

사용자가 "memory manager의 fifo는 어느 논문에서 가져온거야? anomaly scorer의
deep-svdd와 lof는?"이라고 질문해 확인한 결과, 기존 `FIFOMemoryManager`는
어느 논문에도 근거가 없는 일반 소프트웨어 엔지니어링 관행(단순 FIFO 큐)이었고,
`LOFScorer`/`DIFScorer`는 CND-IDS 원 논문 자체가 제안한 방법이 아니라 CND-IDS가
비교 실험을 위해 인용한 제3자(다른 논문) baseline이었다(PRD Appendix A가
"CND-IDS 자체 방법(pca)"과 "CND-IDS 비교 baseline(lof, dif)"을 이미 구분해
놓았음). 이는 PRD 0절의 "네 논문이 실제로 제안한 메커니즘만 재조합한다"는
원칙에 어긋나므로, 사용자 지시로 다음과 같이 정리했다:

1. **`memory_manager='fifo'` 삭제 → `memory_manager='spider'`로 대체.**
   사용자가 SPIDER 원 논문을 직접 확인해 제공한 내용에 근거해 새로 구현
   (`components/spider_gpm/spider_memory_manager.py`). SPIDER는 GPM(이미
   anti_forgetting 슬롯의 `gpm`으로 구현됨)과 별도로 "유한 버퍼 메모리(M)"를
   두며, 세 가지 특징을 갖는다: (a) 라벨 없는 샘플만 저장(privacy-preserving),
   이전 태스크에서 무작위 선택, (b) 복잡한 memory reconstruction policy 없이
   experience 종료 시 버퍼 전체를 현재 태스크의 무작위 샘플로 완전히 교체
   ("No MRP"), (c) replay 시 라벨이 없으므로 바로 직전 태스크까지 학습된
   모델(f_θ^(t-1))의 스냅샷으로 실시간 pseudo-labeling. (c)는 표준
   `BaseMemoryManager` 계약 밖의 정보(모델 참조)가 필요해, `NoAnomalyScorer.
   set_model()`/`CNDIDSAntiForgetting.on_experience_start()`와 같은 패턴으로
   선택적 훅 `set_snapshot_model()`을 추가했다 — `CLClient`가 step 8
   (`anti_forgetting.on_task_end`와 같은 시점)에서 그 라운드까지 학습된
   모델의 deepcopy를 넘겨준다(`pipeline/cl_client.py`).
   "라벨 없는 무작위 교체 버퍼"라는 핵심 성질은 Track A/B 어느 backbone과도
   무관하게 동일하게 적용 가능하므로, 사용자 지시로 Track A/B 양쪽
   `memory_manager` 그리드에 모두 포함시켰다(Track B의 `CNDIDSAntiForgetting`은
   애초에 replay 라벨을 쓰지 않으므로 pseudo-label 값 자체는 영향이 없다).
2. **`anomaly_scorer='lof'`, `'dif'` 삭제, Track B는 `'pca'`만 유지.**
   PCA만 CND-IDS 원 논문 자체의 제안 방법이다. `components/novelty_baselines/
   lof_scorer.py`, `dif_scorer.py`를 삭제하고 위 3번 문단의 DIF 자체구현 채택
   결정은 이 삭제로 무효화되었다(파일 자체가 더 이상 없음).

결과: `TRACK_A_GRID`는 108개 그대로, `TRACK_B_GRID`는 9개(1·1·1·3·3) →
3개(1·1·1·3·1)로 축소, 전체 조합 수는 117개 → **111개**로 변경했다
(`common/compatibility.py`).

## drift_detector 무효 조합 제거 (Track A 108개 → 최종 90개, 전체 111개 → 최종 93개 — 최초 시도는 72/75였다가 아래 "정정" 절에서 90/93으로 재수정됨)

111개 조합의 실제 결과(`leaderboard_for_chart.json`, 222행)를 슬롯별로
전수 대조한 결과, `drift_detector`만 완전한 중복을 만드는 축임을 확인했다
(다른 4개 슬롯은 어떤 값을 바꿔도 f1/precision/recall이 단 한 쌍도 완전히
일치하지 않음 — 실제로 소비되고 있다는 뜻).

`drift_detector`의 출력은 `SSFSampleSelector`(drift_score 소비)와
`SSFMemoryManager`(drift_detected 소비) 두 곳에서만 쓰인다. 게다가 SSF/CADE
두 detector 모두 "비교할 과거 표본"이 있어야 하는데, `memory_manager='none'`
이면 애초에 버퍼(`buf_ref`)가 없어 비교 자체가 불가능하다. 따라서
`sample_selector='random'`이면서 `memory_manager`가 `'none'`/`'spider'`인
경우, 그리고 `memory_manager='none'`인 경우(sample_selector 무관)는
`drift_detector` 값이 파이프라인 어디에도 영향을 줄 수 없다.

실측 대조: 이 조건에 해당하는 (dataset×track×sample_selector×memory_manager×
anti_forgetting×anomaly_scorer) 216개 dd-비교 쌍 중 36쌍에서 `dd=none`과
`dd=ssf`의 f1/precision/recall이 소수점까지 완전히 동일했다. 반면 `dd=cade`는
같은 조건에서도 단 한 번도 동일해지지 않았는데(`CADEDriftDetector.fit()`이
사설 contrastive AE를 실제로 학습시키며 전역 RNG를 소비, 이후 랜덤 시드
흐름을 오염시켜서), 이는 "drift 신호 효과"가 아니라 순수 RNG 오염 잡음이므로
오히려 더 비교 근거가 없는 차이다.

이 두 발견(무의미한 완전 동률 36쌍 + 근거 없는 RNG 오염으로 인한 가짜 차이)에
따라, 사용자 지시로 그리드 자체를 축소했다 — "생성 후 제외"가 아니라
`enumerate_valid_combos()`가 `TRACK_A_DD_ACTIVE_SS_MM`(drift_detector가 실제로
소비되는 (sample_selector, memory_manager) 3개 조합: `(random,ssf)`,
`(ssf,spider)`, `(ssf,ssf)`)에 대해서만 drift_detector 3개 값을 전부
순회하고, 나머지 3개 비활성 조합(`(random,none)`, `(random,spider)`,
`(ssf,none)`)에서는 `TRACK_A_DD_INERT_VALUES`만 순회하도록 직접 구성한다
(`common/compatibility.py`).

### 정정 — 첫 구현이 "완전 동일 중복 제거"를 넘어 과도하게 축소했음

**최초 구현은 비활성 조합에서 `drift_detector='none'` 하나로만 고정했는데,
이는 사용자가 실제로 요청한 범위를 넘어선 과도한 축소였다.** 사용자가 지운
대상은 "값까지 소수점 단위로 완전히 동일해지는 진짜 중복"(`dd=none`≡`dd=ssf`)
뿐이었는데, `dd=cade`는 값 자체가 `dd=none`과 다르므로(RNG 오염 때문이지
신호 소비 때문은 아니지만, "완전히 동일"은 아니다) 중복이 아니다. 그런데도
비활성 조합에서 `dd=cade`까지 함께 제거해버렸고, 그 결과 예전 111개 조합
기준 UNSW-NB15 1위였던 `A_dd=cade_ss=random_mm=spider_af=none_as=cade_mad`
(F1=0.8824)가 새 리더보드에서 사라져 사용자가 지적했다.

**수정**: 비활성 (sample_selector, memory_manager) 조합에서는
`TRACK_A_DD_INERT_VALUES = ["none", "cade"]`만 순회한다 — `'ssf'`만
`'none'`과의 완전한 중복이라 제외하고, `'cade'`는 값이 다르므로 남긴다.
`validate_combo()`도 이에 맞춰 비활성 조합에서 `dd not in
TRACK_A_DD_INERT_VALUES`(즉 `dd='ssf'`)만 거부하도록 수정했다.

결과: Track A 108개 → **90개**(6개 (ss,mm) 셀 중 활성 3개는
dd=3개×af3×as2=18개, 비활성 3개는 dd=2개(`none`,`cade`)×af3×as2=12개 →
3·18+3·12=90), 전체 111개 → **93개**로 축소했다(72/75가 아니라 90/93이
최종값). 기존 111개 조합 기준으로 이미 생성된 `results/`의 222개 결과 중
93개 조합(×2데이터셋=186개)은 그대로 유효하며 재실행 없이 즉시 쓸 수 있다
(`dd=cade` 결과 파일도 애초에 삭제한 적이 없어 그대로 남아있었다) — 남는
파일은 `dd='ssf'`가 비활성 조합에 있던 18개 조합(×2데이터셋=36개)뿐이며,
이는 `dd='none'`과 완전히 동일한 결과이므로 버려도 정보 손실이 없다.

## GPU 이식성 확보 + 하이퍼파라미터 상향 + CICIDS2018 추가 (2026-07-28)

사용자가 "학습이 너무 빨리 끝난다"며 네트워크 크기·에폭을 논문 수준으로
올리고 싶어했고, 그러려면 GPU가 사실상 필요했다. 이 개발 환경(현재 노트북)은
`torch==2.12.0+cpu`(CUDA 미포함 빌드)이고 그래픽 카드도 Intel Iris Xe
내장뿐이라(NVIDIA/AMD 별도 GPU 없음, `nvidia-smi` 자체가 없음) CUDA를 쓸 수
없다 — 사용자가 연구실 GPU 서버로 옮겨 실행하기로 했다.

### 1. 디바이스 이식성 버그 8건 수정

코드 자체는 애초에 CPU 전용으로 설계되지 않았다 — `CLClient`/`grid_runner.py`
모두 `device` 인자를 받아 `torch.device(device)`로 넘긴다. 다만 CPU에서만
돌려온 탓에 안 드러난 디바이스 불일치 버그가 여럿 있었다(정적 감사로 발견,
CUDA 하드웨어가 없어 실기 검증은 못 했지만 CPU 재실행으로 수치 회귀가
없음은 확인함):

- `CADEDriftDetector`가 사설 `ContrastiveAutoEncoder`를 자체 소유하는
  유일한 컴포넌트인데, `CLClient`는 `self.model`만 `.to(device)`하고 이
  컴포넌트는 옮기지 않고 있었다 — `to(device)` 메서드를 추가하고
  `CLClient`가 `hasattr(component, "to")` 패턴으로 호출하도록 연결
  (`components/cade/cade_drift_detector.py`, `pipeline/cl_client.py`).
- `SSFSampleSelector`(`edges`/`target_dist`/`mask_logit`/`bin_weights`/
  `drift_weight` 텐서), `SSFMemoryManager`(`_representativeness`의
  `torch.zeros`, `get_replay_batch`의 `torch.randperm`),
  `CNDIDSAntiForgetting`(`_pseudo_labels_for_batch`의 폴백/생성 텐서),
  `SPIDERMemoryManager`/`CNDIDSMemoryManager`(`get_replay_batch`의
  `torch.randperm`), `CADEMADScorer`/`PCAScorer`(fallback `torch.zeros`,
  `torch.tensor(1e-8)`) — 전부 `device=` 없이 새 텐서를 만들어서, 모델이
  CUDA에 있으면 CPU 텐서와 섞여 RuntimeError가 날 지점이었다. 소스 텐서의
  `.device`를 그대로 넘기도록 수정.
- `grid_runner.py`/`smoke_test.py`의 추론 레이턴시·predict-shape 검증
  부분도 test 텐서를 device로 안 옮기고 모델에 바로 넣고 있어서 같은
  문제였다 — `.to(client.device)` 추가.
- 수정 과정에서 `SSFSampleSelector`의 `torch.tanh(torch.tensor(drift_score))`를
  `math.tanh(float(drift_score))`로 바꿨다가 float32/float64 정밀도 차이로
  top-k 선택이 실제로 바뀌는 회귀를 직접 발견했다(4개 대표 조합을
  `results/`의 기존 값과 대조해서 적발) — `torch.tanh`/텐서 계산은 그대로
  두고 `device=`만 추가하는 것으로 정정. 이 검증 방식(재실행 후 기존
  결과와 소수점까지 비교) 덕분에 CPU 동작이 이번 수정으로 전혀 안 바뀌었음을
  확인할 수 있었다.

### 2. `--device` CLI 플래그, `requirements.txt`

`grid_runner.py`/`smoke_test.py`에 `--device`(cpu/cuda) 인자를 추가했다.
Windows 콘솔(cp949) 인코딩으로 `--help`가 크래시하는 것도 발견해 argparse
`description`에서 em-dash를 빼서 고쳤다(이 프로젝트에서 반복돼 온 Windows
콘솔 인코딩 문제와 같은 종류). `testbed/requirements.txt`를 새로 작성해
현재 확인된 의존성 버전과 CUDA 빌드 torch 설치 방법을 기록했다.

### 3. `epochs_per_experience` 10→200, `hidden_dim`/`latent_dim` (128,32)→(256,64)

사용자 지시. 근거 없이 2배로 키운 게 아니라 — SSF 원 논문 공식
(`utils.py:28-36`)을 UNSW-NB15(196차원)에 적용하면 정확히 hidden=256/
latent=64가 나온다(이미 "SSF 원문 대조" 절에서 확인한 수치). 두 데이터셋에
동일 아키텍처를 강제해야 하는 이 테스트베드 제약상, 더 큰 쪽(UNSW-NB15)의
논문 공식값을 공통 기본값으로 채택했다. 에폭 200은 각 논문의 실제 학습
규모(SSF epochs=4+온라인 1, CND-IDS train_epochs=20, SPIDER max_epochs=100)에
근접하도록 올린 값이다. 이 값으로 CPU에서 전체 그리드(93개 조합×2데이터셋)를
돌리면 기존 실측(10에폭 기준 총 30.5분)에서 추정할 때 20~40시간 규모라
GPU 실행을 전제로 한다. `configs/global_hparams.yaml`을 직접 덮어써서
기존 `results/`의 모든 결과는 이 시점부터 새 설정 기준으로 재실행해야
유효하다(이전 10에폭/128·32 기준 결과와는 직접 비교 불가).

### 4. CICIDS2018 데이터셋 추가

`data/dataset_loader.py`에 `_load_cicids2018_raw()`/
`_read_cicids2018_features()`를 추가했다. NSL-KDD/UNSW-NB15와 달리 공식
train/test 분리 파일이 없는 데이터셋이라(원본이 일별 캡처 CSV들로만
배포됨), `load_dataset('cicids2018', ...)`는 `preserve_official_split`을
항상 False로 강제해 병합+재셔플 프로토콜만 적용한다. 사용자가 원본
CSE-CIC-IDS2018 CSV 파일들을 직접 구해 `<repo_root>/CICIDS2018/*.csv`에
넣는 방식으로 연동한다(폴더 안의 모든 CSV를 자동으로 찾아 병합).

전처리 시 Flow ID/Src IP/Dst IP/Timestamp 같은 식별자 컬럼은 일반화 불가능한
피처라 제외하고, `Label` 컬럼은 'Benign'이 아니면 전부 공격(1)으로
이진화한다. CICIDS2018 원본에 알려진 결측치/Infinity/헤더 중복 행 문제가
있어 숫자 변환 실패·inf 값은 해당 행째로 제거한다. 실제 원본 데이터 없이
동일한 스키마 문제(inf/NaN 섞인 합성 CSV)를 재현한 합성 데이터로 로더
로직을 끝까지 검증했다 — 실제 CSE-CIC-IDS2018 파일의 정확한 컬럼 구성과는
다를 수 있어(배포 버전에 따라 78~83개 컬럼으로 조금씩 다를 수 있음), 사용자가
실제 파일을 넣고 처음 실행했을 때 컬럼명이 예상과 다르면 조정이 필요할 수 있다.

## kagglehub 자동 다운로드 연동 + 실데이터로 셔플 누락 버그 발견 (2026-07-28)

사용자가 `kagglehub.dataset_download("primus11/cic-ids-2018-dataset")`을
직접 써서 받고 싶어했다. `_load_cicids2018_raw()`를 수정해 `<repo_root>/
CICIDS2018/*.csv`에 수동으로 넣은 파일이 없으면 이 함수로 자동 다운로드
(Kaggle API 토큰 필요)하도록 폴백을 추가했다. `import kagglehub`를 모듈
최상단에 두면 kagglehub가 없는 환경에서 `grid_runner.py`/`smoke_test.py`
전체가 임포트 단계에서 깨지므로(단위 테스트는 이 모듈을 안 거쳐 못 잡아냄),
함수 안으로 옮겨 지연 임포트(lazy import)로 처리했다.

실제로 이 명령을 실행해 검증한 결과(인증 없이 바로 성공, 76.4MB 다운로드,
`primus11/cic-ids-2018-dataset`는 단일 CSV `cic.csv`, 1,048,575행×80컬럼,
`Benign`/`FTP-BruteForce`/`SSH-Bruteforce` 3-클래스), **`preserve_official_split
=False` 경로(`_chunk_by_row_order`)가 셔플 없이 원본 행 순서 그대로 청크를
나누는 심각한 버그를 실데이터로 발견했다** — CICIDS2018 원본 파일이 공격
유형별로 뭉쳐서 정렬돼 있어(UNSW-NB15 원본 파일의 라벨 정렬 문제와 동일한
종류, 9.1절 docstring 참고), 셔플 없이 5등분하면 experience별 양성 비율이
`0.998 → 0.825 → 0.000 → 0.000 → 0.000`으로 완전히 붕괴했다(실측). 합성
CSV로만 검증했을 때는 합성 데이터 자체가 이미 무작위로 생성돼 있어 이
문제가 드러나지 않았다 — 실제 원본 데이터로 검증해야만 잡을 수 있는
종류의 버그였다.

수정: `else`(`preserve_official_split=False`) 분기에서 병합한 풀을
`_chunk_by_row_order()`로 청크 나누기 전에 `_shuffle()`로 고정 seed 셔플을
추가했다. 재검증 결과 5개 experience 모두 양성 비율이 0.364~0.366으로
균일해졌다. 이 분기는 현재 CICIDS2018 전용으로만 쓰이고(NSL-KDD/UNSW-NB15는
전부 `preserve_official_split=True` 기본값을 씀), 그 경로는 안 건드렸으니
기존 결과에는 영향이 없다.

## CICIDS2018 자동 다운로드를 kagglehub → 공식 AWS S3로 교체 (2026-07-28)

사용자가 "CIC-IDS는 원래 큰 데이터셋으로 아는데 왜 작냐"고 지적해 직접
대조한 결과, kagglehub `primus11/cic-ids-2018-dataset`가 **공식 10일 중
하루(2018-02-14, FTP/SSH 브루트포스일)만, 그마저 잘려 있음**을 확인했다:
- 공식 `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv`는 341.6MB인데
  kagglehub로 받은 건 76.4MB.
- 행 수가 정확히 1,048,575개 — 엑셀 시트 최대 행 수(2^20=1,048,576, 헤더
  포함)와 일치해, 업로드 과정에서 엑셀 등을 거치며 그 이상이 잘려나간
  것으로 보인다.
- 라벨도 `Benign`/`FTP-BruteForce`/`SSH-Bruteforce` 3종류뿐 — 공식
  10일치에 있는 DoS/DDoS/Botnet/Web Attack/Infiltration 등은 전혀 없다.

**교체한 소스**: AWS Open Data Registry의 공식 버킷(`s3://cse-cic-ids2018/`,
registry.opendata.aws/cse-cic-ids2018 — 계정/자격증명 없이 익명(unsigned)
접근 가능, 직접 `list_objects_v2`로 실측). 버킷은 두 폴더로 구성된다:
- `Original Network Traffic and Log data/` — 원본 pcap, 하루치가
  40~60GB(전체 10일 초 400GB) — 대상 아님.
- `Processed Traffic Data for ML Algorithms/` — CICFlowMeter로 이미 피처
  추출된 하루별 CSV 10개, 합계 **약 6.9GB**(실측: 107MB~4.05GB/파일).

`_download_cicids2018_from_s3()`를 새로 작성해 이 10개 CSV를 전부
`<repo_root>/CICIDS2018/`에 받도록 했다(이미 같은 크기의 파일이 있으면
스킵). `_load_cicids2018_raw()`는 이제 kagglehub가 아니라 이 함수를
폴백으로 쓴다. 실제로 가장 작은 파일(Thursday-01-03-2018, 107MB, 잘리지
않은 원본)을 받아 전체 파이프라인(로딩→피처 추출→셔플→청크→80/20 분할)을
끝까지 검증했다 — 78개 피처, 5개 experience 전부 양성 비율 27~29%로
균일함을 확인했다. 10개 파일 전체(6.9GB, 수천만 행 규모)를 한 번에 받는
것까지는 이 개발 환경(노트북)에서 검증하지 못했다 — 첫 실행은 GPU
서버에서 이뤄질 것이다.

`_read_cicids2018_features()`의 반환 dtype을 `float64`→`float32`로 바꿨다
— 10일치를 다 이어붙이면 수천만 행 규모라 메모리를 불필요하게 두 배 쓰는
것을 피하기 위함이다(NSL-KDD/UNSW-NB15는 행 수가 훨씬 적어 float64
유지). `requirements.txt`에서 `kagglehub`를 `boto3`로 교체했다.

## Track별 에폭 분리 (Track A=200, Track B=20) + 전체 라인별 최종 검토 (2026-07-29)

사용자 지시로 `epochs_per_experience_track_b=20`(CND-IDS 원 논문 값)을
Track B에만 적용하도록 분리했다(`global_hparams.yaml`, `grid_runner.py`/
`smoke_test.py`의 `run_combo_full`/`run_smoke_test_for_combo`에서
`combo["track"]=="B"`일 때 오버라이드). 근거는 이전 절(kagglehub→S3 교체
바로 위 실측 비교)의 catastrophic forgetting 분석 그대로.

이어서 사용자 지시로 전체 코드베이스를 디렉터리별로 한 줄씩 다시 읽으며
"테스트베드로서의 역할, gold-plating, 보여주기식 코드, 구조적 합리성"을
점검했다. 발견하고 수정한 것:

1. **BWT 공식 버그(실제 계산 오류)**: `common/metrics.py`의 `bwt()`가
   분모를 `T(T-1)/2`로 쓰고 있었는데, "CND-IDS 원 논문 공식 그대로"라는
   docstring의 주장과 달리 실제 CND-IDS 코드(`CND-IDS/AutonomousDCN/
   ADCNmainloop.py:418` — `BWT = 1/(nTask-1)*(sum(allTaskAccuracies)-
   sum(postTaskAcc))`, 406행 주석 "except the last task")를 직접 대조해보니
   분모는 `(T-1)`이고 마지막 태스크는 애초에 합산에서 제외돼야 했다. T=5
   기준 이전 값은 실제 CND-IDS 공식 대비 2.5배 작게(0에 더 가깝게) 나오고
   있었다 — F1 순위나 조합 간 비교에는 영향 없지만(모든 조합이 같은 T라
   같은 배율로 축소됐을 뿐), 절대 BWT 수치 자체는 전부 틀려 있었다.
   `common/metrics.py`/`tests/test_metrics.py` 수정, `results/*.json` 186개
   전부 이미 저장된 `perf_matrix`로부터 재계산(재실행 불필요), 이 문서에
   기록된 이전 BWT 수치(예: epoch=200/20 비교)들도 이 버그가 있던 시점의
   값이라 절대 크기는 이 수정 이후 재실행하면 달라진다(방향과 결론은
   동일).
2. **SPIDERMemoryManager GPU 버그**: `update()`의 `torch.randperm(n)`에
   `device=`가 빠져 있었다 — CPU에서는 안 드러나지만 GPU에서 `selected_data`
   (CUDA)를 CPU 인덱스 텐서로 인덱싱하면 크래시한다. 같은 파일의
   `get_replay_batch()`, 다른 memory_manager들은 전부 `device=`가 있는데
   이 한 곳만 이전 GPU 이식성 수정 때 빠뜨렸었다. 수정 완료.
3. **죽은 코드**: `common/scenario_loader.py`(`load_scenario()`)와
   `testbed/scenarios/*.yaml` 2개 파일이 실제 실행 경로(`grid_runner.py`/
   `smoke_test.py`) 어디에서도 호출되지 않는다 — PRD 원래 구조(9절)에
   있던 것을 만들어만 놓고, 실제로는 `labeling_budget`/`n_experiences`를
   `global_hparams.yaml`과 코드에 직접 하드코딩하는 방식으로 구현이
   진행되어 이 파일들이 고아가 됐다. 삭제할지 실제로 연결할지는 사용자
   결정 필요 — 아직 손대지 않았다.
4. **문서 정합성**: `base/anomaly_scorer.py` docstring이 이미 삭제된
   `lof`/`dif`를 Track B 예시로 계속 언급하고 있어 수정. `global_hparams.yaml`
   최상단 "조합별로 다르게 튜닝하지 않는다" 원칙 옆에 Track별 에폭 예외를
   명시적으로 덧붙였다(안 그러면 바로 아래 `epochs_per_experience_track_b`
   와 자기모순으로 보임). `n_experiences` 주석의 "CICIDS2017" 인용이
   "CICIDS2018"과 혼동될 수 있어 명확히 구분하는 문구 추가.

발견했지만 낮은 우선순위라 손대지 않은 것: `backbone_type`/
`required_backbone` 클래스 속성이 모든 컴포넌트에 선언은 되어 있지만
런타임에 한 번도 검사되지 않는다(현재는 `common/compatibility.py`의
그리드 자체가 애초에 잘못된 조합을 못 만들게 막고 있어 실질적 위험은
없음 — 추가 방어선을 원하면 `CLClient.__init__`에서 한 번 assert하는
정도로 값싸게 강화 가능). `GPMAntiForgetting._update_basis`가 Track A의
손실에 전혀 안 쓰이는 decoder 레이어의 activation basis(SVD)도 매번
계산한다(틀린 건 아니고 그냥 낭비 — Track A는 x_hat을 안 쓰므로 decoder
그래디언트가 항상 None이라 `project_gradients`에서 걸러지긴 함).

## CICIDS2018 GPU 실행 중 발견된 문제 일괄 수정 (2026-07-30)

CICIDS2018을 GPU에서 실제로 돌리는 과정에서 스모크 테스트가 특정 조합에
대해 "score 분포가 사실상 상수 출력"으로 FAIL을 내는 것을 계기로, 관련된
전체 파라미터를 원 논문 코드와 다시 대조했다. 아래는 그 과정에서 확인된
실제 버그와 수정 내역이다.

1. **`hidden_dim` 계산 오류(가장 영향이 큼)**: 이전 세션에서 SSF 공식
   (`SSF-Strategic-Selection-and-Forgetting/utils.py:28-36`,
   `nearest_pow2=2**round(log2(input_dim))`, `hidden=nearest_pow2//2`,
   `latent=nearest_pow2//4`)을 UNSW-NB15(196차원)에 적용한 결과를
   `hidden=256`으로 기록했었는데, 직접 재계산하면 `nearest_pow2=256`
   자체를 hidden으로 잘못 대입한 계산 오류였다 — 올바른 값은
   `hidden=128`(`latent=64`는 원래 맞았음). `global_hparams.yaml`의
   `hidden_dim`을 128로 정정. 이 값은 Track A/B 모든 조합·모든 데이터셋의
   공유 아키텍처에 쓰이므로 영향 범위가 가장 크다.
2. **`labeling_budget` 방식 오류**: 이전에는 `{mode: fixed_ratio,
   value: 0.1}`(전체의 10%)를 썼는데, 이건 이 테스트베드가 임의로 만든
   값이었다. SSF 원 논문(`ssf.py:23`, `--num_labeled_sample` 기본값 200)은
   라운드당 고정 개수 200을 라벨링 예산으로 쓴다. 비율 방식은 CICIDS2018처럼
   experience당 행 수가 많은 데이터셋에서 라벨 예산이 수십만 개로 불어나
   SSF 원 논문의 설계 의도(적은 라벨로 학습)를 벗어난다 — CICIDS2018에서
   관찰된 이상 현상의 근본 원인 중 하나. `global_hparams.yaml`에
   `labeling_budget: {mode: fixed_count, value: 200}`로 옮기고,
   `grid_runner.py`/`smoke_test.py`가 하드코딩 대신 이 값을 읽도록 수정.
3. **Track B `batch_size` 미분리**: CND-IDS 원 논문 실제 값(`CND_IDS.py:103`,
   `batch_size = 64`)이 Track A(SSF 근거, 128)와 다른데 지금까지 전 트랙에
   128을 그대로 적용했다. `epochs_per_experience_track_b`와 같은 논리로
   `batch_size_track_b: 64`를 추가하고 `grid_runner.py`/`smoke_test.py`
   양쪽에 Track B 오버라이드를 반영.
4. **CICIDS2018 로더가 어느 논문 선례도 따르지 않던 문제**: 기존 로더는
   10일치 전체(~1600만 행)를 그대로 병합하고 `Dst Port`/`Protocol`도
   raw 숫자로 MinMaxScaler에 넣었다. CICIDS2018을 실제로 쓴 유일한 논문인
   CADE(`CADE/IDS_data_preprocess/clean_data.py`,`gen_IDS_data.py`)는
   완전 중복 행 제거, `Dst Port` 빈도 기반 3단계 범주화 후 원-핫,
   `Protocol` 원-핫을 수행한다 — `dataset_loader.py`에 동일하게 반영
   (`_dedup_rows`, `_bucket_port_frequency`, `_one_hot`). 단, CADE 자신의
   실험 설계(특정 날짜 하루 + 특정 공격 유형 2종 + 10% 다운샘플링)는
   CADE 고유의 "신규 공격군 드리프트 탐지"라는 연구 질문에 맞춘 것이라
   그대로 가져오지 않았다 — 10일치·전체 공격 유형 비율은 유지한 채,
   `CICIDS2018_SUBSAMPLE_TARGET=200,000`(NSL-KDD 전체 풀 148,517행/
   UNSW-NB15 257,673행과 같은 자릿수, 테스트베드 자체 기본값)으로 층화
   랜덤 서브샘플링만 적용한다.
5. **`normal_reference_size=500`은 애초에 논문에 없는 개념임을 재확인**:
   CADE 원 논문의 median/MAD 계산(`cade/detect.py:151-158`)은 고정 참조
   샘플이 아니라 해당 클래스 학습 데이터 "전체"로 계산한다. 이 테스트베드는
   매 라운드 전체 데이터를 재인코딩하는 계산 비용을 피하려고 고정 크기
   참조 표본으로 단순화한 것 — 4번 항목의 서브샘플링으로 전체 모집단
   규모가 다른 두 데이터셋과 비슷해지면서 이 500이라는 고정값의 대표성
   문제도 함께 완화된다.
6. **문서 정합성**: `cndids.yaml`의 "global_hparams.latent_dim=32" 주석이
   실제 현재 값(64)과 어긋나 있어 수정. `cade.yaml`의 `encoder_epochs=5`
   근거 설명이 옛 "라벨 예산 ~10%" 전제를 인용하고 있어 새 fixed_count=200
   기준으로 갱신.

낮은 우선순위로 보류한 것: CADE 원 논문의 IDS2018 전용 실행 설정
(`CADE/run_cade_exp_ids_infiltration.sh:13`, `--cae-hidden 64-32-16`,
은닉층 2개)은 이 테스트베드의 `ContrastiveAutoEncoder`(은닉층 1개, 현재
global_hparams 오버라이드로 hidden=128/latent=64 적용)와 구조 자체가
다르다. Track A 전체(SSF/CADE/GPM)에 통일된 아키텍처를 강제한다는 기존
설계 원칙과 CADE 고유 아키텍처 재현 사이에 트레이드오프가 있어 사용자
판단이 필요한 채로 남겨뒀다.

## 전체 코드베이스 최종 라인별 재검토 (2026-07-30, 참조 논문 코드 직접 대조)

이전 검토들이 "내부 정합성"에 집중했던 것과 달리, 이번에는 `base/`,
`common/`, `components/` 전 파일, `pipeline/cl_client.py` 전체, `experiments/`,
`tests/`를 SSF/CADE/CND-IDS 원본 저장소 코드, 그리고 GPM 논문(SPIDER 저장소는
여전히 코드 없음)의 **공식 GitHub 저장소**(`github.com/sahagobinda/GPM`,
`main_pmnist.py` — 이번에 WebFetch로 처음 실제 확인)와 한 줄씩 대조했다.

### 실제로 고친 것: GPM이 공식 코드와 다르게 구현되어 있었음

`components/spider_gpm/gpm_anti_forgetting.py`의 이전 기록은 "GPM 원 논문
Algorithm 1"이라고 표현했지만, 공식 코드(`main_pmnist.py`)와 대조한 결과
실제로는 두 지점에서 달랐다 — 파일 상단 docstring에 대조 내역을 남기고 아래
두 가지를 공식 코드와 동일하게 수정했다:
1. **평균 중심화(centering)**: 이전 구현은 SVD 전에 activation에서
   평균을 뺐다(`centered = activation - activation.mean(...)`). 공식 코드는
   원본 activation에 그대로 SVD를 적용한다(`np.linalg.svd(activation, ...)`,
   중심화 없음). 제거했다.
2. **기저 개수 off-by-one**: 이전 구현은
   `k = sum(cumulative < threshold) + 1`였다. 공식 코드는
   `r = sum(cumsum(energy_ratio) < threshold)`로 `+1`이 없다(`U[:,0:r]`).
   `+1`을 제거했다.

**의도적으로 공식 코드를 따르지 않고 유지한 것**: 공식 코드는 태스크마다
새 기저를 `np.hstack`으로 이어붙이기만 하고, 차원이 넘치면 그냥 앞부분만
잘라낼 뿐 QR 재직교화를 하지 않는다. 이 테스트베드는 5개 experience에 걸쳐
기저가 계속 누적되므로, 서로 다른 태스크의 SVD 결과를 그대로 이어붙이기만
하면(태스크 간 직교성이 보장되지 않아) `basis @ basis.T`가 참된 직교
사영행렬이 아니게 될 수 있다 — 그래서 QR 재직교화만은 공식 코드와 다르게
유지한다(이전 기록이 이걸 "원 논문 그대로"라고 잘못 표현했던 것만 정정).

### 발견했지만 코드는 바꾸지 않고 문서로만 남긴 것 (근본적인 다중 논문 통합 트레이드오프)

이 항목들은 전부 "이 테스트베드가 SSF/CADE/CND-IDS/GPM 네 가지 서로 다른
아키텍처를 **하나의 공유 백본**으로 통합해야 한다"는 근본 제약에서 나오는
필연적 단순화다. 어느 한쪽에 맞추면 다른 쪽에서 어긋나므로, "고친다"가
성립하지 않는다 — 대신 정확히 무엇이 다른지 여기 기록한다.

1. **분류기가 z에서 바로 나온다 vs SSF는 decode(x_hat)에서 나온다**:
   SSF의 실제 `AE_classifier.forward()`(`utils.py:59-61`)는
   `classify = self.classifier(decode)`다 — 분류기가 **재구성 x_hat**을
   입력으로 받고, `nn.Sequential(ReLU, Linear(input_dim,1), Sigmoid)`로
   확률을 낸다. 이 테스트베드의 `FCLAutoEncoder.classifier`는
   `nn.Linear(latent_dim,1)`이고 **z(잠재표현)을 직접** 입력받으며 Sigmoid도
   없다(raw logit). CADE는 애초에 분류기와 CAE가 완전히 분리된 별도 모델
   (`cade/classifier.py`의 MLPClassifier는 raw feature나 별도 관리되는
   피처를 입력받지, CAE의 z를 공유하지 않음)이고, CND-IDS는 지도 분류기
   헤드 자체가 없다(pseudo-label + metric learning). 즉 SSF·CADE·CND-IDS
   세 논문의 "분류기가 뭘 입력받는가"가 전부 다르므로, 하나의 공유
   backbone에서 어느 한쪽을 그대로 반영하면 나머지 둘과 어긋난다 —
   가장 단순한 절충(z에서 바로 선형 분류)을 택했다. 이로 인해
   `SSFAntiForgetting`의 LwF distillation도 SSF 원본처럼 sigmoid 확률
   간 MSE가 아니라 raw logit 간 MSE가 된다(스케일이 다름, 학습에 미치는
   영향의 절대 크기는 다르지만 방향은 동일).
2. **공유 아키텍처의 깊이(은닉층 1개)가 CADE·CND-IDS의 실제 깊이보다 얕음**:
   CADE의 IDS 전용 실행 설정은 은닉층 2개(`64-32-16`), CND-IDS의 실제
   인코더는 은닉층 3개(`nInput→128→256→128→nLatent`,
   `CND_IDS.py:16-23`)다. 이 테스트베드의 공유 `FCLAutoEncoder`는 SSF
   공식(`utils.py:28-36`)을 따라 은닉층 1개다. `hidden_dim`/`latent_dim`의
   **폭**을 SSF 공식에 고정한 것과 같은 논리로, **깊이**도 SSF를 기준으로
   삼았다 — CADE·CND-IDS 양쪽 모두보다 얕지만, 그 둘조차 서로 깊이가
   다르므로(2개 vs 3개) 어느 하나를 "더 정확한 기준"으로 세울 근거가
   없었다.
3. **`CLClient` Step 3~5의 "selected_data만 학습 + memory_manager는 보조
   replay"라는 구조 자체가 SSF 원본과 다름**: SSF 원본은 `x_train_this_epoch`
   라는 하나의 누적 데이터셋에 매 라운드 대표 샘플을 계속 이어붙여
   (`x_train_this_epoch = torch.cat([x_train_this_epoch, ...])`, `utils.py:254`)
   그 전체를 매번 재학습한다 — "새 데이터"와 "리플레이"를 분리하지 않는다.
   이 테스트베드의 `BaseAntiForgetting.compute_loss(model, new_batch,
   replay_batch)` 인터페이스는 SSF뿐 아니라 CADE/CND-IDS/GPM의 서로 다른
   망각방지 메커니즘(distillation, gradient projection, 결합 손실)을 전부
   하나의 계약으로 담아야 해서 나온 이 테스트베드 고유의 통합 설계다 —
   PRD 13절 8단계 자체가 이 설계를 전제하므로, 이번 검토에서 되돌리지
   않았다.

### 발견했지만 의도적으로 그대로 둔 것 (원 논문 자체의 버그로 보임)

**CND-IDS의 LwF 손실이 실제로는 이중 가중치(0.01)로 적용된다**:
`CND_IDS.py:59-70`의 `LwFloss()`는 내부에서 이미
`self.reg_strength * criterion(...)`(reg_strength=0.1)를 계산해 반환하는데,
호출부(`CND_IDS.py:159`)가 다시 `* self.LwF_strength`(0.1)를 곱한다 —
최종 가중치가 `0.1 * 0.1 = 0.01`이지, 설정값이 시사하는 `0.1`이 아니다.
이건 원 논문 저장소 자체의 변수 이름 혼동(reg_strength를 LwFloss 내부에
잘못 넣은 것으로 보임)으로 보이는 버그다. 이 테스트베드의
`CNDIDSAntiForgetting`은 `lambda_cl`(0.1)을 한 번만 곱한다 — 원 저장소의
문자 그대로의 버그를 재현하는 대신, 설정값이 원래 의도한 가중치를 그대로
쓰기로 한다(재현이 아니라 "실제로 의도된 메커니즘"을 재조합한다는 0절
원칙에 부합).

### 검토했지만 문제없음을 확인한 것 (재확인 완료, 변경 없음)

- SSF `kl_max_iter=100`(`utils.py:109,147`), `drift_threshold=0.05`
  (`ssf.py:49`), `lwf_lambda=0.5`(`ssf.py:45`) — 전부 정확히 일치.
- SSF `bs=128`(`ssf.py:48`) — Track A `batch_size=128`과 정확히 일치(우연이
  아니라 SSF가 Track A 아키텍처의 근거 논문이기 때문).
- CADE `t_mad=3.5`, `margin=10.0`, `cae-lambda-1=0.1`, `cae-lr=0.001`,
  `cae-epochs=250`(원본, 이 테스트베드는 continual 재학습 구조 차이로
  의도적으로 5로 축소, 근거는 `cade.yaml` 참고) — 전부 확인.
- CND-IDS `reg_strength=0.1`, `LwF_strength=0.1`, `TripletMarginLoss(margin=2)`,
  `nLatent=30`(참고용, global_hparams가 덮어씀), `train_epochs=20`,
  `batch_size=64`(Track B 전용으로 이번에 반영) — 전부 확인.
- `SSFMemoryManager`/`SPIDERMemoryManager`/`CNDIDSMemoryManager`의
  `max_size=1000`(공통 고정값)은 SSF 원 논문의 실제 메모리 크기 공식
  (`ssf.py:104`, `memory = x_train.shape[0]*(1-percent)` — NSL-KDD 기준
  약 2.5만, UNSW-NB15 기준 약 3.5만)과 다르다. 하지만 `memory_manager`
  슬롯 자체가 SSF/SPIDER/CND-IDS 메커니즘을 **같은 용량 예산 안에서
  비교**하는 축이라, 어느 한쪽만 자기 논문의 실제 용량으로 키우면 그
  비교 축 자체의 공정성이 깨진다(hidden_dim/batch_size를 Track 전체에
  통일한 것과 같은 논리) — 그래서 세 매니저 모두 공통 테스트베드
  기본값(1000)을 그대로 유지하기로 했다.
- `common/metrics.py`, `common/result_schema.py`, `common/compatibility.py`,
  `experiments/leaderboard_builder.py`, `pipeline/common_baselines.py`,
  `components/novelty_baselines/*.py` — 전부 재검토, 문제 없음.
