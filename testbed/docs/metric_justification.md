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
2. **`labeling_budget`을 fixed_count=200으로 바꿨다가 실측 회귀로 다시
   되돌림**: 처음에는 `{mode: fixed_ratio, value: 0.1}`(전체의 10%)이
   "이 테스트베드가 임의로 만든 값"이라 판단해, SSF 원 논문(`ssf.py:23`,
   `--num_labeled_sample` 기본값 200)의 고정 개수 200으로 바꿨다. 그런데
   SSF의 "200"은 이 테스트베드가 재현하지 않기로 이미 결정한 SSF만의 구조
   (매 라운드 계속 불어나는 누적 데이터셋 `x_train_this_epoch`,
   `ssf.py:254`)에 매 라운드 "새로 추가되는 증분"일 뿐이다 — SSF는
   NSL-KDD 기준 ~2.5만 개짜리 누적 풀로 시작해 거기에 200개씩만 보태며
   그 풀 전체를 매번 재학습한다. 이 테스트베드는 그 누적 구조를 안 쓰므로
   (13절 step 3~5, "발견했지만 코드는 바꾸지 않고 문서로만 남긴 것" 3번
   참고), "200"을 라운드당 전체 학습 데이터 수로 그대로 쓰면 SSF의 실제
   규모(누적 풀 전체)와 전혀 다른, 훨씬 작은 양이 된다. 실측 확인:
   `A_dd=none_ss=ssf_mm=none_af=none_as=cade_mad`(NSL-KDD)에서
   fixed_count=200은 f1=0.236(6초), fixed_ratio=0.1은 f1=0.675(33초) —
   `ss=random`은 k=200에서도 별문제 없지만 `SSFSampleSelector`는 라운드
   데이터의 극히 일부(0.1~0.7% 수준)만 뽑아야 하는 상황에서 KL-마스크
   기반 선택이 균등 목표 분포를 맞추려다 오히려 희귀 구간(극단값)에
   편중된 비대표 표본을 뽑는 부작용까지 겹쳐 급격히 나빠졌다. 
   `fixed_ratio: 0.1`로 되돌리고, 이 경위를 `global_hparams.yaml`
   주석에 남겼다.
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

### GPM: 공식 코드와 대조 후 문자 그대로 맞췄다가, 실측 회귀로 다시 되돌림

`components/spider_gpm/gpm_anti_forgetting.py`의 이전 기록은 "GPM 원 논문
Algorithm 1"이라고 표현했지만, 공식 코드(`main_pmnist.py`)와 대조한 결과
실제로는 두 지점(평균 중심화 여부, 기저 개수 `+1` 여유)에서 달랐다. 처음에는
공식 코드에 문자 그대로 맞췄으나(중심화 제거, `+1` 제거), 스모크 테스트에서
`af=gpm` 조합 30개 전수검사를 돌려본 결과 NSL-KDD에서 2개 조합
(`ss=ssf`+`mm=ssf` 또는 `spider`, `as=cade_mad`)이 exp1에서 분류기가 완전히
한 클래스로 퇴화하는 회귀가 실측으로 확인됐다(unsw-nb15는 0/30, 회귀 전에는
nsl-kdd도 0/30이었음). 원인 분리 실험(중심화만 복원/​`+1`만 복원 각각 테스트)
결과 **둘 중 어느 하나만 복원해도** 통과했다 — 즉 이 테스트베드의 얕은
공유 아키텍처(은닉층 1개)에서는 공식 코드의 문자 그대로의 계산(중심화 없음
+ 여유 없음)이 기저를 지나치게 작거나 잘못된 방향(ReLU의 항상-양수 편향
방향)으로 만들어, 분류기 학습에 필요한 그래디언트 성분까지 사영으로
지워버리는 경계 사례를 만든다는 뜻이다.

**최종 결정**: 평균 중심화와 `+1` 여유 둘 다 원래대로 복원했다(공식 코드와
다시 달라짐). 이 프로젝트의 목적(0절 — 논문 재현이 아니라 재조합·비교)상,
"공식 코드와 문자 그대로 같다"보다 "실제로 조합이 붕괴하지 않고 작동한다"를
우선했다 — 재검증 결과 nsl-kdd/unsw-nb15 모두 30/30 통과로 회귀 없음을
확인했다. `_compute_basis()` 주석에 이 경위 전체를 남겨뒀다.

**의도적으로 공식 코드를 따르지 않고 유지하는 나머지 지점**: 공식 코드는
태스크마다 새 기저를 `np.hstack`으로 이어붙이기만 하고, 차원이 넘치면 그냥
앞부분만 잘라낼 뿐 QR 재직교화를 하지 않는다. 이 테스트베드는 5개
experience에 걸쳐 기저가 계속 누적되므로, 서로 다른 태스크의 SVD 결과를
그대로 이어붙이기만 하면(태스크 간 직교성이 보장되지 않아) `basis @
basis.T`가 참된 직교 사영행렬이 아니게 될 수 있다 — 그래서 QR
재직교화만은 공식 코드와 다르게 유지한다.

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

## `normal_reference` 재설계 — 별도 고정 참조 표본 제거 (2026-07-30)

### 문제 제기와 근거

사용자가 "CADE 원 논문을 그대로 쓰면 될 텐데 왜 `normal_reference`라는,
논문에 없는 개념을 따로 만들었나"라고 질문했다. 확인 결과: CADE 원 논문의
median/MAD 계산(`cade/detect.py:151-158`)은 실제로 "그 시점에 라벨이 있는
정상 데이터 전체"로 계산하지만, **CADE 자체에는 이 테스트베드처럼 반복되는
라운드(5개 experience) 개념이 없다** — CADE는 모델을 한 번 학습시키고
`detect()`를 딱 한 번 돌리는 정적(static) 방법이다. "매 라운드 뭘 기준으로
재보정할지"는 CADE 원문에 애초에 답이 없는 질문이었고, 이전의 "experience
0에서 한 번 뽑은 고정 500개 표본을 실험 내내 재사용"하는 방식은 그 답 없는
질문에 대해 이 테스트베드가 임의로 만든 답이었다(그리고 `normal_reference_
size=500`이라는 숫자 자체도 데이터 규모와 무관한 고정값이라, CICIDS2018처럼
큰 데이터셋에서 대표성 문제를 일으켰던 원인이기도 했다).

### 새 설계

별도 고정 표본을 만드는 대신, **이번 라운드에 이미 라벨 예산
(`labeling_budget`)으로 선택된 `selected_data` 중 label=0인 것만 걸러
(`normal_subset`) 정상 참조로 쓴다** (`pipeline/cl_client.py`). 이러면:
- `normal_reference_size`라는 별도 파라미터 자체가 필요 없어진다.
- `labeling_budget` 비율에 자동으로 비례해 데이터 규모와 무관하게 대표성이
  유지된다.
- CADE-MAD 재보정(step 6)뿐 아니라 CND-IDS의 "알려진 정상 참조"
  클러스터링(step 3, `on_experience_start`)도 같은 `normal_subset`을 쓴다
  (이전엔 이 두 곳이 서로 다른 근거 없이 같은 고정 표본을 재사용하고
  있었다).

### 안전장치 (사용자 지시로 시간을 들여 꼼꼼히 검증)

이번 라운드의 라벨 예산 선택분에 정상 라벨 샘플이 하나도 없는 극단적
경우(현실적인 NIDS 데이터에서는 매우 드묾)를 위해:
- `PCAScorer.fit()`에 빈 입력 가드를 새로 추가했다(이전엔 없어서 크래시
  위험이 있었다 — `CADEMADScorer.fit()`은 이미 가드가 있었음).
- `CLClient` 쪽에서 `normal_subset`이 비어있으면 재보정 자체를 건너뛰고
  마지막으로 성공한 `self._s_ref`/scorer 내부 상태를 그대로 쓴다(빈 텐서로
  `score()`를 호출하면 Track A의 `compute_threshold`가 `median()`을 빈
  텐서에 호출해 크래시하는 경로가 있었다 — 발견 즉시 수정).
- CND-IDS의 `on_experience_start`도 `normal_subset`이 비어있으면 건너뛴다
  (이전 라운드의 K-Means 상태 유지, 한 번도 없었으면 기존 폴백대로 전부
  "정상" 간주).
- 인위적으로 모든 라운드의 라벨을 전부 공격(1)으로 강제한 극단 테스트로
  Track A/B 양쪽 다 크래시 없이 동작함을 직접 확인했다.

### 검증 (전부 이번 세션에서 직접 실행, GPM/labeling_budget 사고 이후 사용자
지시로 반영 전 필수)

- `pytest` 전체 13/13 통과(`test_refit_receives_current_encoder_output_
  not_stale_cache`는 `normal_subset` 크기가 라운드마다 달라질 수 있어
  shape 비교를 먼저 하도록 수정).
- 스모크 테스트(93개 조합) × NSL-KDD/UNSW-NB15 = **186/186 전부 통과**,
  실패 0건.
- NSL-KDD 전체 그리드(5-experience, 93개 조합 전부)를 실제로 돌려 F1을
  기존 정상 기준값과 직접 대조 — 예:
  `A_dd=none_ss=random_mm=none_af=none_as=cade_mad` 이전 f1=0.772 → 지금
  f1=0.771, `B_dd=none_ss=random_mm=none_af=cndids_as=pca` 이전 f1=0.894
  → 지금 f1=0.892 — 전부 오차 범위 내로 일치. 93개 조합 F1이 0.57~0.90에서
  정상 분포, 붕괴/이상치 없음.

## CICIDS2018 서브샘플링 제거 — 중복 제거 후 전체 사용 (2026-07-30)

`normal_reference_size`가 데이터 규모에 자동으로 비례하는 방식으로 바뀌면서
(위 절 참고), CICIDS2018을 20만 행으로 서브샘플링했던 두 가지 이유(대표성,
계산 시간) 중 대표성 문제는 사라졌다. 남은 건 순수 계산 시간 트레이드오프
였고, 사용자가 "CICIDS2018을 더 크고 다양한 3번째 데이터셋으로 추가한
애초 취지"를 우선해 서브샘플링 없이 중복 제거 후 전체(약 1200만 행)를 쓰기로
결정했다. `CICIDS2018_SUBSAMPLE_TARGET`과 그 사용처를 `dataset_loader.py`
에서 제거했다 — 중복 제거(`_dedup_rows`)와 Dst Port/Protocol 인코딩은 그대로
유지한다(CADE 원 논문 선례, 위 "CICIDS2018 로더가 어느 논문 선례도 따르지
않던 문제" 절 참고).

**트레이드오프를 사용자에게 명시적으로 고지함**: experience당 행 수가
NSL-KDD/UNSW-NB15보다 80배 이상 많아지므로, Track A 조합 하나당 학습
시간이 (hidden_dim=256/서브샘플링 없음 기준 실측치인) 약 100분 안팎으로
늘어날 수 있고, Track A 조합만 90개라 CICIDS2018 그리드 전체가 완주까지
며칠 단위로 걸릴 수 있다. 로컬(CPU, 합성 데이터)로는 이 스케일을 실측
검증할 수 없어(실제 CICIDS2018 원본이 로컬에 없음), 로더 코드 자체가
서브샘플링 없이도 정상 동작하는지(합성 CSV로 end-to-end 재확인, 중복
제거/인코딩 로직은 그대로)만 확인했다 — 실제 대규모 실행 결과는 GPU
서버에서 사용자가 직접 확인해야 한다.

## CICIDS2018 전체 데이터 첫 실행에서 발견된 버그 2건 수정 (2026-07-30)

서브샘플링을 없애고 실제로 GPU에서 스모크 테스트를 돌린 결과 두 가지
문제가 나타났다.

### 1. GPM `_pending_data` 무한 누적 → CUDA OOM (실제 크래시로 확인, 새로 발견한 버그)

`GPMAntiForgetting.compute_loss()`가 매 미니배치 스텝마다(epoch 루프
전체에 걸쳐) 그 배치를 `_pending_data`에 그냥 계속 append했다. 각 epoch은
`selected_data`를 다시 섞어 도는 것뿐이라, `epochs_per_experience`(Track A
200)번 반복하면 같은 데이터가 200배 중복 누적된다. NSL-KDD/UNSW-NB15
규모(선택 샘플 2500~3500개)에서는 누적해도 수백 MB라 지금까지 한 번도
안 터졌지만, CICIDS2018 전체(선택 샘플 약 19만개, labeling_budget=10%
기준) × 200 epoch에서 약 3860만 행(~11.8GB)까지 쌓여 실제로
`torch.OutOfMemoryError`가 났다(사용자가 GPU에서 실제로 재현). `__init__`에
`activation_sample_size=2000` 상한을 추가해 experience당 최대 그만큼만
모으고 이후로는 더 안 쌓도록 고쳤다 — GPM 논문의 실제 취지(활성화의
"대표 표본"으로 SVD 기저 계산)에도 맞는 방향이다.

**검증**: 로컬에서 CICIDS2018과 같은 자릿수(선택 샘플 5000개, batch=128,
epoch=200 ≈ 7800회 미니배치 호출)로 직접 재현해, 상한 도입 전에는
무한정 커지고 상한 도입 후에는 정확히 `activation_sample_size` 근방에서
멈추는 것을 확인했다. 같은 GPM 조합을 상한 있음/없음으로 A/B 비교한 결과
F1(0.678 vs 0.675)·BWT가 오차 범위 내로 일치 — 상한을 둬도 품질 저하
없음을 확인했다. 이 수정 후 스모크 테스트(93개 조합 × NSL-KDD/UNSW-NB15)
186/186 재확인, pytest 13/13 통과.

### 2. 15.2 체크 공식(min-max range) — 실제로 반영을 미뤘던 것을 지금 반영

CICIDS2018 관련 훨씬 이전 분석에서 이미 "min-max range가 극단치 하나에도
확 벌어져 CADE-MAD처럼 원래 heavy-tailed인 정상 분포를 오탐낸다"는 것을
찾아 percentile 기반으로 바꾸자고 제안했었는데, 그 뒤 CADE 전처리 조사·
서브샘플링 논의로 넘어가면서 실제 코드 반영을 안 했었다. 서브샘플링을
없애고 전체 데이터로 돌리면서 이 오탐이 다시(그리고 더 크게) 나타나
(`smoke_test.py`의 15.2 체크) 이번에 실제로 반영했다 — min-max 대신
1~99 percentile 기반 range로 바꿨다.

**한계를 정직하게 밝힘**: 이 수정이 NSL-KDD/UNSW-NB15에서 회귀를 만들지
않는다는 것(스모크 테스트 186/186, pytest 13/13)은 확인했지만, 로컬에는
실제 CICIDS2018 원본 데이터가 없어 이전에 FAIL 났던 정확히 그 조합·그
스코어 분포로 이 수정이 실제로 통과로 바뀌는지는 직접 재현해 확인하지
못했다 — percentile 기반 range의 효과는 극단치가 전체의 몇 %를 차지하는지에
따라 달라지므로, GPU 서버에서 재실행해 실제로 해결됐는지 확인이 필요하다.

## CICIDS2018 Track B 붕괴 원인 규명 및 K-Means 클러스터 수 스케일링 수정 (2026-08-04)

**증상(실측)**: CICIDS2018 전체 그리드 93개 완료 후 리더보드를 데이터셋별로
분리해서 보니, Track B(CND-IDS) 3개 조합(`mm=none`/`spider`/`cndids`, 셋 다
`af=cndids_as=pca`)이 전부 `f1=0.000`, `roc_auc≈0.49999`(사실상 완전
무작위 수준)이었다. 더 결정적으로, `memory_manager`만 다른 세 조합의
`precision`/`recall`/`roc_auc`가 소수점 15자리까지 완전히 동일했다
(`memory_footprint`는 각각 0/1000/1000으로 버퍼가 실제로 다르게 찼는데도
결과는 같음) — memory_manager 선택이 결과에 전혀 영향을 못 준다는 뜻이라
버그를 의심할 근거가 됐다.

**원인(코드 추적으로 확인)**: `cndids_anti_forgetting.py`의
`_CLUSTER_K_CANDIDATES = [5,10,20,30,50,80]`는 "라운드당 선택 데이터가
수천 건"이라는 가정(주석에 명시) 하에 NSL-KDD/UNSW-NB15 기준으로 정한
고정값이었다. 실측 결과 NSL-KDD 2,519건·UNSW-NB15 3,507건은 그 가정과
맞지만, CICIDS2018 전체 데이터(~1200만 행)에서는 라운드당 선택 데이터가
약 240,000건으로 100배 가까이 벗어난다. K가 최대 80이면 클러스터 하나에
평균 수천 건이 뭉치고, `on_experience_start()`의 "정상 라벨이 하나라도
속한 클러스터 = 정상 클러스터" 판정 기준(CND-IDS 원본
`FeatureExtractors/CND_IDS.py:105-115`과 동일한 메커니즘)이 이 스케일에서는
사실상 거의 모든 클러스터를 "정상"으로 덮어버린다. 그 결과
`_pseudo_labels_for_batch()`가 거의 항상 0(정상)만 반환하고,
`_metric_loss()`의 "같은 pseudo-label 쌍은 거리를 줄여라" 항이 "거의 모든
샘플을 한 점으로 모아라"가 되어 인코더가 붕괴한다 — memory_manager가
공급하는 replay 데이터 내용과 무관하게 매 미니배치의 지배적 손실(metric_loss)
자체가 이미 붕괴를 유도하므로, 세 조합의 결과가 완전히 같아진 것도 설명된다.

**수정**: 고정 K 리스트 대신 `_cluster_k_candidates(n)` 함수로 실제 선택
데이터 크기 `n`에 따라 K를 계산한다. 선형 비례(K ∝ n)는 CICIDS2018
스케일에서 최대 K가 7621까지 커지는데, 로컬 벤치마크(합성 무작위 데이터,
n=240,000·80차원)에서 K=80 하나조차 5분 넘게 끝나지 않는 것을 확인해
계산량 폭발 위험이 크다고 판단, K-means에서 흔히 쓰는 경험적 기준인
`K ∝ sqrt(n)` 스케일링을 채택했다. NSL-KDD 실측값(n_ref=2519)을 기준점으로
비율을 고정해뒀기 때문에 `_cluster_k_candidates(2519)`는 기존
`[5,10,20,30,50,80]`과 정확히 같은 값을 낸다(회귀 없음). 실제 계산 결과:

| 데이터셋 | n(라운드당 선택) | K 후보 |
|---|---|---|
| NSL-KDD | 2,519 | [5, 10, 20, 30, 50, 80] (기존과 동일) |
| UNSW-NB15 | 3,507 | [6, 12, 24, 35, 59, 94] |
| CICIDS2018 | ~240,000 | [49, 98, 195, 293, 488, 781] |

**검증 상태**: NSL-KDD/UNSW-NB15 Track B 3개 조합 × 2개 데이터셋 = 6개
전부 로컬 스모크 테스트 PASS(회귀 없음, NSL-KDD는 K값 자체가 기존과
동일하므로 사실상 항등 확인). **CICIDS2018에서 이 수정이 실제로 붕괴를
해결하는지는 로컬에 원본 데이터가 없어 확인하지 못했다** — 기존
`results/B_*__cicids2018.json` 3개 파일은 삭제했으므로 `grid_runner.py`를
다시 돌리면 자동으로 재계산된다. GPU 서버에서 재실행 후 `roc_auc`가
0.5에서 벗어나는지, `f1`이 0보다 유의미하게 커지는지, 세 memory_manager
조합의 결과가 이제 서로 달라지는지 확인이 필요하다. 또한 K가 커진 만큼
K-means elbow 탐색 자체의 소요 시간이 늘 수 있으므로, CICIDS2018 Track B
조합의 실행 시간이 비정상적으로(예: 몇 시간 이상) 길어지지 않는지도
같이 확인해야 한다.

## Track B가 label_budget 없이 experience 전체를 쓰도록 수정 (2026-08-05)

**문제 제기**: CND-IDS 원 논문(Fuhrman et al., DAC 2025) Algorithm 1은
`labeling_budget` 개념 없이 매 라운드 experience 전체(Xtrain)를 그대로
학습에 쓴다. `CNDIDSAntiForgetting.compute_loss()`도 `selected_labels`를
전혀 쓰지 않는 라벨-프리 설계인데(12.5절), 이 테스트베드는 Track A/B
구분 없이 모든 조합에 `labeling_budget`(10%) 게이트를 강제하고 있었다 —
Track B는 라벨을 애초에 하나도 안 쓰는데, "라벨링 비용 절약"을 명목으로
데이터만 1/10로 줄어드는 상태였다(사용자 문제 제기 후 확인·수정 결정).

**수정**: `cl_client.py` Step 3에서 `track=="B"`일 때 `sample_selector`/
`label_budget` 게이트를 건너뛰고 `new_data` 전체를 그대로 쓰도록 분기
추가. Track A 코드 경로는 들여쓰기만 바뀌고 글자 하나 안 바뀜 — 같은
NSL-KDD Track A 조합을 수정 전/후 코드로 로컬에서 각각 실행해 소수점
10자리까지 동일한 결과(`git stash`로 전/후 코드를 오가며 비교)로 직접
검증했다. `grid_runner.py`의 `labeling_cost`도 Track B는 실제 라벨
소비량(항상 0)을 반영하도록 수정.

## CND-IDS pseudo-labeling을 매 미니배치 sklearn 호출 대신 캐시된 GPU 텐서 연산으로 교체 (2026-08-05)

**문제**: 위 수정으로 Track B가 라운드 전체 데이터를 쓰게 되면서,
`_pseudo_labels_for_batch()`가 매 미니배치마다 `KMeans.predict()`를
CPU(sklearn)에서 새로 호출하는 기존 구조의 비용이 감당 불가능한 수준으로
커졌다(CICIDS2018 기준 라운드당 최대 약 75만 번 호출 추정). 원본
`CND_IDS.py:fit()`을 다시 대조한 결과, **원본은 라운드 시작 시 전체
데이터에 대해 `cluster_labels = self.labeler.fit_transform(x)`를 단
한 번만 호출**하고 그 결과를 라운드 내내 재사용한다 — 매 배치 재계산은
이 테스트베드 구현에서 생긴 비효율이었다(K 스케일링 수정과 label_budget
제거가 겹치며 처음 드러남).

**수정**: `KMeans.predict()`는 정의상 "유클리드 거리로 가장 가까운
클러스터 중심 찾기"이므로, `on_experience_start()`에서 클러스터 중심
(`cluster_centers_`)과 "정상 클러스터 여부" 불리언 벡터를 GPU 텐서로
캐시해두고, `_pseudo_labels_for_batch()`는 `torch.cdist(data,
centers).argmin(dim=1)`로 직접 계산한다. `cl_client.py`는 전혀 건드리지
않았다(Track A는 이 파일 자체를 안 쓰므로 영향 불가능). 클러스터링을
실제로 학습(elbow 탐색 + 최종 fit)하는 부분은 원본과 동일하게
sklearn/CPU를 그대로 쓴다 — 반복 호출되는 predict()만 대체했다.

**검증(둘 다 실측 완료)**:
1. 합성 데이터(5,000 샘플 · 37클러스터)로 sklearn `predict()`와
   `torch.cdist(...).argmin(dim=1)`를 float32/float64 양쪽으로 대조 —
   **5,000개 전부 100% 일치**.
2. `git stash`로 이 수정 전/후 코드를 오가며 실제 Track B 조합
   (`B_dd=none_ss=random_mm=none_af=cndids_as=pca`, NSL-KDD, label_budget
   제거 반영된 상태)을 각각 끝까지 실행 — f1/precision/recall/pr_auc/bwt
   **소수점 10자리까지 완전히 동일**(`f1=0.7254635911` 등).

**남은 확인 사항**: 이 수정 자체는 결과를 안 바꾸므로 기존 Track B
결과와 재계산 결과가 (플랫폼 차이로 인한 부동소수점 오차 범위 내에서)
같아야 하지만, GPU 서버에서 실제로 CICIDS2018 규모까지 실행 시간이
감당 가능한 수준으로 줄었는지는 아직 확인 전이다.

## 공유 백본 hidden_dim/latent_dim을 데이터셋별 SSF 공식으로 전환 (2026-08-11)

**문제 제기**: `FCLAutoEncoder`의 `hidden_dim=128`/`latent_dim=64`가 UNSW-NB15
(196차원)에서 SSF 공식으로 한 번 계산한 값을 3개 데이터셋 전부에 고정
적용하고 있었다. SSF 원 논문 저장소(`ssf.py:51-54,116-120`)는 이 공식
(`nearest_pow2=2**round(log2(input_dim))`; `hidden=nearest_pow2//2`;
`latent=nearest_pow2//4`)을 데이터셋을 로드할 때마다 매번 새로 계산한다 —
이 테스트베드의 기존 주석은 NSL-KDD엔 이미 이 값이 틀렸다는 걸(공식대로면
hidden=64/latent=32) 스스로 인정하고 있었다.

**수정**: `base/models.py`에 `ssf_backbone_dims(input_dim)` 함수 추가.
`experiments/grid_runner.py`/`experiments/smoke_test.py`가 매 데이터셋의
`input_dim`으로 이 함수를 호출해 `hp["hidden_dim"]`/`hp["latent_dim"]`을
그 자리에서 계산하도록 수정 — `global_hparams.yaml`의 고정값 필드는 제거.
`pipeline/cl_client.py`가 이 값을 `merged_component_kwargs`에 그대로 주입하는
기존 구조 덕분에 CADE 사설 encoder를 포함한 모든 "encoder-like" 컴포넌트가
동일 데이터셋에서는 여전히 통일된 크기를 공유한다(추가 코드 불필요, 기존
설계가 이미 이렇게 되어 있었음 — `cl_client.py:48-62` 확인).

- NSL-KDD(121차원): `hidden=64, latent=32` (기존 128/64에서 축소)
- UNSW-NB15(196차원): `hidden=128, latent=64` (기존과 동일값 — 원래 이 값을
  기준으로 절충했던 것이므로 무변화)
- CICIDS2018: 로컬에 원본 데이터가 없어 실제 input_dim을 미리 확인할 수
  없음(2026-07-30 원-핫 인코딩 수정 이후 값 미기록) — GPU 서버에서 실제
  실행 시 로그로 자동 확인됨.

**검증**: `pytest testbed/tests/`(13개 전부 통과, 회귀 없음) +
`python -m testbed.experiments.smoke_test --datasets nsl-kdd,unsw-nb15`로
93개 조합이 새 차원에서 에러 없이 도는지 확인(결과는 아래 별도 기록).

## CND-IDS K-means 후보를 원 논문 고정 리스트로 복귀 + fit 성능 최적화 (2026-08-11)

**재검토 배경**: 2026-08-04에 도입한 sqrt(n) 스케일링(`_cluster_k_candidates`)은
CND-IDS 원문을 다시 정밀 대조하지 않고 내가 고안한 공식이었다. 원문
(`K_Means.py:11-36`)을 재확인한 결과 `[100,300,500,1000,2000]`을 7개 평가
데이터셋 전부에 데이터셋 무관 고정값으로 쓴다 — 스케일링 공식 자체가
없다. 테스트베드가 이 리스트를 `[5,10,20,30,50,80]`으로 축소했던 이유는
당시 Track B에 적용되던 label_budget 서브샘플링(~10%)을 보정하기
위해서였는데(코드 주석에 명시), 그 서브샘플링은 2026-08-05에 이미
제거됐다 — 즉 축소해야 했던 이유 자체가 사라진 상태였다.

**수정**: `_cluster_k_candidates(n)`을 제거하고 `_CLUSTER_K_CANDIDATES =
(100, 300, 500, 1000, 2000)` 고정값으로 복귀. `_elbow_kmeans_fit`에
`fit_sample_size` 매개변수 추가 — 데이터가 이 값보다 크면 무작위
부분표본에서만 elbow 탐색·최종 fit을 수행하고(GPM의 `activation_sample_size`와
동일 패턴), 실제 pseudo-label 배정은 이미 검증된 GPU 벡터화 predict로
전체 데이터에 적용한다. `cluster_fit_sample_size: 10000`을
`component_hparams/cndids.yaml`에 추가.

**성능 실측(합성 데이터, N=200,000·80차원, K=2000)**:
| cap | elbow(6 fits) | 최종 fit | 합계/round | ×5 experience |
|---|---|---|---|---|
| 10,000 | 56.8s | 93.7s | 150.5s | **약 12.5분/콤보** |
| 20,000 | 100.4s | 176.8s | 277.3s | 약 23.1분/콤보 |

cap=10,000 채택 — CICIDS2018 규모(~24만 건/round)에서도 콤보당 K-means
부분이 약 12.5분으로, 기존 7시간+ 정체 대비 압도적으로 개선. NSL-KDD(약
2,519건/round)·UNSW-NB15(약 3,507건/round)는 cap보다 훨씬 작아 부분표본
경로 자체가 발동하지 않는다(항상 전체 데이터로 fit).

**품질 실측 1 — 부분표본 fit이 전체 fit과 동등한가(합성, N=200,000, 실제
코드처럼 normal_subset=label=0 전체(18만 건) 사용)**:
| | fit 시간 | 정상 클러스터 수 | pseudo attack 비율 | 참라벨(10%) 일치율 |
|---|---|---|---|---|
| 전체 fit | 594.5s | 1854/2000 | 0.1000 | **1.0000** |
| 부분표본 fit(cap=10,000) | 38.2s | 1943/2000 | 0.1000 | **1.0000** |

전체 fit vs 부분표본 fit의 pseudo-label 자체도 200,000개 전부 **100% 일치**.
(주의: 최초 시도에서 normal 참조 표본을 임의로 200개만 준 합성 실험은
"K=2000이 정상/공격 비율을 뒤집는다"는 잘못된 결과를 냈었다 — 실제 코드는
`normal_subset = selected_data[selected_labels==0]`로 그 라운드의 정상
데이터 전체를 참조하므로(`cl_client.py:207`) 200개는 비현실적으로 작은
설정이었다. 위 표는 실제 코드와 같은 조건으로 재실행한 결과다.)

**품질 실측 2 — 원 논문 K리스트가 작은 데이터(NSL-KDD)에서도 나은가**
(`git stash`로 신구 코드를 오가며 `B_dd=none_ss=random_mm=cndids_af=cndids_as=pca`
NSL-KDD 콤보를 5-experience 전체 완주, Phase 1의 hidden_dim 수정은 양쪽
다 적용된 상태로 K리스트만 격리):
| | f1 | precision | recall | pr_auc | roc_auc | bwt |
|---|---|---|---|---|---|---|
| 기존 `[5,10,20,30,50,80]` | 0.7255 | 0.5692 | 0.9999 | 0.8039 | 0.7652 | -0.0882 |
| 신규 `[100,300,500,1000,2000]` | **0.8556** | **0.8494** | 0.8620 | **0.8947** | **0.8649** | **-0.0031** |

기존 축소 리스트는 recall≈1.0·precision=0.57 패턴(거의 모든 샘플을
이상치로 판정하는 퇴화 상태)이었고, 원 논문 리스트로 복귀하니 precision이
크게 개선되며 F1/PR-AUC/ROC-AUC/BWT 전부 개선됐다 — CICIDS2018 붕괴만
고치는 게 아니라 NSL-KDD 규모에서도 원 논문 리스트가 더 낫다는 것을 실측
확인. "작은 데이터에 큰 K를 쓰면 나빠질 것"이라는 사전 우려는 기각됐다.

## CADE 사설 encoder 미니배치 학습 도입 + encoder_lr 보정 (2026-08-11)

**발견**: `components/cade/cade_drift_detector.py`의 `fit()`이
`encoder_epochs`(=5)회 반복하며 매번 `train_step`에 selected_data 전체를
미니배치 분할 없이 한 번에 넘기고 있었다 — 즉 "5 epoch"이 아니라 총 5회의
그래디언트 업데이트에 불과했다. CADE 원문(`run_drebin_cade.sh`/
`run_ids_cade.sh`)을 재대조한 결과 Drebin(`--cae-batch-size 64`)/
IDS2018(`--cae-batch-size 512`) 모두 배치 크기만 다를 뿐 미니배치 자체는
항상 쓴다. `encoder_lr=0.001`도 CADE 원문의 "실제로 쓰이지 않는 argparse
기본값"을 인용한 것이었다 — 실제 CADE 실험은 두 데이터셋 모두
`--cae-lr 0.0001`로 override해서 돈다.

**수정**: `fit()`에 표준 미니배치 학습(매 epoch마다 셔플 후 batch_size 단위
분할) 추가. batch_size는 CADE 원문의 데이터셋별 값(64/512)이 이 테스트베드의
3개 데이터셋과 대응되지 않으므로(NSL-KDD·UNSW-NB15는 CADE가 평가하지 않은
데이터셋) 새로 추측하지 않고 `global_hparams.batch_size`(Track A 공유값,
`pipeline/cl_client.py`가 `merged_component_kwargs`에 주입)를 재사용.
`component_hparams/cade.yaml`의 `encoder_lr`을 0.0001로 수정.

**검증(`git stash`로 신구 코드를 오가며 pure-CADE 콤보
`A_dd=cade_ss=random_mm=none_af=none_as=cade_mad` NSL-KDD 5-experience
전체 완주)**:
| | f1 | precision | recall | pr_auc | roc_auc | bwt |
|---|---|---|---|---|---|---|
| 기존(통짜 5스텝, lr=0.001) | 0.7297 | 0.9554 | 0.5903 | 0.9090 | 0.8693 | -0.0261 |
| 신규(미니배치, lr=0.0001) | **0.7866** | 0.9573 | **0.6676** | 0.9082 | 0.8429 | **0.0125** |

두 경우 모두 붕괴 없이 정상 범위(precision 0.95대)로 동작하며, F1/recall/BWT는
신규 쪽이 뚜렷이 낫고 PR-AUC는 거의 동일, ROC-AUC는 소폭(0.026) 하락 —
전반적으로 순개선이며 최소한 퇴보는 아님을 확인.

## 분류기 헤드: classifier(z) 유지, classifier(decoder(z))는 채택하지 않음 (2026-08-11)

**문제 제기**: `base/models.py`의 `FCLAutoEncoder.classifier`가 SSF 원문
(`utils.py:59-61`, `classifier(decoder(z))`)과 달리 잠재 표현 `z`에서 바로
분류한다 — SSF 방법론을 온전히 쓰지 못하는 것 아니냐는 문제 제기.

**검토**: `classifier(decoder(z))`로 바꾸는 것 자체의 기술적 blast radius는
작다(`base/models.py` 생성자·forward 및 `pipeline/common_baselines.py`의
`NoAnomalyScorer.score()` 두 곳 — GPM은 `nn.Linear`를 이름으로 제네릭
순회하므로 자동 적응, CADE/CND-IDS는 `z`/`x_hat`을 classifier를 거치지
않고 직접 소비하므로 무관). 하지만 이 변경을 채택할 근거의 성격이 같은
세션에서 고친 다른 두 버그(공유 백본 hidden_dim/latent_dim, CADE 미니배치
학습)와 근본적으로 다르다:

- hidden_dim/latent_dim은 이 테스트베드 스스로 "NSL-KDD엔 이미 틀렸다"고
  인정하고 있던 자기 시인 버그였고, CADE도 "폭은 데이터셋마다 달라야
  한다"는 원칙엔 동의해 근거가 두 논문에서 나왔다. CADE 미니배치 학습
  누락도 원문과 명백히 다르게 구현된, 마찬가지로 자기 시인급 버그였다.
  두 경우 모두 "대안이 없어 채택 외에 선택지가 없는" 상황이었다.
- `classifier(decoder(z))`는 그런 종류의 버그가 아니다. `classifier(z)`는
  이미 "가장 단순한 절충"으로 문서화된, 그 자체로 정당한 설계다(SSF는
  `decoder(z)`를 쓰지만, CADE는 분류기와 오토인코더가 아예 분리된 별도
  모델이라 이 축에 의견이 없고, CND-IDS는 지도 분류기 헤드 자체가 없어
  역시 의견이 없다). 즉 지금 있는 근거는 "SSF가 이렇게 한다" 하나뿐이다.

**결정: 채택하지 않는다.** 실측(A/B) 없이 이 상태로 Track A 90개 조합
전부가 공유하는 백본을 SSF 하나의 설계로 바꾸면, 근거 없이 특정 논문을
우대하는 것이 되어 이 테스트베드의 목적(0절 — 재조합·비교, 특정 논문
재현이 아님)에 어긋난다. "고칠 수 없어서"가 아니라 "지금 가진 근거로는
공유 그리드에 강제할 명분이 없어서" 그대로 둔다 — 이 판단은 hidden_dim/
CADE 미니배치 건과 마찬가지로 실증적 근거를 먼저 확인한 뒤 내린 것이지,
반대로 근거 없이 바꾸지 않기로 한 것도 아니다(둘 다 실측·원문 대조를
거쳤다는 점은 동일하다 — 결과만 "바꾼다"/"안 바꾼다"로 갈렸을 뿐).

## Experience 분할을 class-incremental 구조로 전환 (2026-08-12)

**문제 제기**: `n_experiences=5`로 나누는 것 자체는 CND-IDS 원 논문 근거였지만,
실제 분할 방식(`data/dataset_loader.py`)은 데이터 전체(정상+공격 구분 없이)를
고정 seed로 무작위로 섞은 뒤 그냥 5등분하는 것뿐이었다 — experience 사이에
의도된 분포 차이가 전혀 없어, 지속학습이 방어해야 할 진짜 "새로운 것이
등장"하는 상황 자체가 시나리오에 없었다.

**검토했다가 기각한 대안**: SSF 원 논문 방식(`ssf.py`의 가변 길이 스트리밍
윈도우 + drift 감지)도 검토했으나, 코드를 끝까지 재확인한 결과 SSF도 drift
감지 여부와 무관하게 매 라운드 무조건 재학습함을 확인했다(`if drift: ...
else: ...` 양쪽 분기 다 `for epoch in range(epoch_1): ...` 재학습 루프를
포함 — 이전에 "drift 감지 시에만 재학습"이라고 판단했던 것은 오독이었다,
정정함). 게다가 SSF는 데이터 크기에 따라 라운드 수가 달라지고(고정
n_experiences 구조와 안 맞음), "대표 표본 선택+망각" 메커니즘 자체가 SSF
고유 알고리즘이라 공유 시나리오로 채택하면 SSF 방법론을 전체 그리드에
강제하는 셈이 된다(0절 위반). 또한 "UNSW-NB15 원본 파일 순서가 우연히
정상→공격으로 깨끗하게 갈린다"는 방식도 검토했으나, 특정 논문 근거가 없는
발견물이라 기각했다(주관 배제 원칙).

**채택한 방식**: CND-IDS 원 논문의 실제 experience 분할 메커니즘
(`CND-IDS/utils.py:275-299`, `create_split_experiences`)을 그대로 이식했다
— `n_experiences=5`라는 숫자를 원래 의미 있게 만드는 메커니즘이며, 데이터를
"언제 등장시킬지"의 시나리오 설계 문제일 뿐 특정 컴포넌트 알고리즘이
아니라서 93개 조합 전부에 공평하다. 정상 트래픽은 무작위로 고르게 5개
experience에 분배하고, 공격은 세부 카테고리별로 묶어 라운드로빈으로
experience에 배정한다(`class_order[i % 5].append(category)`, category는
정렬된 순서로 순회) — 한 experience는 자신에게 배정된 공격 유형만 본다.
train/test 양쪽에 같은 class_order를 재사용해 experience i의 test가
experience i의 train과 같은 카테고리를 반영하도록 했다(CND-IDS 원문과 동일).
`data/dataset_loader.py`의 `_class_incremental_split`이 핵심 구현이고,
기존 `_chunk_shuffled`/`_chunk_by_row_order`/`_shuffle`은 삭제했다.

**데이터셋별 카테고리 확보**:
- NSL-KDD: `labels5`(정상/DoS/Probe/R2L/U2R) — 기존에 버려지던 컬럼을
  살렸다. `_read_nslkdd_features`가 category로 반환.
- CICIDS2018: 이진화 전 원본 `Label` 문자열(예: 'DDoS attacks-LOIC-HTTP')을
  보존 — 기존엔 이진화만 하고 버려졌다. `_read_cicids2018_features`가
  category로 반환.
- UNSW-NB15: 지금 쓰는 SSF 전처리본엔 이진 `label`만 있고 다중클래스
  `attack_cat`이 없다. 공식 `UNSW_NB15_training-set.csv`(175,341행)/
  `UNSW_NB15_testing-set.csv`(82,332행)의 행 수가 SSF 전처리본과 정확히
  일치함을 확인해(같은 원본에서 나온 것으로 판단), 250만행짜리 원본을
  재전처리하지 않고 공식 파일의 `attack_cat`만 같은 행 위치에서 가져와
  결합한다(`_load_unsw_attack_cat`). **행 정렬은 가정하지 않고 label
  컬럼 완전 일치로 실측 검증한 뒤에만 신뢰**하도록 구현했다(불일치 시
  즉시 예외). 공식 원본은 SharePoint 호스팅이라 CICIDS2018과 달리 자동
  다운로드는 지원하지 않고, 사용자가 https://research.unsw.edu.au/projects/unsw-nb15-dataset
  에서 받아 `UNSW-NB15-raw/`에 수동 배치해야 한다.

**검증(전부 실측 완료)**:
1. NSL-KDD 실제 실행 결과가 사전 계산한 표와 완전히 일치: exp0=DoS(45,927),
   exp1=Probe(11,656), exp2=R2L(995), exp3=U2R(52), exp4=공격 없음(카테고리
   4개 vs experience 5개라 자연 발생, 인위적으로 완화하지 않음).
2. 합성 데이터(다중 카테고리)로 `_class_incremental_split` 단독 검증 —
   전체 행 수 보존, class_order 배정 정상.
3. `pytest testbed/tests/` 13/13 통과(회귀 없음).
4. `python -m testbed.experiments.smoke_test --datasets nsl-kdd` — **93/93
   조합 전부 통과**.
   **2026-08-26 정정(문서 부정확성으로 확인)**: 이 문장이 "U2R 52건짜리
   희소 라운드, 공격 0건짜리 라운드 포함"이라고 적어 마치 스모크 테스트가
   그 라운드들까지 실제로 검사한 것처럼 서술했는데, 사실이 아니었다 —
   당시 `SMOKE_N_EXPERIENCES=2`(하드코딩)였으므로 이 명령은 5개
   experience 중 앞 2개(exp0=DoS, exp1=Probe)만 검사했고, U2R(exp3)과
   공격 0건 라운드(exp4)는 한 번도 검사한 적이 없다. 이 잘못된 "이미
   확인됨" 문구가 4개 논문 컴포넌트 전수 재감사(2026-08-26) 전까지
   CND-IDS의 pseudo-label 붕괴(정확히 그 안 본 라운드들에서 발생)를
   못 잡은 원인 중 하나로 보인다 — 자세한 경위는 아래 "4개 논문 컴포넌트
   전수 재감사" 절 참고. `SMOKE_N_EXPERIENCES`는 이제 전체 experience를
   보도록 고쳤다.
5. Pure-CADE 조합(`A_dd=cade_ss=random_mm=none_af=none_as=cade_mad`,
   `af=none`이라 망각방지 장치 없음) NSL-KDD 5-experience 전체 실행 —
   **BWT가 기존 i.i.d 분할 대비 +0.013 → -0.177로 뚜렷하게 악화**. 이는
   버그가 아니라 의도한 신호다: 예전 무작위 분할에서는 방어할 진짜 분포
   변화가 없어 BWT가 0 근처로 나온 것이고(그럴듯해 보이지만 무의미한
   비-망각), 이제 진짜 class-incremental 구조에서 처음으로 진짜
   catastrophic forgetting이 드러난 것이다.

**남은 작업**: UNSW-NB15는 사용자가 공식 CSV를 배치해야 로컬 검증 가능.
CICIDS2018은 로컬에 원본이 없어 실제 카테고리 분포·smoke 통과 여부는 GPU
서버에서 최종 확인 필요. 이 변경은 93×3 전체의 데이터를 바꾸므로 이미
필요했던 하이퍼파라미터 재실행(백본 크기/K-means/CADE 미니배치)과 묶어서
한 번에 전체 그리드를 재실행해야 한다.

## 3개 병렬 에이전트 전수 원문 대조 감사 + 구조적 충실도 보강 (2026-08-12)

**배경**: class-incremental 분할 전환 후 "지금까지 수십 번 검토했는데도 왜
이런 문제가 계속 나오냐"는 문제 제기에 따라, SSF/CND-IDS/CADE·SPIDER·
파이프라인 3개 영역을 각각 별도 에이전트가 원문 코드와 줄 단위로 대조하는
전수 감사를 수행했다. 기존 검토들은 대부분 "공식/하이퍼파라미터가 일치하는가"
수준이었는데, 이번 감사는 배치/pair 구성 방식, 연산 순서, drift 등 조건에
따른 상태 변화 방향, 손실항 누락 여부, 차용한 메커니즘의 실제 출처까지
추적했다(이 관점을 앞으로의 감사 표준으로 삼는다 — memory
`feedback-structural-fidelity-audits` 참고). 결과: 17개 대조 항목 중 11개에서
이전에 문서화되지 않은 실제 불일치를 발견했다.

### 1. 귀속/문서 정정만 (동작 변경 없음)

1. **`pipeline/cl_client.py` 모듈 docstring**: "이 순서는 각 논문에서 그대로
   도출된 것"이라는 주장이 SSF에 대해서는 사실과 반대였다 — SSF 원문
   (`ssf.py:236-291`)은 대표 표본 재선택으로 메모리를 먼저 갱신한 뒤 그
   갱신된 세트로 학습하는데, 이 파이프라인은 학습(Step4) 후 메모리 갱신
   (Step5) 순서다. **코드는 바꾸지 않았다** — Step4가 매 미니배치
   `get_replay_batch()`로 "이전" 버퍼를 읽는 공유 리플레이 계약 때문에,
   순서를 뒤집으면 그 라운드의 `selected_data`가 학습 전에 먼저 버퍼로
   들어가 버려 같은 라운드 안에서 자기 자신을 리플레이하는 결과가 된다
   (SPIDER/CND-IDS 방향 메모리 매니저에서 치명적 — get_replay_batch를
   실제로 소비하는 유일한 두 컴포넌트). docstring에서 "논문에서 도출됐다"는
   과장된 인과 주장만 제거했다.
2. **`components/cndids/cndids_memory_manager.py` docstring**: "CND-IDS
   원문 근거"라는 프레이밍을 정정했다. CND-IDS의 실제 제안 방법
   (`CND_IDS.py`, 196줄)에는 메모리/리플레이가 전혀 없다 — 이 컴포넌트
   (max_size=1000 포함)는 실제로는 같은 저장소의 `CFE.py`(ADCN 베이스라인용
   별도 피처 추출기, CND-IDS와 무관)의 `Memory` 클래스와 닮아 있을 뿐이다.
3. **`common/metrics.py`의 `bwt()`**: "CND-IDS 원 논문 공식 그대로"라는
   표현을 정정했다. `AutonomousDCN/ADCNmainloop.py:418`(파일:줄 인용 자체는
   원래도 정확했다)은 CND-IDS 저자 자신의 제안 방법이 아니라 같은 저장소의
   ADCN 비교 베이스라인 평가 코드다. 공식 자체는 표준 continual-learning
   문헌의 BWT 정의와 일치하므로 구현이 틀린 것은 아니고, 귀속 표현만
   정정했다.

### 2. 실제로 반영한 것 (코드 변경, 전부 A/B 실측 후 반영 — NSL-KDD 기준)

4. **CADE class-aware pairing 추가** (`contrastive_ae.py`
   `build_paired_batches()`, `cade_drift_detector.py`). 원문(`cade/data.py:
   268-345`)은 배치 구성 시 `similar_ratio`(기본 0.25)로 same/different-class
   쌍을 강제하는데, 이 부분이 아예 없어 순수 무작위 셔플에 의존하고
   있었다 — class-incremental 분할이 만드는 극단 불균형 라운드(U2R 52건
   단독 등)와 결합하면 배치에 dissimilar 쌍이 없어 margin loss가 죽을 위험이
   실제로 있었다. (label, similar 여부) 조합별 그룹화 + `torch.randint`
   벡터화로 이식(원문의 이중 for-loop `np.random.choice` 대신). 클래스가
   하나뿐인 라운드는 dissimilar 풀을 similar 풀로 대체(원문 미정의 상황에
   대한 안전한 폴백).
   **A/B (`A_dd=cade_ss=random_mm=none_af=none_as=cade_mad`)**: f1
   0.4984→0.6482(+30%), precision 0.796→0.795(유지), recall 0.363→0.547,
   bwt -0.177→-0.140.
5. **SSF InfoNCE 재구성-대조 손실항 추가** (`components/ssf/ssf_infonce.py`
   신규, `ssf_anti_forgetting.py`). SSF의 실제 task_loss는 BCE 하나가
   아니라 `weighted_con_loss.mean() + weighted_classification_loss.mean()`
   (`ssf.py:310-318`)로, InfoNCE 기반 재구성-대조 손실(`utils.py:458-492`,
   디코더 출력 `recon_vec`에 적용, `tem=0.02`)이 통째로 빠져 있었다.
   **A/B (`A_dd=none_ss=ssf_mm=none_af=lwf_ssf_as=cade_mad`)**: f1
   0.6680→0.7107(+6%), bwt -0.0161→-0.0005(거의 완전한 망각 방지).
   **new_sample_weight=100은 채택하지 않았다** — SSF에서 100은 "누적 풀
   전체(~2.5만) 대비 신규 표본(~200개)"라는 극단적 비율(~1:125)을 보정한
   값인데, 이 테스트베드는 new_batch/replay_batch 크기가 비슷해(~1:1)
   같은 100을 곱하면 gradient 기여도가 ~99:1로 replay가 사실상 무력화된다.
   실측: weight=100 적용 시 f1 0.7107→0.5655, bwt -0.0005→-0.1341로 급격히
   악화. `labeling_budget`(global_hparams.yaml)과 같은 종류의 함정이라 같은
   방식으로 처리 — 배치 단위 가중치라는 **구조**는 그대로 가져오되 값은
   1.0(신규/과거 동등 취급)으로 이 테스트베드의 new:old 비율에 맞게
   재보정했다. 자세한 근거는 `ssf.yaml`의 `new_sample_weight` 주석 참고.
6. **SSF 메모리 버퍼 drift 반응 방향 수정** (`ssf_memory_manager.py`). SSF
   원문(`utils.py:259-388`)은 drift 시에도 버퍼를 목표 크기로 유지한 채
   회전율만 높이는데("보충", not "축소"), 이전 구현은 유지 개수 자체를
   `max_size*drift_retention_ratio`로 줄였다. 이제 drift 시 **기존 버퍼
   쪽만** 대표성 상위 일부로 먼저 추리고, 거기에 이번 라운드 신규 표본
   전체를 합친 뒤 평시와 동일하게 대표성 상위 max_size개를 최종 유지 —
   버퍼 총량은 항상 max_size로 수렴하되 drift 시 과거 표본이 더 많이
   교체된다.
   **A/B (`A_dd=ssf_ss=ssf_mm=ssf_af=none_as=cade_mad`)**: f1
   0.4551→0.5122(+12.5%), precision 0.495→0.493(유지), recall
   0.421→0.533, bwt -0.1103→-0.0725.
7. **CND-IDS multi-teacher LwF 누적** (`cndids_anti_forgetting.py`). 원문
   (`CND_IDS.py:42,54-69,195`)은 `self.old_models`에 매 experience 종료 시
   모델 스냅샷을 계속 추가만 하고(절대 비우지 않음), `LwFloss()`가 누적된
   **모든** 과거 모델과 개별 MSE를 구해 합산한다 — 이전 구현은 직전 1개
   teacher만 유지·매번 덮어써서 라운드가 진행될수록 원문 대비 망각방지
   압력이 약해지고 있었다. 가중치 적용 방식도 다시 대조해 함께 고쳤다:
   원문은 항목별로 `lambda_r`을 곱하고 그 합계에 다시 `lambda_cl`을 곱해
   실효 가중치가 `lambda_r*lambda_cl`인데(`:66,159,180`), 이전 구현은
   `lambda_cl`만 곱하고 있었다(두 값이 우연히 같은 0.1이라 결과가 크게
   갈리진 않았지만 원문과 어긋났다).
   **A/B (`B_dd=none_ss=random_mm=none_af=cndids_as=pca`, Track B)**: f1
   0.8774→0.8757(거의 동일), precision 0.826→0.873(+5.6%), recall
   0.935→0.879(-6.0%), pr_auc 0.876→0.897, **bwt -0.0099→+0.1341**(약한
   망각에서 양의 backward transfer로 — 누적 teacher의 정규화 압력이 과거
   experience 표현을 오히려 개선하는 방향으로 작용, F1 손실 없이 얻은
   개선).

전부 `pytest testbed/tests/`(13/13) 통과 확인.

### 3. 의도적으로 채택하지 않은 것 (문서화만)

- **CND-IDS의 optimizer 매 experience 재생성**: SSF 원문(`ssf.py:122,
  273,291`)을 다시 확인한 결과, optimizer를 스트리밍 전체에서 **한 번만**
  생성해 끝까지 재사용한다(재설정 안 함) — CND-IDS(`CND_IDS.py:104`)와
  정면으로 다른 관행이다. 두 논문이 서로 다르게 하는 이상 어느 한쪽을
  공유 클라이언트(`cl_client.py`, 모든 조합이 쓰는 단일 optimizer)에
  강제하면 classifier(decoder(z))와 같은 이유로 0절 위반이다. 현재(유지)가
  이미 SSF와 일치하므로 유지.
- **CND-IDS의 validation split + best-checkpoint 선택**: `CND_IDS.py:
  131-192`가 하는 이 모델 선택 휴리스틱은 SSF/CADE/SPIDER 어디에도 없는
  CND-IDS 고유 설계다. cndids_af에만 특별 적용하면 4개 컴포넌트의 학습
  루프가 비대칭해지고, 전체에 적용하면 나머지 3개 논문이 요구한 적 없는
  로직을 강제하게 된다 — 채택하지 않음.

### 4. 우선순위 낮음 (보류, 재평가 대상)

- SSF LwF의 무조건 적용(원문은 non-drift 라운드에서만) — 다른 논문과
  상충하지 않는 순수 SSF 자체 충실도 문제라 고쳐도 되지만 영향 범위가
  좁다(`dd≠none & af=lwf_ssf`).
- SSF SampleSelector/MemoryManager의 목표분포·representativeness 산정
  단순화(균일분포 vs 원문의 KL-divergence dual-mask) — 원문의 최적화
  절차 전체를 포팅해야 하는 큰 작업 대비 실익 불확실.

### 5. 조치 불필요 (재확인 완료)

- SPIDER 버퍼 크기(`max_size=1000`) — SPIDER 자체 논문 수치는 아니지만
  이미 "SSF/SPIDER/CND-IDS를 같은 용량 예산에서 비교"라는 근거가 있다
  (위 "검토했지만 문제없음을 확인한 것" 절 참고). SPIDER 코드 자체가
  로컬에 없어 원천적으로 그 이상 검증 불가능.
- SSFDriftDetector 최소 윈도우 크기 가드 누락 — 실제 크래시 사례 없는
  경미한 엣지케이스.
- CND-IDS PCAScorer 죽은 코드 — 버그 아님.

## 테스트베드 구조적 완성도 감사 (2026-08-14)

**배경**: "논문과 일치하는가"(위 절들)는 충분히 봤지만, "테스트베드 자체가
'가장 뛰어난 지속학습 조합을 찾는다'는 목적에 맞게 엔지니어링적으로
완성도 있게 짜였는가"는 별도로 다시 봐야 한다는 문제 제기에 따라, 데이터
파이프라인/시간흐름, 학습 루프·그래디언트 흐름, 평가·결과·리더보드
파이프라인 3개 영역을 각각 별도 에이전트가 다시 감사했다. Track A/B 그리드
분리 자체(`common/compatibility.py`)는 사용자가 그대로 두라고 확정해 감사
대상에서 제외했다. 총 13개 항목을 발견했고, 안전하고 기계적인 6개는 바로
반영, 실측 검증이 필요했던 GPM 1건은 A/B 두 차례로 확정, 나머지는 사용자
판단이 필요해 보류했다.

### 1. 바로 반영한 것 (기계적, 위험도 낮음)

1. **낡은 `results/` 캐시 270개**: `grid_runner.py`의 결과 캐싱(`if
   os.path.exists(out_path): continue`)이 코드 버전을 검사하지 않아,
   오늘 CADE/SSF/CND-IDS 컴포넌트 4건과 class-incremental 분할을 고친
   뒤에도 그 이전(2026-07-31~08-04) 코드로 계산된 결과 270개가 재계산
   없이 그대로 남아있었다 — 지금 그리드를 돌리면 낡은 결과가 "최신"으로
   반영될 뻔했다. `testbed/archive/2026-08-14_pre-structural-audit/`에
   보존(이미 git에 커밋돼 있어 git으로도 복원 가능, 이 폴더는 편의용
   사본)한 뒤 `testbed/results/`를 비웠다. 재발 방지로 `grid_runner.py`에
   `compute_code_version()`(components/base/pipeline/dataset_loader.py
   내용을 해시)을 추가해, 캐시된 결과의 `code_version`이 지금 코드와
   다르면 스킵하지 않고 재계산하도록 했다(`result_schema.py`에
   `code_version` 필드 추가).
2. **`_class_incremental_split`의 class_order 미포함 카테고리 조용한
   드롭**: train에서 계산한 `class_order`를 test에 재사용하는데, test
   파일에 train에 없던 공격 category가 있으면 그 표본이 에러도 경고도
   없이 어떤 experience에도 배정되지 못하고 사라지는 구조였다. NSL-KDD는
   실측으로 문제없음을 확인했지만(대분류 5종 기준 train/test category
   집합 일치) UNSW-NB15는 로컬에 원본이 없어 검증 못 했다 — 이 코드베이스의
   "추측으로 조용히 진행하지 않는다" 원칙(예: `_load_unsw_attack_cat`의
   행 정렬 검증)에 맞춰, class_order가 다루지 못하는 category가 있으면
   즉시 `ValueError`를 던지도록 안전장치를 추가했다.
3. **`best_f1_reference=0.0`의 오독 가능성**: Track B는 이 참고 지표
   자체를 계산하지 않는데 `0.0`을 그대로 넣어서, CSV를 열어보는 사람이
   "이 조합이 도달 가능한 최선의 F1이 0"이라고 오독할 위험이 있었다 —
   "계산 안 함"을 `float('nan')`으로 명시했다.
4. **`memory_footprint`가 마지막 라운드 스냅샷 하나뿐**: `SPIDERMemoryManager`
   처럼 매 라운드 버퍼를 그 라운드 selected_data 크기로 통째로 교체하는
   memory_manager는, class-incremental 분할이 만드는 라운드별 데이터量
   변동(마지막 experience가 우연히 희소 카테고리면 버퍼가 작게 찍힘)
   때문에 마지막 스냅샷만으론 오해를 살 수 있었다 — 라운드별 크기를
   전부 기록해 `memory_footprint_peak`/`memory_footprint_avg`를 추가로
   남기도록 `grid_runner.py`/`result_schema.py`를 고쳤다.
5. **smoke_test가 "손실이 실제로 줄어드는 방향으로 갔는가"를 검증하지
   않음**: 기존 15.1a(파라미터 변화량)/15.1b(optimizer step 횟수)는
   "학습 루프가 설계대로 실행됐는가"만 보고, 손실 발산 여부는 15.2/15.3의
   간접적인 NaN 전파로만(진단 메시지가 근본 원인을 가리키지 못하는 채로)
   걸러졌다. `cl_client.py`의 `run_experience()`가 이제 experience당
   `first_epoch_avg_loss`/`last_epoch_avg_loss`를 함께 반환하고,
   `smoke_test.py`에 15.1d 게이트(finite 여부 + 첫 epoch 대비 10배+
   증가 시 발산 의심 실패 처리)를 추가했다.

### 2. GPM 기저(basis) 풀랭크 문제 — 실측 검증 후 반영

GPM(`components/spider_gpm/gpm_anti_forgetting.py`)의 기저 누적에 크기
상한이 전혀 없다는 게 감사에서 발견됐다. 실측(NSL-KDD, `af=gpm`, 5
experience)으로 직접 재현했다: `encoder.0`(121차원)이 exp4에서 정확히
121/121(풀랭크)에 도달해 그 레이어의 그래디언트가 마지막 라운드에
완전히 0이 됐다(`project_gradients()`의 `grad - grad@basis@basis.T`가
기저가 전체 공간을 덮으면 항상 grad 전체를 지워버리기 때문). `encoder.2`도
93.75%까지 갔다.

원인으로 GPM 원 논문 Algorithm 2의 residual projection 단계(SVD 전에
이미 누적된 기저 방향을 먼저 제거)가 빠져 있다고 보고 `_compute_basis()`에
추가했으나, A/B 실측(같은 조합)으로 **f1 0.7006→0.6440, bwt
-0.108→-0.142로 오히려 악화**됨을 확인해 되돌렸다 — 이미 잘 커버된
작은 레이어(`classifier`/`decoder.0`, latent_dim=32)일수록 residual을
빼고 남은 에너지가 작고 흩어져 있어 오히려 그 레이어 차원 대부분이
"새로 필요한 성분"으로 채택되는 역효과가 났다(기존엔 37.5%만 고정됐는데
residual 적용 후엔 87.5%까지 고정 — 가장 직접적으로 예측에 관여하는
레이어일수록 더 심하게 다쳤다). 이 파일의 2026-07-30 기록(공식 코드를
문자 그대로 맞췄다가 실측 회귀로 되돌린 사례)과 같은 종류의 함정이라
같은 원칙(실측 우선)으로 처리했다.

대신 `max_basis_ratio=0.9`(신규, 원 논문에 없는 이 테스트베드 고유의
안전장치)를 추가해, 기저가 ambient dimension의 90%를 넘지 못하게 상한만
뒀다(넘칠 QR 열은 오래된 방향부터 유지하고 최근 방향부터 잘림 — "오래된
태스크를 더 우선 보호"라는 GPM의 취지와 부합). A/B 실측 결과 이 cap만
적용한 버전은 기존(무제한) 대비 f1 0.7006→0.7012(사실상 동일),
bwt -0.1084→-0.0990(약간 개선)로, 풀랭크로 인한 치명적 실패만 막고
나머지는 그대로 유지하는 것을 확인했다.

### 3. 사용자 판단이 필요한 것 (아직 반영하지 않음)

- **MinMaxScaler가 미래 experience 정보를 누수**: `dataset_loader.py`의
  두 프로토콜(`preserve_official_split` True/False) 모두, 5개
  experience로 나누기 **전에** 전체(또는 train 파일 전체)에 스케일러를
  fit한다 — experience 0을 정규화할 때 이미 experience 4의 값 범위까지
  반영된 스케일러를 쓴다는 뜻이다. CICIDS2018(train+test 병합분까지
  포함)이 더 심하고 NSL-KDD/UNSW-NB15는 정도가 약하지만 같은 종류의
  문제다. 제안된 수정: experience로 먼저 나눈 뒤 `MinMaxScaler.partial_fit()`
  으로 그 시점까지의 experience만 누적 반영 — 후속 experience 값이
  `[0,1]` 밖으로 나가는 것은 버그가 아니라 "분포가 실제로 이동했다"는
  의미 있는 신호로 그대로 둬야 한다. 다만 이건 93×3 전체의 입력 데이터를
  바꾸는 변경이라(이미 계산된 것과 재계산될 것을 다시 갈라놓음), A/B
  실측과 함께 사용자 확인 후 반영 여부를 정하기로 했다.
- **리더보드 F1 정렬이 BWT(망각)를 전혀 반영하지 않음**: "왜 BWT를
  정렬 기준에서 뺐는가"에 대한 근거가 코드/문서 어디에도 없다 — 의도된
  설계라는 기록이 없는 채로 지속학습 테스트베드의 핵심 지표(망각 방지)가
  "최선의 조합" 판정에서 완전히 빠져 있다. 대안(F1 우선 + BWT 임계치
  이하 경고 표시, F1 순위와 BWT-패널티 순위 병기 등)은 있지만 어느 쪽을
  택할지는 사용자가 정할 문제로 남긴다.

### 4. 사용자 질문에 대한 정직한 답

- **데이터가 시간 흐름에 따라 배치 단위로 잘 입력되는가**: experience
  순서·미니배치 셔플·train/test 분리는 감사로 문제없음을 확인했다(위
  스케일러 항목 제외). 다만 "시간 흐름을 반영한 배치"라는 서사는
  **공격 종류의 등장 순서에만** 있다 — 정상 트래픽의 experience 배분은
  순수 무작위 균등분배라 시간 흐름과 무관하다(`_class_incremental_split`,
  CND-IDS 원 논문의 설계를 그대로 이식한 결과, 버그는 아니지만 절반만
  "시간 흐름 반영"이다).
- **5개 experience에 걸쳐 조합별로 다 잘 학습되는가**: GPM을 제외한
  나머지(SSF/CADE/CND-IDS/SPIDER 유래 컴포넌트, null baseline)는 loss
  그래프 연결성·replay 그래디언트 흐름·optimizer 분리·추상 계약 일관성·
  train/eval 전환 전부 코드 근거로 문제없음을 확인했다. GPM만 위 2번
  항목의 문제가 있었고 지금은 고쳤다.
- **Backbone은 잘 짜여져 있는가**: `FCLAutoEncoder`(encoder→z,
  decoder→x_hat, classifier→z 직결)는 구조 자체가 단순하고 문제없다.
  `classifier(z)`(SSF 원문의 `classifier(decoder(z))`가 아님)는 이미
  의도적 절충으로 문서화되어 있다(위 절 참고). `ssf_backbone_dims()`도
  검증됐다.

## 5개 영역 병렬 재감사 (2026-08-14, 2차) — "논문과 일치하는가" 전면 재검증

**배경**: 위 구조적 완성도 감사 이후, "93개 조합이 쓰는 방법들이 논문의
실제 방법과 일치하는지 모든 코드를 줄마다 재확인해달라"는 요청에 따라
CADE/SSF/CND-IDS/GPM·SPIDER/파이프라인-조합로직 5개 영역을 각각 별도
에이전트로 처음부터 다시 대조했다(오늘 이미 적용한 수정들도 재검증
대상에 포함). GPM 에이전트는 WebFetch로 공식 저장소
(`sahagobinda/GPM/main_pmnist.py`)를 직접 가져와 대조했다.

### 재확인된 기존 수정 (전부 MATCH)

CADE class-aware pairing, SSF InfoNCE, SSF 메모리 버퍼 drift 방향, CND-IDS
multi-teacher LwF — 4건 전부 각 원문과 항별로 재대조해 정확함을 재확인했다.

### 이번에 새로 반영한 것

1. **CND-IDS `_metric_loss`의 "본질적으로 동일" 서술 정정**
   (`cndids_anti_forgetting.py`): 원문은 `TripletMarginLoss`+semihard
   마이닝(상대 마진 트리플릿, `CND_IDS.py:38-39,76-78`)인데 구현은 절대
   마진 페어와이즈다 — 방향(같은 클러스터는 가깝게, 다른 클러스터는
   margin 이상)은 같지만 손실의 수학적 형태가 달라 "동일한 근사"가 아니라
   "다른 형태로 근사"임을 명시했다. 새 의존성(`pytorch_metric_learning`)
   설치 위험 회피라는 채택 이유 자체는 유효해 구현은 유지.
2. **SSFSampleSelector의 "균일분포=SSF 대표성 개념" 서술 정정**
   (`ssf_sample_selector.py`): SSF 원문(`utils.py:109-190`)의 목표 분포는
   균일분포가 **전혀 아니다** — 현재 관측되는(드리프트된) 분포의 경험적
   히스토그램이다(drift-추종형 선택). 이 테스트베드의 균일분포 대체는
   인터페이스 제약(SampleSelector가 old/control 분포에 접근 불가, 원문의
   M_t 최적화 자체가 M_c에 의존)상 불가피했지만, "SSF 개념을 표현한 것"이
   아니라 "SSF의 핵심 메커니즘을 포기하고 완전히 다른 대체 휴리스틱(균일
   커버리지)을 쓴 것"이라고 정정했다.
3. **component_registry.py의 "CND-IDS 순정 조합" 표에서 `mm=cndids` 제거**:
   `CNDIDSMemoryManager` 자신의 docstring이 이미 "CND-IDS 원문에는 메모리가
   없다"고 밝히고 있는데, 레지스트리 요약 표는 여전히 `mm=cndids`를
   `mm=spider`와 나란히 "CND-IDS 순정" 선택지처럼 적어놓고 있었다 —
   `mm=none`이 실제로 더 순정에 가깝다는 점을 명시했다.
4. **CADEMADScorer `compute_threshold()` — 재검토 후 이중 MAD 유지, 인용만
   정정**: 원문(`detect.py:99`)은 이미 MAD-정규화된 점수를 상수
   `mad_threshold`(3.5)와 직접 비교하는데, 이 구현은 그 점수 위에 다시
   `median+t_mad*MAD`를 씌운다. A/B 실측(NSL-KDD, 순정 CADE 콤보)으로
   원문처럼 상수 `t_mad`만 쓰도록 바꿔봤더니 f1 0.6482→0.5746,
   bwt -0.1403→-0.1896으로 **오히려 악화**됨을 확인했다(pr_auc/roc_auc는
   불변 — score() 자체는 안 바뀜). 원인으로 보는 것: CADE 원문은 크고
   안정된 단일 코퍼스로 1회만 보정하는 정적 설계인데 이 테스트베드는
   라벨 예산만큼의 작은 정상 참조로 매 라운드 다시 보정해야 해서, 이중
   MAD가 라운드별 스케일 잡음을 흡수하는 적응적 역할을 하는 것으로
   보인다. 코드는 유지하고 잘못된 인용(`detect.py:91,150-158`을 이
   임계값 공식의 근거인 것처럼 쓴 부분)만 정정했다.
5. **GPM residual projection — 공식 Eq-9 그대로 재구현 후 재검증, 최종
   기각 확정**: 1차 시도(2026-08-14 앞선 절)는 정규화 방식이 공식과
   달랐을 가능성이 있다는 지적을 받아, WebFetch로 확인한 공식 Eq-9
   정규화(잔차가 아닌 원본 activation 총 에너지로 정규화, 누적치가 "기존
   기저가 이미 설명한 비율"에서 시작)로 다시 구현해 재검증했다. 결과는
   더 나빴다(f1 0.7006→0.5538, bwt -0.108→-0.170 — 1차 시도의
   f1 0.6440보다도 나쁨). 구현 결함 가능성을 배제한 채로도 나빠졌으므로,
   "residual projection이 이 아키텍처(얕은 backbone, latent_dim=32, 5
   experience)에 안 맞는다"는 결론을 확정하고 최종 되돌렸다 —
   `max_basis_ratio=0.9` cap만 유지.

### 사용자 판단이 필요한 것 (아직 반영하지 않음, 심각도순)

1. ~~**[최우선] "순정 CADE" 조합에서 CADE의 두 핵심 요소가 실제로 연결되지
   않음**~~ — **2026-08-14 해결 완료**, 아래 "CADE 인코더-MAD 연결" 절 참고.
2. **[중요, 미해결] 파이프라인 전체가 이진 라벨만 다뤄 CADE의
   "class-aware"가 실질적으로 퇴화**: class-incremental 분할에 쓰는
   다중클래스 category가 `experiences`에 담겨 어떤 컴포넌트에도 전달되지
   않는다(`dataset_loader.py`). 그 결과 CADE의 pairing은 "여러 공격
   유형별"이 아니라 "정상 vs 뭉뚱그려진 공격 평균"으로 단순화된다 — CADE의
   핵심 주장(어떤 known family와도 안 닮은 새 유형 탐지)이 온전히 재현되지
   않는다. 재검토 결과 범위가 처음 판단보다 크다 — `CADEDriftDetector`의
   pairing뿐 아니라 `CADEMADScorer`의 centroid 모델도(원문은 정상+각 family별
   centroid 여러 개 중 최솟값을 쓰는데, 지금은 정상 centroid 하나뿐) 다중
   클래스로 바꿔야 한다. 게다가 class-incremental 분할상 한 라운드엔 배정된
   카테고리만 등장하므로 family별 centroid는 라운드에 걸쳐 누적해야 하고,
   희소 카테고리(예: U2R 52건) 라운드의 centroid는 노이즈가 심할 수 있어
   "여러 노이즈 centroid 중 최솟값"이 오히려 지금 방식보다 못할 위험도 있다
   — GPM/SSF에서 실측으로 확인된 것과 같은 종류의 함정일 가능성. 아직
   구현하지 않았다 — 사용자 판단 대기.
3. **[참고] 4개 논문이 이 파이프라인과 원문에 얼마나 가까운지 순위 —
   CND-IDS ≫ SSF ≳ SPIDER &gt; CADE**(2번 항목 미해결 기준): n_experiences=5·
   class-incremental 분할·Track B 배치/에폭이 전부 CND-IDS 원문 그대로라,
   평가 운동장 자체가 CND-IDS의 "안방"이다. SSF는 부품(대표샘플 선택,
   KL-마스크, LwF)은 정확히 이식됐지만 절차(누적 풀+스트리밍) 수준은
   재현되지 않고 epoch=200은 원문(4~5)의 약 40배다. SPIDER는 원 코드 부재로
   검증 자체가 근본적으로 제한된다. CADE는 1번(연결)은 해결했지만 2번
   (다중클래스)과 "라운드 개념 자체가 원문에 없다"는 문제가 남아있다.
   리더보드 순위를 "어떤 논문이 최선인가"로 해석할 때 이 비대칭을 반드시
   함께 고지해야 한다.
4. ~~SSF LwF가 drift 시엔 원문에서 아예 꺼짐~~ — **2026-08-14 시도했다가
   되돌림**, `ssf_anti_forgetting.py` 모듈 docstring "2026-08-14" 절 참고.
5. SSFDriftDetector 최소 윈도우 크기 가드 누락 — class-incremental 분할이
   만드는 불균등한 라운드 크기(예: exp3=52건)에서 실제 영향 가능성 재확인.
   경미하지만 실재하는 문제. 아직 미해결.

### CADE 인코더-MAD 연결 (2026-08-14 해결)

`CADEDriftDetector`가 학습시키는 사설 대조학습 인코더의 출력이 파이프라인
어디에도 안 쓰이고, `CADEMADScorer`는 무관한 공유 backbone의 z에 MAD
공식만 적용하던 문제(위 1번)를 해결했다. `dd=cade`와 `as=cade_mad`가
함께 선택된 콤보에서만 `CLClient.__init__`이 `CADEMADScorer.
set_private_encoder()`로 둘을 연결한다 — `drift_detector.
uses_shared_representation`과 대칭인 `anomaly_scorer.
uses_shared_representation` 플래그를 추가해 Step 6/7(및 `grid_runner.py`의
추론 지연 측정, `smoke_test.py`의 15.5 게이트)의 인코딩 경로를 분기했다.
`dd=cade`가 아닌 조합에서 `as=cade_mad`를 쓰면 여전히 공유 backbone의
z를 쓴다(버그가 아니라 "MAD 통계 판정이 대조학습 없이도 통하는가"를 보는
정당한 재조합 실험으로 남김).

A/B 실측(NSL-KDD, 순정 CADE 콤보 `dd=cade/ss=random/mm=none/af=none/
as=cade_mad`)으로 f1 0.6482→0.7898(+22%), precision 0.795→0.893,
recall 0.547→0.708, pr_auc 0.749→0.922, roc_auc 0.741→0.877,
bwt -0.1403→-0.0810 — **전 지표가 큰 폭으로 개선**됨을 확인했다. 이는
이번 세션 전체를 통틀어 가장 큰 단일 개선폭이며, CADE가 자기 알고리즘대로
작동하지 않고 있었다는 진단이 정확했음을 강하게 뒷받침한다. 93개 조합
스모크 테스트(93/93 통과)와 `pytest`(13/13)로 회귀 없음을 확인했다.

전부 `pytest`(13/13)와 93개 조합 스모크 테스트로 회귀 없음을 확인했다.

## CADE 다중클래스(family) centroid 연결 — 두 차례 구현 결함 발견·수정 (2026-08-25)

### 배경

위 연결(인코더-MAD)은 됐지만, `CADEDriftDetector.fit()`에 넘기는 `labels`가
여전히 이진(정상/공격)이었다 — CADE 원문의 실제 단위는 "정상 + 각 공격
family"고(`detect.py:62` centroid, `data.py:268-345` pairing), DoS/Probe/
R2L/U2R을 "공격" 한 뭉치로 묶으면 CADE의 핵심(family 단위 최근접 판정)이
성립하지 않는다. `data/dataset_loader.py`가 이미 class-incremental 분할에
쓰던 `category`를 `train_category`로 노출하고, `cl_client.py`가 이를
`CADEDriftDetector.fit_with_category()`(신규)에 넘기도록 배선했다.

### 1차 시도 — raw 데이터로 그 라운드에 한 번만 centroid 계산 → 대실패

`group.unique()`로 이번 라운드에 있는 category만 그 라운드 raw 데이터로
centroid를 만들고, 없어진 category는 이전 값을 그대로 두는 방식이었다.
NSL-KDD 순정 CADE 콤보로 A/B: **f1 0.7713→0.0804(recall 0.66→0.04)로
붕괴**. 원인: 사설 인코더는 CADE 원문(정적 1회 학습)과 달리 매 라운드
계속 미세조정되는데, 한 번 등장했다가 사라진 family의 centroid는 등장
당시 인코더 좌표에 박제된 채 남아 이후 라운드의(이미 이동한) 인코더
좌표계와 어긋난다. 특히 NSL-KDD 마지막 experience(exp4)는 공격이 전혀
없어(class-incremental 분할 설계상 자연 발생) 그 라운드는 "정상끼리만
뭉치기" 대조학습만 수행 — 정상 centroid만 갱신되고 나머지는 낡은 채로
남아 최종 판정 기하가 완전히 어긋났다.

**사용자 지적("성능이 이렇게 붕괴한 건 분명 문제가 있다는 뜻인데, 왜 이렇게
자잘한 문제가 많이 생기는지 이해가 안 간다. 처음부터 끝까지 다시 검토해줘")
에 따라 "구조적 불일치"로 성급히 결론 내리지 않고 원인을 계속 추적했다.**

### 2차 시도 — 참조 표본을 매 라운드 현재 인코더로 재인코딩 → 여전히 실패(f1=0.0)

category별 raw 참조 표본을 인스턴스 수명 전체에 걸쳐 누적 보관(`_category_
refs`, 최근 500개 캡)했다가, 매 라운드 **알려진 모든 category**의 centroid를
**현재** 인코더로 다시 계산(`_recompute_all_centroids`)하도록 재설계했다.
1차 문제(좌표계 어긋남)는 없앴지만 결과는 더 나빠졌다(**f1=0.0, precision=
recall=0** — 어떤 표본도 threshold를 못 넘음). 진단 결과: 라운드가 진행될수록
`min_anomaly_score`의 (전체 표본 대상) 최댓값이 14.98→3.83→3.06→1.87로
단조 감소하다 마지막 라운드에 threshold(2.80)가 그 압축된 스케일 위에서
계산돼 아무 표본도 넘지 못했다. 근본 원인: 이 테스트베드는 한 라운드에
"정상 + 공격 family 하나"만 등장하므로, 사설 인코더가 매 라운드 그 2-클래스
로만 재학습되면서 **예전에 본 family와 정상을 구분하는 능력을 라운드가
지날수록 잃어버린다**(파국적 망각) — 재인코딩은 "지금 좌표"를 쓰게는
해줬지만, 그 "지금 좌표" 자체가 이미 예전 family들과 정상을 구분 못 하는
붕괴된 좌표였다.

### 3차 시도 — 사설 인코더 전용 최소 리플레이 도입 → 채택

`_category_refs`에 쌓아둔 과거 라운드의 참조 표본을 이번 라운드 대조학습
배치에 섞어 넣어(`_replay_known_categories`), encoder가 이번 라운드에 없는
과거 family와의 구분도 계속 "리허설"하도록 했다. NSL-KDD A/B: **f1 0.0→
0.4334(precision 0.810, recall 0.296), pr_auc 0.595→0.764, bwt -0.198→
0.097**로 회복 — 더 이상 붕괴하지 않고 정상적으로 작동한다(min_anomaly_score
최댓값이 라운드가 지나도 13~15 수준으로 유지됨, 단조 압축 현상 사라짐).

### 최종 판단 — 이진 baseline(f1 0.7713) 대비 여전히 낮지만 채택

남은 격차는 버그가 아니라 시나리오 자체의 데이터 희소성으로 판단한다 —
label_budget(10%) 적용 후 U2R은 라운드당 약 4~5건만 선택되고, 이 family는
class-incremental 분할 설계상 정확히 한 라운드에만 등장해 리플레이로도
표본 수 자체를 늘릴 수 없다(원 논문도 이 정도로 극단적으로 적은 표본으로는
안정된 centroid를 못 만든다 — CADE 고유의 한계이지 이식 결함이 아니다).

이 테스트베드의 평가 기준은 성능이 아니라 "각 논문의 실제 기법을 얼마나
충실하게, 안정적으로 재현하는 조합인가"다(세션 내 사용자 확정 기준). 이
기준으로 보면: (1) 다중클래스 family centroid + family 단위 contrastive
pairing은 CADE 원문의 핵심 메커니즘을 이진 방식보다 훨씬 충실하게 재현하고,
(2) 3차 시도 이후로는 안정적으로(f1=0 같은 붕괴 없이) 작동한다. GPM
residual projection/CADEMADScorer 단일 t_mad/SSF drift-gated LwF와
달리 — 저 세 사례는 "원문을 충실히 재현했더니 이 테스트베드 구조와 안 맞아
**항상** 더 나빴다"는 반복 확인 끝에 되돌린 것이고, 이번 건은 "구현 결함
두 개를 실제로 찾아 고쳤고, 남은 격차는 CADE 자체도 못 피하는 극단적 데이터
희소성"이라는 점에서 성격이 다르다 — **되돌리지 않고 유지**한다.

`max_category_ref=500`(신규 hparam, 원문 근거 없음 — 이 테스트베드가 encoder를
계속 재학습한다는 구조적 차이를 보정하기 위한 전용 장치)는
`configs/component_hparams/cade.yaml`에 문서화한다.

### 93개 조합 스모크 테스트로 발견한 4번째 문제 — ss=ssf와의 상호작용

위 3차 시도까지는 `ss=random` 콤보 하나만 A/B로 확인한 상태였다. 사용자
지적("처음부터 끝까지 다시 검토해달라")에 따라 전체 93개 조합 스모크
테스트로 넓혀서 재확인한 결과, `dd=cade`+`ss=ssf`+`as=cade_mad` 9개 조합
(mm/af 조합과 무관하게 전부)에서 새로운 실패를 발견했다 — exp0에서
threshold=33.81인데 그 라운드 전체 평가 score 범위가 [0.0002,15.17]밖에
안 돼(15.2 예측 완전 퇴화, 15.3 threshold 범위 이탈) 모든 표본이 "정상"으로
판정됐다. 이 9개 조합은 Fix 2 이전(이진 단일 centroid)에는 전부 정상
통과했다(`smoke_v4_final.log` 확인) — 순수하게 Fix 2가 만든 회귀였다.

원인: `ss=ssf`는 SSF 원문과 달리 이 테스트베드에서 "분포 전 구간에 고르게
퍼지도록" 표본을 뽑는 균일-커버리지 선택으로 대체돼 있다(`ssf_sample_
selector.py` 참고, 원문 mask 최적화 자체는 SampleSelector 인터페이스만으로
재현 불가해 대체한 것 — 별개로 이미 문서화됨). 이 균일-커버리지로 뽑힌
"정상 참조" 표본 중 일부는 원래 특징 공간에서 경계/극단에 위치해, 다중
family min-centroid 점수에서 우연히 어떤 공격 family centroid에 더
가까워 유난히 낮은 점수를 받고, 나머지는 정상 centroid 기준의 정상적인
점수를 받는다 — 이 이질적인 분포 때문에 `compute_threshold`의 이중 MAD
공식(median+3.5*MAD)이 계산 근거였던 eval_scores 자기 자신의 최댓값조차
넘는 값을 냈다. 이진 단일 centroid 때는 점수가 더 균질해 이 문제가
드러나지 않았다.

수정: `CADEMADScorer.compute_threshold()`가 계산한 threshold를
`eval_scores.max()`로 clamp한다 — threshold가 자신의 계산 근거였던
분포의 최댓값조차 넘으면 이미 그 계산이 보증하는 범위를 벗어난 것이므로,
이중 MAD 공식 자체(A/B로 이미 검증된 부분)는 손대지 않고 이런 병적인
경우만 막는 최소 안전장치다. 이후 93개 조합 nsl-kdd 스모크 테스트
**93/93 전부 통과**, `pytest`(13/13) 회귀 없음 확인.

CICIDS2018/UNSW-NB15는 로컬 메모리 제약(전자는 원본 CSV 병합 시 GiB 단위
배열 할당 실패, 후자는 공식 attack_cat 원본 파일 미보유)으로 로컬 검증
불가 — 기존 관례대로 GPU 서버에서 93×3 전체 재실행 시 최종 확인한다.

## 4개 논문 컴포넌트 전수 재감사 (2026-08-26)

사용자 요청("테스트베드로서 역할을 잘 할 수 있는지 모든 파일의 코드를
소스코드와 비교하면서 직접 검토")으로, SSF/CADE/CND-IDS/SPIDER-GPM 4개
영역을 각각 전담하는 병렬 에이전트(로컬에 있는 각 논문 원본 저장소와
직접 대조, 실행 기반 검증 포함)와 파이프라인/공통 인프라 계층에 대한
직접 검토를 병행했다. 이전 감사들이 "공식/citation이 원문과 일치하는가"
위주였다면, 이번엔 그에 더해 "실제로 돌려봤을 때 라운드별 상태가 올바르게
유지되는가", "다른 컴포넌트와 결합했을 때도 안전한가"까지 확인했다.

### 발견 1(확정) — grid_runner.py의 코드 버전 캐시 자체에 구멍

`compute_code_version()`의 `_VERSIONED_PATHS`가 `configs/`(하이퍼파라미터
YAML)와 `common/`(F1/BWT 계산식)을 포함하지 않았다 — `git log`로 이
세션 동안만도 component_hparams/*.yaml이 여러 번 실제로 바뀌었음을
확인했다. `.py` 파일 없이 하이퍼파라미터만 고치고 그리드를 재실행하면
바뀐 값이 반영 안 된 옛 결과가 조용히 재사용될 뻔했다 — 270개 스테일
결과 사고(2026-08-14)를 막으려고 만든 안전장치 자신이 같은 종류의
구멍을 갖고 있었던 것. `configs/`, `common/`, `experiments/grid_runner.py`
자신을 추가하고, 디렉터리 필터가 `.py`만 인식하던 것도 `.yaml`까지
인식하도록 함께 고쳤다(이것도 안 고쳤으면 `configs/`를 목록에 추가해도
실제로는 아무 파일도 안 잡혔을 것).

### 발견 2(확정) — 스모크 테스트가 뒤쪽 라운드를 한 번도 검사하지 못한 구조적 사각지대

`SMOKE_N_EXPERIENCES=2`(하드코딩)가 5개 experience 중 앞 2개만 검사했다.
class-incremental 분할은 설계상 희귀·어려운 category와 공격 0건 라운드를
**항상 뒤쪽**에 배치하므로, 정확히 그 라운드들이 한 번도 스모크 테스트를
거치지 않았다. 이 사각지대 때문에 아래 두 개의 심각한 붕괴(af=gpm,
af=cndids)가 "통과" 처리된 채 방치돼 있었다. `SMOKE_N_EXPERIENCES`를
`None`(=전체)으로 바꾸고, 15.2 게이트를 강화(다수 클래스 비율 ≥0.97을
경고가 아니라 실패로), roc_auc 역전 감지(15.2b, 신규)와 CND-IDS
pseudo-label 쏠림의 조건부 실패 처리(15.4)를 추가했다 — 소요 시간은
늘어나지만(대략 2.5배) 사용자 지시대로 정확성을 우선한다.

### 발견 3(확정, 최종 원인은 GPM 결함이 아니었음) — af=gpm 완전 붕괴

전체 5라운드 실행 시 `af=gpm`이 f1=0.0053, pr_auc=0.52(거의 무작위),
bwt=-0.628까지 붕괴(아무 망각방지 없는 af=none보다 훨씬 나쁨). 두 단계로
원인을 추적했다:
1. bias가 GPM의 gradient projection에서 완전히 빠져 있음을 발견해
   weight+bias를 하나로 증강해 함께 사영하도록 고쳤다 — 하지만 A/B
   실측으로 이 수정만으로는 붕괴가 전혀 안 풀렸다(f1 그대로, roc_auc는
   오히려 0.265로 악화). 이 수정 자체는 이론적으로 정당하고 해롭지도
   않아 유지하지만, 붕괴의 원인은 아니었다.
2. 계속 추적한 결과 **`af=gpm`과 `as=none`(고정 0.5 임계값)의 궁합
   문제**였다 — GPM의 gradient projection 압력 아래서 classifier 로짓
   스케일이 라운드마다 계속 커지는데(weight_norm 3.08→6.16), `as=none`의
   고정 임계값이 이를 전혀 못 따라간다. `as=cade_mad`(매 라운드 재보정)로
   바꾸면 f1 0.0054→**0.6655**, bwt -0.507→**-0.1405**로 완전히 건강해진다
   (naive fine-tuning의 bwt -0.433보다 훨씬 덜 잊음 — GPM이 원래 주장하는
   효과 그대로).
GPM 코드를 더 고쳐 이 조합을 억지로 성능 개선하지 않는다 — 두 컴포넌트
각각은 자기 논문대로 충실하지만 서로 안 맞는 "진짜 비호환 조합"이 발견된
것이고, 발견 2의 강화된 게이트가 이걸 정확히 실패로 잡아 그리드에서
제외하는 게 올바른 처리다.

부수적으로 GPM의 `compute_loss()`가 `replay_batch`를 전혀 쓰지 않는다는
것도 발견했다 — GPM 원 논문 자체는 리플레이가 필요 없다는 게 핵심 주장
이라 `mm=none`(순정 GPM)과 결합됐을 땐 맞는 동작이지만, SPIDER 논문은
바로 그 GPM에 별도 유한 버퍼를 추가한 것이 핵심 기여라 `mm=spider`와
결합됐을 때 그 버퍼를 아예 안 쓰면 "SPIDER"를 재현하지 못한다. replay_batch
가 있을 때만 그 위에도 같은 task loss를 더하도록 고쳤다(`mm=none`과
결합되면 replay_batch가 항상 None이라 원 논문 그대로 동작이 자동
보존된다). A/B 실측: `af=gpm+mm=spider+as=cade_mad`(완전한 SPIDER)
f1=0.8544, **bwt=+0.0423**(과거 태스크 성능이 오히려 개선) — GPM만
쓸 때(f1=0.6655)보다도 뚜렷이 낫다.

**재감사에서 발견 — 이 결과에 두 가지 단서가 붙는다**: (1) `mm=spider`의
리플레이 배치는 실제 라벨이 아니라 SPIDERMemoryManager의 스냅샷 모델이
만든 pseudo-label이다(`spider_memory_manager.py` 참고) — 즉 `af=gpm+
mm=spider`도 `af=lwf_ssf`/`af=none`에 대해서만 분석됐던 자기학습
(self-training) 우려의 대상에 포함된다(뚜렷한 붕괴는 관찰 안 됐지만
"자기학습과 무관한 통제군"은 아니다). (2) 재실행(f1=0.8399, bwt=+0.0992
— 방향은 같지만 수치는 다름, 그 사이 다른 수정들이 겹친 영향으로 보임)
에서 R-matrix를 라운드별로 분해해보니, 양의 BWT 대부분이 R2L 라운드
하나(diag F1≈0.31→최종 F1≈0.67)에서 나왔는데, 그 구간에서 classifier
weight_norm은 거의 안 변한 반면(6.05→6.08) `as=cade_mad`의 threshold는
3.43→3.76으로 라운드마다 다시 계산된다 — 즉 이 양의 BWT는 "진짜 표현
수준의 역전이(backward transfer)"와 "매 라운드 재계산되는 MAD 임계값의
표본 잡음"이 섞인 결과일 가능성이 있다(이중 MAD 재보정 자체가 라운드별
스케일 흔들림을 흡수하는 적응적 성격이라는 건 `cade_anomaly_scorer.py`
의 2026-08-14 절에 이미 문서화됨). 데이터 누출은 아니다(정상 참조와
SPIDER 버퍼 모두 그 라운드의 selected_data에서만 나오고, test는 Step 7
평가에만 쓰인다 — 확인 완료) — 다만 "깨끗한 backward transfer"라는
표현은 이 단서를 달아 이해해야 한다.

### 발견 4(확정) — CND-IDS pseudo-label 참조가 공격 희귀 라운드에서 거의 전체를 뒤덮음

`on_experience_start`가 받는 "정상 참조"(`normal_subset`)가 매 라운드
**그 라운드 자신의** 라벨 구성에서 다시 정의되는데, 원문의 `datastream.
init_normal`은 스트림 시작 전 **한 번** 확정되는 고정 참조다. 공격이
희귀한 라운드(R2L 6.9%, U2R 0.38%)는 `normal_subset`이 라운드의
93~100%까지 차지해, K-Means 클러스터 대부분이 참조 데이터를 하나는
포함하게 되어 거의 모든 클러스터가 "정상"으로 판정됐다(실측: pseudo-label
다수 비율 R2L 0.9844, U2R 1.0000). U2R 라운드 직후 0.870이었던 정확도가
다음 라운드엔 0.707로 떨어지는 실제 망각 효과까지 확인했다. `normal_subset`
을 인스턴스 수명 전체에 걸쳐 누적하는 별도 풀(`_normal_ref_pool`, 최근
5000개 캡)로 바꾸고, "이 클러스터가 정상인가" 판정에 그 누적 풀을 쓰도록
고쳤다(클러스터링 자체는 원문처럼 매 라운드 새로 fit). A/B 실측: f1
→**0.889**, roc_auc→**0.895**, bwt→**+0.037**, U2R 라운드 자체 성능
(diag-F1)이 **0.44**까지 회복(붕괴 이전엔 사실상 미학습 수준).

**2026-08-26 재감사에서 발견 — 위 "누적"이 사실상 전혀 누적되지
않고 있었다(원인 재규명, 효과 자체는 유지)**: 처음 구현(`combined_ref
[-cap:]`, 꼬리 슬라이싱)은 Track B가 label_budget 없이 experience
전체를 쓰는 탓에 `normal_subset` 자체가 이미 매 라운드 cap(5000)보다
훨씬 크다(NSL-KDD 13,468~59,396건) — 그 결과 `combined_ref`의 마지막
`cap`개는 **항상 100% 이번 라운드 자신의 데이터**였다(provenance
추적으로 실측: 5라운드 전체에서 이전 라운드 표본 생존율 0%). 위에서
관찰된 개선은 "라운드를 넘어 기억한다"가 아니라 **"참조 표본 수가
줄어들면 K-Means 클러스터가 덜 뒤덮인다"**는 원 논문과 무관한 우연한
부작용이었다(같은 클러스터링에 참조만 전체 vs 5000건 무작위로 바꿔
분리 검증: R2L pseudo_ratio 0.9720→0.9634, U2R 0.9998→0.9962 — 표본
수 축소 자체가 효과의 전부). 꼬리 슬라이싱을 무작위 표본으로 바꿔 실제
누적이 되도록 고쳤다 — `CNDIDSMemoryManager`(정상 전용 FIFO 버퍼)도
같은 근본 원인(버퍼도 max_size=1000보다 라운드당 표본 수가 훨씬 큼)의
같은 문제가 있어 함께 고쳤다. **재수정 후 프로덕션 규모(K후보 최대
2000, cluster_fit_sample_size=10000) 재실행 결과**: f1=0.8218,
roc_auc=0.8684, bwt=-0.0403, diag-F1=[0.949,0.882,0.613,**0.493**,0.0]
— 꼬리 슬라이싱 버전(f1=0.855, roc_auc=0.851, U2R diag=0.43~0.46) 대비
전체 f1은 소폭 낮아졌지만 roc_auc와 U2R 자체 성능은 오히려 더 낫다(R2L은
소폭 하락). 우연히 작동하던 메커니즘을 실제로 의도한 대로 작동하는
메커니즘으로 바꾼 것이므로, 절대 수치의 소폭 변동은 "성능 하락"이 아니라
"이제 실제로 옳은 걸 계산한다"는 관점에서 받아들인다(이 테스트베드의
판단 기준 — 사용자 확정). 참고로 위 발견4 수치(0.889/0.895/+0.037)는
MinMaxScaler 시간 유출 수정("발견 6-보충") **이전**에 측정된 것이라
그 자체로도 이미 낡았다 — 두 수정을 함께 반영한 재측정에서는 f1=0.855,
roc_auc=0.851, **bwt=-0.047**(부호 반전)로 나왔다(U2R diag-F1은 0.458로
여전히 회복 유지). 이후 A/B는 반드시 그 시점의 최신 코드 전체로
재확인해야 한다는 교훈이 두 번째로 확인된 것 — `ssf_sample_selector.py`
모듈 docstring의 같은 교훈 참고.

### 발견 5(확정) — SSF 리플레이 버퍼의 소수 클래스 완전 소실

`SSFMemoryManager.update()`의 대표성 점수(자기 클래스 centroid까지의
거리)가 클래스 자신의 특징 분산에 스케일이 좌우되는데, 이 점수를 클래스
구분 없이 **전역** `topk`로 랭킹해 버퍼를 채우고 있었다. 실측(NSL-KDD
5라운드 전체, `dd=ssf/ss=ssf/mm=ssf/af=lwf_ssf/as=cade_mad`): 라운드3
(U2R)에서 선택된 attack 5건이 그 즉시 같은 `update()` 호출 안에서 버퍼
attack 슬롯(369개, 전부 DoS/Probe/R2L)에 밀려 **0/5 생존**. 클래스별로
버퍼 슬롯을 구성비만큼(최소 1개) 미리 배정하고 그 쿼터 안에서만 top-k를
매기도록 고쳤다(`_quota_topk_indices`). `SSFSampleSelector`도 같은 종류의
문제(균일-히스토그램 대체가 클래스를 몰라 라운드마다 예측 불가능하게
비율을 왜곡 — R2L 라운드는 비례 기대치 대비 82% 적게 선택됨)를 같은
방식(`_quota_select`)으로 고쳤다.

**다중클래스(category) 쿼터로 더 확장 — 처음엔 둘 다 기각했다가, 재검증
후 선택기만 채택**: 이진 쿼터 수정 후에도 U2R 라운드 자체 성능이
0.048→0.048로 거의 그대로여서, CADE처럼 `train_category`(다중클래스)로
쿼터를 나누는 확장을 선택기/버퍼 양쪽에 각각 시도했다. **처음** A/B
(f1 0.7291 기준선)는 둘 다 순손해로 나와 둘 다 되돌렸는데, 4개 논문
컴포넌트 전수 재감사에서 이 A/B가 그 사이 반영된 MinMaxScaler 시간 유출
수정("발견 6-보충") 이전의 낡은 숫자와 비교된 것이었음이 드러났다 —
전처리가 그리드 전체의 절대 수치를 같이 움직이므로, 4개 변형 중 어느
게 최선인지의 **순위 자체**가 달라질 수 있다는 걸 놓쳤던 것이다.
현재 코드로 4개 변형을 처음부터 다시 측정한 결과:
  - 이진 쿼터만: f1=0.6565, roc_auc=0.7150, diag-F1=[0.851,0.554,
    0.257,0.009,0.0]
  - **선택기만 category 쿼터: f1=0.7040, roc_auc=0.6451, diag-F1=
    [0.846,0.556,0.254,0.067,0.0]** — 전체 f1도 U2R도 이진보다 낫다.
  - 버퍼만 category 쿼터: f1=0.5107, roc_auc=0.5101(거의 무작위) — 여전히 나쁨.
  - 둘 다 category 쿼터: f1=0.5879, roc_auc=0.5276 — 여전히 이진보다 나쁨.
**결론이 뒤집혔다**: 선택기의 category 쿼터는 채택(`SSFSampleSelector.
select_with_category()`, 최종 코드에 반영), 버퍼의 category 쿼터는
재검증 후에도 기각 유지(`SSFMemoryManager`는 이진 쿼터만) — CADE의
다중클래스 centroid(family별 완전히 분리된 표현 공간을 만드는 게
메커니즘 자체)와 달리, SSF의 "대표성" 대체 휴리스틱에서는 "예산을 어떻게
배정할까"(선택기)는 category 단위가 도움이 되지만 "이미 담긴 표본
중 누구를 내보낼까"(버퍼)는 여전히 손해라는, 두 메커니즘의 비대칭적
반응으로 보인다. **교훈**: 그리드 전체에 영향을 주는 변경(전처리 등)
이후에는 이전 A/B 결론을 그대로 믿지 않고 반드시 재확인해야 한다.

### 발견 6-보충 — 전처리 시간 유출: MinMaxScaler + CICIDS2018 포트 빈도

4개 컴포넌트 재감사와는 별개로 직접 검토한 결과, 이미 알려져 있던(보류
상태였던) "MinMaxScaler가 train 파일 전체에 한 번에 fit된다" 문제가
스케일러 하나만의 문제가 아니라는 걸 확인했다 — CICIDS2018의 Dst Port
빈도 범주화(`_bucket_port_frequency`)도 10일치 전체(=모든 experience를
합친 것)를 병합한 뒤 계산되고 있었다. 둘 다 "라운드로 나누기 전에 전체
데이터로 전처리 통계를 미리 계산한다"는 같은 패턴이라, experience 0의
모델이 실제로는 아직 등장하지 않은 미래 experience의 데이터 범위/포트
빈도까지 이미 알고 있는 셈이었다 — 지속학습 벤치마크의 "미래 정보 없음"
원칙에 어긋난다. 둘 다 고쳤다:
- **MinMaxScaler**: 원본(미정규화) 데이터로 먼저 class-incremental 분할을
  하고, `partial_fit()`으로 라운드 순서대로 누적 갱신 — experience i는
  0..i의 통계만 안다. 3개 데이터셋 전부 적용, NSL-KDD로 로컬 검증 완료
  (pytest 15/15, 여러 콤보 A/B 실행 정상).
- **CICIDS2018 포트 빈도**: `_class_incremental_split`에 `extra_arrays`
  파라미터를 추가해(NSL-KDD/UNSW-NB15는 안 넘기므로 기존 2-튜플 반환에
  영향 없음) port/protocol 원본을 X/y/category와 같은 순서로 라운드별
  슬라이싱한 뒤, `_IncrementalPortBucketer`로 라운드 누적 빈도 기준
  버킷을 매긴다. protocol 원-핫의 범주 **집합**은 예외적으로 전체
  데이터에서 한 번 고정한다(스키마 정보이지 통계가 아니라 미래 누출이
  아니라고 판단 — "이 필드가 어떤 값을 가질 수 있는가"는 프로토콜
  표준이 정하는 것이지 라운드별 관측 분포가 아니다). CICIDS2018 자체는
  로컬 메모리 제약으로 끝까지 실행 못 하지만, 합성 데이터로 핵심 불변
  조건(라운드마다 input_dim이 안 바뀌는가, 버킷이 항상 {0,1,2}인가,
  extra_arrays 슬라이싱이 X/y/category와 정확히 같은 행 순서인가)을
  회귀 테스트로 고정했다(`tests/test_cicids_incremental_preprocessing.py`)
  — GPU 서버 실행 시 실제 규모로 최종 확인한다.

### 발견 6-보충2 — 스모크 테스트 15.2/15.4 게이트만으로는 안 잡히는 경우

CADE 감사 에이전트가 `dd=cade/ss=ssf/mm=spider/af=gpm/as=cade_mad`를
축소된 epoch(20, 기계 부하로 인한 진단용 단축)으로 실행해 roc_auc=0.151
(역전 의심)을 발견했었다. 발견 3·4·5·6의 수정이 모두 반영된 뒤 같은
콤보를 **전체 200 epoch**로 재실행한 결과 roc_auc=0.635~0.711로 건강함을
확인했다 — 축소된 epoch에서의 불안정성이었거나, 오늘의 다른 수정들이
부수적으로 해결한 것으로 보인다. 별도의 CADE 전용 수정은 필요 없었다.
다만 이 경험 자체가 "축소된 설정으로 진단한 결과는 반드시 전체 설정으로
재확인해야 한다"는 걸 재확인시켜준다 — 이후 모든 A/B는 가능한 한 실제
프로덕션 하이퍼파라미터(200 epoch 등)로 확인했다.

### 발견 6(시도했다가 되돌림) — SSF LwF distillation을 replay_batch까지 확장

`compute_loss()`의 distillation 항이 `new_batch`에만 적용되고
`replay_batch`에는 전혀 적용되지 않고 있음을 발견했다 — SSF 원문
(`ssf.py:296-330`)은 old+new를 합친 전체 배치에 distillation을 적용하므로,
replay_batch에도 teacher와의 MSE를 더하도록 확장해봤다. **A/B 실측
결과 오히려 나빠져(f1 0.6565→0.6128, roc_auc 0.7150→0.5981) 되돌렸다**
— SSF의 teacher는 매 라운드 그 시점 모델 전체를 스냅샷한 것이라,
replay_batch(과거 라운드 데이터) 위에서도 teacher와 가까워지라고
강제하면 teacher 자신이 이미 그 데이터로 학습된 상태라 distillation
신호가 자기 자신을 재확인하는 것에 가까워지고, 정작 new_batch(이번
라운드에 새로 배워야 할 것) 쪽으로 가야 할 gradient 용량을 깎아먹는
것으로 보인다 — GPM residual projection/CADEMADScorer 단일 t_mad와
같은 "원문에 더 충실하지만 이 테스트베드 구조와 안 맞는" 패턴. 원래의
new_batch만 distillation하는 방식을 유지한다(`ssf_anti_forgetting.py`
모듈 docstring 참고).

## Track A/B 전체 재감사 후 발견 — Track B에 anomaly_scorer=cade_mad 추가 (2026-09-01)

Track A/B 구조 전수 재감사(사용자 지시: "아예 원초적인 걸로 돌아가서 track
A/B로 나눈 것이 옳은 건지 등 큰 문제 뿐 아니라 사소한 문제까지 전부 다
분석해")에서, Track B의 `anomaly_scorer`가 `pca`로만 고정된 것이 실제
코드 구조상 근거가 없는 제약이었음을 발견했다:

1. **"classifier 전용/autoencoder 전용" 구분이 런타임에 전혀 검사되지
   않는다**: `CADEMADScorer.required_backbone = "classifier"`,
   `PCAScorer.required_backbone = "autoencoder"`로 선언은 돼 있지만
   (`발견했지만 낮은 우선순위라 손대지 않은 것` 절 참고), `base/models.py`의
   `FCLAutoEncoder`가 Track A/B 공통으로 z/x_hat/logit을 모두 만드는
   하나의 모델이라 — CADEMADScorer가 소비하는 z는 Track A든 B든
   구조적으로 완전히 동일하다.
2. **이미 있던 대칭 사례**: Track A에 `dd=none`+`as=cade_mad`(CADE의
   대조학습 인코더 없이 MAD 채점 방식만 공유 z 위에 적용)가 정당한
   재조합으로 이미 존재한다(`cade_anomaly_scorer.py` "2026-08-14" 절
   참고). 이와 대칭으로 "CND-IDS의 라벨-프리 표현학습 위에 CADE의
   median+MAD 채점 방식만 얹으면 어떤가"도 구조적으로 똑같이 타당하다.
3. **threshold 계산 방식이 실제로는 Track이 아니라 scorer 자체 속성으로
   결정된다**: 기존엔 `CLClient`가 `self.track == "A"`로 분기해 Track A는
   정상 참조(s_ref) 기반 median+MAD, Track B는 Best-F(라벨 필요)를 썼는데,
   `pca`=라벨 필요, `cade_mad`/`none`=라벨 불필요가 지금까지 Track과
   100% 겹쳤을 뿐 논리적 인과관계는 아니었다. `BaseAnomalyScorer`에
   `threshold_needs_labels` 플래그(기본 False, `PCAScorer`만 True)를
   추가해 `CLClient` Step 7이 이 플래그로 분기하도록 일반화했다
   (`base/anomaly_scorer.py`/`pipeline/cl_client.py` "2026-09-01" 절 참고).
   이 리팩터는 Track A의 기존 동작을 그대로 보존한다(기본값이 예전 분기와
   정확히 같은 결과를 내도록 설계) — Track A 90개 조합은 영향을 받지 않는다.

사용자 승인("지금 진행(권장)") 후 `common/compatibility.py`의
`TRACK_B_GRID["anomaly_scorer"]`에 `cade_mad`를 추가했다(전체 유효 조합
93→96개, Track B 3→6개). 새로 생긴 3개 조합
(`mm=none/spider/cndids` × `as=cade_mad`)을 NSL-KDD로 `run_combo_full`
전체 실행(200/20 epoch, 5 experience 전부, CPU)해 기존 `as=pca` 3개와
직접 A/B 비교했다:

| memory_manager | scorer | f1 | precision | recall | roc_auc | bwt |
|---|---|---|---|---|---|---|
| none | cade_mad | 0.8739 | 0.9047 | 0.8452 | 0.9329 | +0.0152 |
| none | pca | 0.8218 | 0.8621 | 0.7852 | 0.8684 | -0.0403 |
| spider | cade_mad | 0.8495 | 0.9011 | 0.8035 | 0.8933 | -0.0686 |
| spider | pca | 0.8718 | 0.8819 | 0.8618 | 0.8856 | +0.0146 |
| cndids | cade_mad | 0.8558 | 0.9005 | 0.8153 | 0.9088 | -0.0495 |
| cndids | pca | 0.8497 | 0.8768 | 0.8242 | 0.8705 | -0.0316 |

6개 조합 전부 f1 0.82~0.87, roc_auc 0.87~0.93 범위의 건강한 값을 내고
어느 쪽으로도 퇴화(단일 클래스 쏠림, roc_auc 역전, threshold 범위 이탈)
하지 않았다 — `mm=spider`에서는 `pca`가, `mm=none`/`mm=cndids`에서는
`cade_mad`가 더 나아 결과가 memory_manager에 따라 갈린다(어느 한쪽이
항상 우세하지 않음). 이 테스트베드의 목적은 "조합의 성능이 올라가는 것"이
아니라 "논문에 충실한 재조합의 다양성"이므로, 이 mixed 결과 자체가 —
`cade_mad`가 Track B에서 `pca`의 단순 중복이 아니라 memory_manager와
상호작용하는 독립적인 축임을 보여주는 — 이 추가가 진짜로 의미 있는 조합
확장이라는 근거가 된다. perf_matrix 마지막 라운드(exp4) 대각선이 6개
전부 0.0인 것은 버그가 아니라 NSL-KDD의 class-incremental 분할에서
exp4에 공격이 아예 없는 것으로 이미 알려진 구조적 특성이다(위 "Phase 2"
계획 문서 참고).

검증 방법: `run_combo_full()`을 직접 호출해(smoke_test의 pass/fail
게이트보다 더 엄격한 실제 정량 지표로) NSL-KDD 전체 데이터·전체 5
experience·CND-IDS 원 논문 epoch(20)로 6개 조합을 전부 실행했다(로컬
CPU, 총 약 12시간 — Track B의 매 라운드 K-means 재클러스터링이 병목).
Track A 90개 조합은 위 3번 리팩터가 기존 분기 결과를 그대로 보존하도록
설계했으므로 재검증 대상에서 제외했다(회귀 테스트 `pytest
testbed/tests/` 15/15 통과로 별도 확인). 96개 조합 전체에 대한
smoke_test(15.1~15.5 게이트) 재실행은 로컬 CPU로는 시간이 지나치게
오래 걸려(Track B 6개만도 12시간) 사용자가 이미 계획한 GPU 서버 이전
시점에 함께 수행하기로 한다.

## 스모크 테스트 역할 재정의 — "배선 검증"으로 축소, 행동 게이트는 경고로 (2026-09-02)

GPU 서버에서 CICIDS2018 스모크가 지나치게 오래 걸린다는 사용자 지적을
계기로 재검토했다. 원인: CICIDS2018은 중복 제거 후 약 1,208만 행이라
라운드당 train 약 193만/test 약 48만 행(NSL-KDD의 80~100배)인데,
스모크가 본 그리드와 같은 데이터·epoch로 96개 조합을 돌려 사실상 본
그리드의 복제였다 — 통과할 대다수 조합에 대해서도 같은 비용을 두 번
내는 구조라 총 계산량으로 손해다. 사용자 결정: 스모크는 "코드가 제대로
연결돼 실행되는가"만 확인하고, 조합의 실제 성능/퇴화 여부는 본 그리드
결과가 판정한다. 이에 따라 `experiments/smoke_test.py`를 세 단계로 바꿨다.

1. **epoch 축소(200/20 → 10/5, 2026-09-01)**: 15.1b/c/15.2/15.2b/15.4는
   라운드의 데이터·설정만으로 결정되고 epoch와 무관하며(15.4는 학습 루프
   이전 K-means 결과), 15.1d(발산)도 초반 몇 epoch에 드러난다.
2. **라운드당 행 수 상한(train 2만/test 5천, category별 최소 50개 보장)**:
   라운드 수와 class-incremental 구조는 그대로 두고 행 수만 NSL-KDD
   규모로 제한한다. 2026-08-26의 `af=gpm`/`af=cndids` 붕괴는 "몇 번째
   라운드에 어떤 category가 오는가"에서 나온 문제였지 데이터 양에서 나온
   게 아니라, 라운드 구조를 유지하면 그 종류의 문제는 계속 잡힌다.
   category 최소 개수는 NSL-KDD U2R 라운드(1.35만 중 52건)와 같은 "이미
   검증된 희귀 regime"인 50으로 두어 CICIDS2018의 희귀 공격이 라운드에서
   통째로 사라지지 않게 했다. 참조 캡(CADE 500/CND-IDS 5,000)·
   `cluster_fit_sample_size`(1만)도 이 규모에서 전부 발동한다. 상한보다
   작은 라운드는 손대지 않는다(NSL-KDD로 실측: exp0 59,396→20,000,
   exp1 25,125→20,000, exp2~4는 그대로, category 집합 전 라운드 보존,
   실행 간 결정적, 원본 객체 불변).
3. **행동 게이트를 경고로 강등(`SMOKE_BEHAVIORAL_GATES_AS_WARNINGS`)**:
   2번 적용 후 `A_dd=cade_ss=ssf_mm=ssf_af=lwf_ssf_as=cade_mad`가 exp0/
   exp1에서 15.2·15.2b 실패(roc_auc 0.43/0.46)했는데, 이게 조합의 문제인지
   축소의 인공물인지 분리하려고 NSL-KDD에서 4조건을 대조했다:

   | 행 수 | epoch | exp0 | exp1 | exp2 | 판정(기존 실패 기준) |
   |---|---|---|---|---|---|
   | 전체 | 200 | 통과 | 경고 0.9367 | **실패 0.9964** | 실패 |
   | 전체 | 10 | 통과 | 경고 0.9666 | **실패 0.9962** | 실패 |
   | 축소 | 200 | **실패 0.9851 / roc 0.4309** | **실패 0.9939 / 0.4871** | 경고 0.9271 | 실패 |
   | 축소 | 10 | **실패 0.9851 / roc 0.4309** | **실패 0.9865 / 0.4570** | 경고 0.9072 | 실패 |

   읽는 법: (a) 행 수 전체에서는 exp0가 통과하는데 축소하면 exp0가
   roc_auc 역전으로 실패한다 — 축소 자체가 만든 인공물이다. 이 조합은
   `dd=cade`+`as=cade_mad`라 점수가 CADE 사설 인코더에서 나오고(메인 모델
   epoch와 무관 — exp0 수치가 epoch 10/200에서 소수점까지 동일한 이유),
   라운드 행 수를 줄이면 라벨 예산 10%로 선택되는 표본이 5,940→2,000개로
   줄어 인코더(`encoder_epochs=5`, step 수가 표본 수에 비례) 학습량이
   그만큼 깎인다. (b) 반대로 행 수 전체에서 실제로 나타나는 exp2(R2L
   라운드) 붕괴(0.996)는 축소하면 경고 수준(0.91~0.93)으로 **가려진다**.
   즉 축소 설정의 행동 게이트는 양방향으로 신뢰할 수 없다 — 없는 실패를
   만들고 있는 실패를 숨긴다. 실패로 처리하면 `grid_runner.py`가 그 조합을
   본 그리드에서 제외해 조합 커버리지(이 테스트베드의 핵심 목적)를 해치므로,
   축소 설정에서는 15.2의 0.97 등급·15.2b·15.4를 경고로만 기록하고, 배선/
   수치 결함 신호인 15.1a~d·15.2 "완전 퇴화(상수 예측)"·"상수 점수"·15.3·
   15.5만 실패로 남긴다. 변경 후 같은 조합은 통과(경고 6건)하고 기본
   조합(`A_dd=none_ss=random_mm=none_af=none_as=none`)은 7.7초에 깨끗이
   통과한다. 기존 15/15 회귀 테스트 통과.

   exp2 붕괴 자체는 새 버그가 아니라 이미 문서화된 `ss=ssf`+`dd=cade`
   다중 family min-centroid 상호작용(`components/cade/cade_anomaly_scorer.py`
   "2026-08-25" 절 — threshold가 s_ref 최댓값에 clamp되어 1.0 완전 퇴화는
   막히지만 거의 전부 정상으로 판정)이다. 이제 이런 조합은 스모크에서
   걸러지지 않고 본 그리드가 낮은 F1/diag로 그대로 **기록**한다 — "이
   재조합은 희귀 공격 라운드에서 붕괴한다"는 것 자체가 테스트베드가
   내야 할 결과다.

**운영 메모**: 스모크 출력의 WARN은 정보용이다(제외 안 됨). 본 그리드
결과에서 f1/roc_auc/diag를 보고 판단한다. 전체 규모 스모크(상한 None,
실전 epoch)로 되돌리면 `SMOKE_BEHAVIORAL_GATES_AS_WARNINGS=False`로
행동 게이트를 다시 실패로 둘 수 있다. `--shard`로 나눠 돌릴 때는
서브샘플이 seed·라운드 번호로 고정되어 모든 샤드가 같은 데이터를 본다.
