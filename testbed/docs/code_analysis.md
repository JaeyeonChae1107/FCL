# FCL 코드베이스 분석 보고서

> 분석 대상: `Anoshift/`, `SSF-Strategic-Selection-and-Forgetting/`, `CADE/`, `CND-IDS/`
> 작성일: 2026-05-13

---

## 목차

1. [AnoShift](#1-anoshift)
2. [SSF (Strategic Selection and Forgetting)](#2-ssf-strategic-selection-and-forgetting)
3. [CADE](#3-cade)
4. [CND-IDS](#4-cnd-ids)
5. [기능별 비교 요약](#5-기능별-비교-요약)

---

## 1. AnoShift

### 1.1 파일 구조

```
Anoshift/AnoShift/
├── data_processor/
│   ├── data_loader.py              # Kyoto-2016 데이터셋 로딩
│   ├── generate_kyoto_tokenizer.py
│   └── parse_kyoto_logbins.py
├── language_models/
│   ├── data_utils.py               # BERT 토크나이저/데이터 준비
│   ├── model_utils.py              # BERT/Electra 학습 루프
│   ├── evaluation_utils.py         # 이상 점수 계산 (anomaly_score_bert)
│   └── tokenizer_utils.py
├── baselines_ID_setup/             # In-Distribution 평가용 베이스라인
│   ├── baseline_BERT.py
│   ├── baseline_InternalContrastiveLearning.py
│   ├── baseline_InternalContrastiveLearning/
│   │   ├── train.py                # ICL 학습 루프
│   │   ├── data_loader.py
│   │   ├── helper_functions.py     # 대조 점수 계산
│   │   ├── load_anoshift.py
│   │   └── main.py
│   ├── baseline_LOF.py
│   ├── baseline_isoforest.py
│   ├── baseline_ocsvm.py
│   ├── baseline_deep_svdd/         # Deep SVDD 구현
│   │   ├── deepSVDD.py
│   │   ├── optim/
│   │   │   ├── deepSVDD_trainer.py # SVDD 학습 + 이상 점수
│   │   │   └── ae_trainer.py
│   │   └── networks/
│   └── baselines_PyOD.py
└── baselines_OOD_setup/            # Out-of-Distribution 평가용 베이스라인
    └── (ID_setup와 동일 구조)
```

### 1.2 핵심 클래스 및 함수

| 파일 | 클래스/함수 | 역할 |
|------|------------|------|
| `data_processor/data_loader.py` | `load_kyoto_principal()` | Kyoto-2016 데이터를 연도별로 로드, IID/NEAR/FAR 분할 |
| `data_processor/data_loader.py` | `split_set()` | inlier/outlier 분리 |
| `language_models/model_utils.py` | `train_model()` | BERT MLM 학습 루프 |
| `language_models/model_utils.py` | `distil_model()` | Teacher→Student 지식 증류 |
| `language_models/evaluation_utils.py` | `anomaly_score_bert()` | MLM 기반 이상 점수 |
| `language_models/evaluation_utils.py` | `eval_rocauc_ds()` | ROC-AUC/PR-AUC 평가 |
| `baselines_OOD_setup/baseline_deep_svdd/deepSVDD.py` | `DeepSVDD` | Deep SVDD 전체 파이프라인 |
| `baselines_OOD_setup/baseline_deep_svdd/optim/deepSVDD_trainer.py` | `DeepSVDDTrainer.train()` | SVDD 손실 계산 |
| `baselines_OOD_setup/baseline_deep_svdd/optim/deepSVDD_trainer.py` | `init_center_c()` | 하이퍼스피어 중심 초기화 |
| `baselines_OOD_setup/baseline_InternalContrastiveLearning/train.py` | `trainer.train_and_evaluate()` | ICL 학습/평가 루프 |
| `baselines_OOD_setup/baseline_InternalContrastiveLearning/helper_functions.py` | `scores_calc_internal()` | ICL 대조 점수 계산 |

### 1.3 5대 기능 구현 위치

#### (1) Drift / Distribution Shift 탐지
- **명시적 탐지 모듈 없음** — AnoShift는 벤치마크 프레임워크
- 대신 **시간적 분할(temporal split)** 로 drift를 정의:
  - `data_processor/data_loader.py::load_kyoto_principal()` (line 39–61)
  - IID: 학습과 같은 연도, NEAR: 근접 연도(+2~3년), FAR: 원거리 연도(+4~8년)
  - 각 테스트 집합에서의 성능 저하 = drift 측정

#### (2) Sample Selection
- `data_processor/data_loader.py::load_kyoto_principal()` (line 77–84)
  - `contamination` 파라미터로 학습 데이터에 포함할 이상치 비율 지정
  - 기본값 `contamination=0.0` → 정상 샘플만 학습에 사용
  - Deep SVDD: `df[df['label'] == 0]` (정상만 선택)

#### (3) Memory Buffer 관리
- **미구현** — 일회성 학습 벤치마크이므로 증분 업데이트 없음

#### (4) Catastrophic Forgetting 방지
- `language_models/model_utils.py::distil_model()` (line 109–199)
  - KL 발산 기반 지식 증류: `(1-α)*L_CE + α*L_KL`
  - α=0.9 (Teacher 지식을 강하게 유지)
- BERT 파인튜닝 시 낮은 학습률(1e-5) + constant scheduler

#### (5) Anomaly / Novelty Scoring
- **BERT**: `language_models/evaluation_utils.py::anomaly_score_bert()` (line 13–55)
  ```
  이상 점수 = 1 - avg(마스킹된 토큰의 예측 신뢰도, 15회 반복)
  → 신뢰도 낮을수록 이상
  ```
- **Deep SVDD**: `deepSVDD_trainer.py` (line 130–134)
  ```
  dist = ||embedding - center||²
  anomaly_score = dist - R²  (soft-boundary)
  ```
- **ICL**: `helper_functions.py::scores_calc_internal()` + CrossEntropy loss를 이상 점수로 사용
- **LOF**: `-1 * clf.score_samples(X)` (사이킷런 표준)

### 1.4 데이터셋 형식

| 항목 | 내용 |
|------|------|
| 데이터셋 | Kyoto-2016 (허니팟 네트워크 트래픽, 2006–2015년) |
| 파일 형식 | Parquet (`{year}_subset.parquet`, ~300K 샘플/연도) |
| 입력 shape | `(N, 14)` — feature 0–13 (범주형 4개, 수치형 9개, 레이블 1개) |
| 레이블 구조 | column `"18"`: `1`=정상, `-1`=알려진 이상, `-2`=알려지지 않은 이상 |
| 학습/테스트 분할 | OOD: 학습=2006–2010, 테스트=2011–2015 |

### 1.5 학습 루프 (Pseudo-code)

```python
# 1. 데이터 로드 (정상 샘플만)
X_train = load_kyoto(years=2006~2010, label==1)

# 2. 전처리 (RobustScaler 정규화)
X_train = scaler.fit_transform(X_train)

# 3. 모델 학습
for epoch in range(num_epochs):
    for batch in DataLoader(X_train, batch_size=256):
        # BERT: MLM loss (마스킹된 토큰 예측)
        loss = MLM_loss(model(mask(batch)))
        # Deep SVDD: 거리 기반 손실
        # loss = R² + (1/nu) * max(0, ||embed - c||² - R²)
        loss.backward(); optimizer.step()

# 4. 평가 (연도별 테스트)
for year in [2006, ..., 2015]:
    X_test_inlier, X_test_outlier = load_kyoto(year)
    for x in X_test:
        score = anomaly_score(model, x)  # 방법별 상이
    roc_auc = compute_roc_auc(scores, labels)
```

---

## 2. SSF (Strategic Selection and Forgetting)

### 2.1 파일 구조

```
SSF-Strategic-Selection-and-Forgetting/
├── ssf.py      # 메인 학습 루프 (GPU 버전)
├── ssf_cpu.py  # 메인 학습 루프 (CPU 버전, ssf.py와 동일)
└── utils.py    # 모델, 손실함수, 샘플 선택, drift 탐지 모두 포함
```

### 2.2 핵심 클래스 및 함수

| 파일 | 클래스/함수 | 역할 |
|------|------------|------|
| `utils.py` | `class AE` | NSL-KDD용 오토인코더 |
| `utils.py` | `class AE_classifier` | UNSW-NB15용 AE + 분류기 헤드 |
| `utils.py` | `class InfoNCELoss` | InfoNCE 대조 손실 |
| `utils.py` | `class SplitData` | 데이터 로딩 및 레이블 처리 |
| `utils.py` | `detect_drift()` | KS-검정 기반 drift 탐지 |
| `utils.py` | `optimize_old_mask()` | KL-div로 기존 데이터 마스크 최적화 |
| `utils.py` | `optimize_new_mask()` | KL-div로 신규 데이터 마스크 최적화 |
| `utils.py` | `select_and_update_representative_samples()` | drift 없을 때 메모리 버퍼 업데이트 |
| `utils.py` | `select_and_update_representative_samples_when_drift()` | drift 감지 시 메모리 버퍼 업데이트 |
| `utils.py` | `evaluate()` | 코사인 유사도 + GMM 기반 이상 점수 |
| `utils.py` | `evaluate_classifier()` | UNSW용 분류기 평가 |
| `utils.py` | `score_detail()` | Accuracy/Precision/Recall/F1 출력 |
| `ssf.py` | (메인 스크립트) | 전체 학습/온라인 학습 루프 오케스트레이션 |

### 2.3 5대 기능 구현 위치

#### (1) Drift / Distribution Shift 탐지
- **파일**: `utils.py` / **함수**: `detect_drift()` (line 646–658)
- **알고리즘**: Kolmogorov-Smirnov 2-sample test
  ```python
  ks_statistic, p_value = ks_2samp(control_data, window_data)
  if p_value < drift_threshold (=0.05): return True  # drift 감지
  ```
- **호출 위치**: `ssf.py` line 211 (NSL-KDD), line 225 (UNSW-NB15)
- 입력: train/test 데이터의 모델 출력값 분포 (NSL=정규화 확률, UNSW=분류 로짓)

#### (2) Sample Selection (어떤 샘플을 학습에 쓸지 결정)
- **파일**: `utils.py` / **함수**: `optimize_old_mask()` (line 109), `optimize_new_mask()` (line 147)
  - KL 발산 최소화로 기존/신규 데이터의 대표성 마스크 M_c, M_t 최적화
  - `M >= 0.5` 인 샘플을 대표 샘플로 선택
- **파일**: `utils.py` / **함수**: `select_and_update_representative_samples()` (line 192)
  - drift 없을 때: M_c 기반 기존 샘플 유지 + M_t 기반 신규 샘플 추가
- **파일**: `utils.py` / **함수**: `select_and_update_representative_samples_when_drift()` (line 259)
  - drift 있을 때: 위와 동일 + 버퍼 여유 공간에 pseudo-labeled 신규 샘플 추가

#### (3) Memory Buffer 관리
- **파일**: `utils.py` / **함수**: `select_and_update_representative_samples_when_drift()` (line 259)
- **관리 방식**:
  - 버퍼 크기: `memory = floor(N_train * (1-percent))` (기본 20% 학습 데이터 크기)
  - **삭제**: non-representative 기존 샘플 제거 (`M_c < 0.5` 인 것부터), 수가 부족하면 M_c 낮은 대표 샘플도 제거
  - **추가**: 신규 데이터에서 M_t 점수 높은 샘플 `num_labeled_sample`개 추가 (true label 제공)
  - **버퍼 보충**: drift 시 버퍼 여유 공간에 pseudo-label 샘플 추가 (line 325–388)
  - x_train_this_epoch가 곧 메모리 버퍼 역할 수행

#### (4) Catastrophic Forgetting 방지
- **파일**: `ssf.py` / **위치**: line 323–334 (drift 없을 때 학습 루프 내)
- **방법**: Learning without Forgetting (LwF)
  ```python
  # Teacher model로부터 distillation loss 계산
  teacher_recon_vec = teacher_model(inputs)
  distillation_loss = F.mse_loss(recon_vec, teacher_recon_vec)
  total_loss = weighted_loss + lwf_lambda * distillation_loss  # λ=0.5
  ```
- **drift 감지 시** (line 262–291): distillation loss 없이 가중 손실만 사용
  - 이유: drift 시에는 빠른 적응이 우선시됨
- Teacher 모델은 매 라운드 종료 후 갱신: `teacher_model.load_state_dict(model.state_dict())`

#### (5) Anomaly / Novelty Scoring (NSL-KDD 기준)
- **파일**: `utils.py` / **함수**: `evaluate()` (line 556–643)
- **알고리즘**:
  ```python
  # 1. 정상 샘플의 재구성 벡터 평균 → normal_recon_temp
  # 2. 전체 샘플과 normal_recon_temp 간 코사인 유사도 계산
  values = cosine_similarity(normalize(recon_vec), normal_recon_temp)
  # 3. 정상/이상 분포에 각각 가우시안 피팅 (MLE)
  result = minimize(log_likelihood, initial_params, args=(values,))
  # 4. 테스트 샘플: pdf_normal > pdf_abnormal이면 정상 (0), 아니면 이상 (1)
  y_pred = (pdf_abnormal > pdf_normal).astype(int)
  ```
- **UNSW-NB15 기준**: `evaluate_classifier()` (line 66) — sigmoid > 0.5이면 이상

### 2.4 데이터셋 형식

| 항목 | NSL-KDD | UNSW-NB15 |
|------|---------|-----------|
| 파일 | `NSL_pre_data/PKDDTrain+.csv` | `UNSW_pre_data/UNSWTrain.csv` |
| 입력 차원 | 121 | 196 |
| 레이블 | binary: 0=정상, 1=공격 | binary: 0=정상, 1=공격 |
| 전처리 | MinMaxScaler 정규화 | MinMaxScaler 정규화 |
| 텐서 타입 | `FloatTensor`, `LongTensor` | 동일 |

### 2.5 학습 루프 (Pseudo-code)

```python
# 초기 학습 (20% 데이터로)
for epoch in range(epochs=4):
    for batch in DataLoader(x_train_init):
        features, recon_vec = model(inputs)
        loss = InfoNCE(recon_vec, labels)  # 대조 손실
        loss.backward(); optimizer.step()

teacher_model.load_state_dict(model.state_dict())

# 온라인 학습 (슬라이딩 윈도우, window_size=20000)
while new_data_available:
    x_new = next_window()

    # drift 탐지
    drift = detect_drift(model_output(x_new), model_output(x_train_buffer))

    # 마스크 최적화 → 대표 샘플 선택
    M_c = optimize_old_mask(train_dist, new_dist)
    M_t = optimize_new_mask(train_dist, new_dist, M_c)

    if drift:
        x_buffer, y_buffer = select_and_update_representative_samples_when_drift(...)
        # distillation 없이 weighted loss만 사용
        loss = weighted_InfoNCE(recon_vec, labels, mask=new_sample_mask)
    else:
        x_buffer, y_buffer = select_and_update_representative_samples(...)
        # LwF distillation 추가
        distill_loss = MSE(recon_vec, teacher_model(inputs))
        loss = weighted_InfoNCE(...) + 0.5 * distill_loss

    loss.backward(); optimizer.step()
    teacher_model.load_state_dict(model.state_dict())

    # 이상 탐지 평가
    scores = evaluate(normal_recon_temp, x_buffer, y_buffer, x_new, model)
```

---

## 3. CADE

### 3.1 파일 구조

```
CADE/
├── main.py                                         # 8단계 파이프라인 진입점
├── setup.py
├── average_all_detection_results.py
├── IDS_data_preprocess/
│   ├── clean_data.py
│   └── gen_IDS_data.py
└── cade/
    ├── __init__.py
    ├── autoencoder.py                              # Autoencoder, ContrastiveAE
    ├── classifier.py                              # MLPClassifier, RFClassifier
    ├── config.py                                  # 경로 설정
    ├── data.py                                    # 데이터 로딩, 배치 페어링
    ├── detect.py                                  # drift 탐지 (MAD 기반)
    ├── evaluate.py                                # 탐지 성능 평가
    ├── explain_by_distance.py                     # 거리 기반 설명
    ├── explain_global_approximation_loose_boundary.py
    ├── logger.py
    ├── mask_exp_by_approximation.py
    ├── mask_exp_by_distance_mask_m1.py
    └── utils.py                                   # argparse 및 유틸
```

### 3.2 핵심 클래스 및 함수

| 파일 | 클래스/함수 | 역할 |
|------|------------|------|
| `cade/autoencoder.py` | `class Autoencoder` | 대칭형 FC 오토인코더 (Keras/TF1) |
| `cade/autoencoder.py` | `class ContrastiveAE` | 대조 손실 + MSE 재구성 AE |
| `cade/autoencoder.py` | `ContrastiveAE.train()` | 대조 손실 학습 루프 |
| `cade/classifier.py` | `class MLPClassifier` | 다층 퍼셉트론 분류기 (Keras) |
| `cade/classifier.py` | `class RFClassifier` | Random Forest 분류기 |
| `cade/data.py` | `load_features()` | npz에서 X_train/y_train/X_test/y_test 로드 |
| `cade/data.py` | `epoch_batches()` | 대조 학습용 Similar/Dissimilar 페어 배치 생성 |
| `cade/detect.py` | `detect_drift_samples()` | MAD 기반 이상 탐지 (메인 탐지 함수) |
| `cade/detect.py` | `get_latent_representation_keras()` | 인코더로 잠재 표현 추출 |
| `cade/detect.py` | `get_MAD_for_each_family()` | 패밀리별 MAD 계산 |
| `cade/detect.py` | `get_latent_distance_between_sample_and_centroid()` | 거리 계산 |
| `cade/evaluate.py` | `evaluate_newfamily_as_drift_by_distance()` | 탐지 성능(PR-AUC) 평가 |

### 3.3 5대 기능 구현 위치

#### (1) Drift / Distribution Shift 탐지
- **파일**: `cade/detect.py` / **함수**: `detect_drift_samples()` (line 45–105)
- **알고리즘**: MAD (Median Absolute Deviation) 기반 이상 점수
  ```python
  # 학습 시: 패밀리별 centroid, MAD 계산
  centroid[i] = mean(z_family[i])
  dis[i][j] = ||z_family[i][j] - centroid[i]||₂
  MAD[i] = 1.4826 * median(|dis[i] - median(dis[i])|)

  # 추론 시: 테스트 샘플 이상 점수
  dist_k = ||z_k - centroid[i]||₂  # 각 패밀리까지 거리
  anomaly_k = |dist_k - median(dis[i])| / MAD[i]
  is_drift = min(anomaly_k) > mad_threshold (기본=3.5)
  ```

#### (2) Sample Selection
- **파일**: `cade/data.py` / **함수**: `epoch_batches()` (line 268–344)
- **방법**: 에폭마다 동적으로 Similar/Dissimilar 페어 생성
  - 배치의 전반부: 랜덤 샘플
  - 후반부 25%: 같은 레이블의 Similar 샘플
  - 후반부 75%: 다른 레이블의 Dissimilar 샘플

#### (3) Memory Buffer 관리
- **미구현** — CADE는 정적 일회성 학습 방식
- 대신 탐지에 필요한 통계를 `.npz` 파일로 저장:
  ```python
  np.savez_compressed(path, z_train, z_family, centroids, dis_family, mad_family)
  ```

#### (4) Catastrophic Forgetting 방지
- **파일**: `cade/autoencoder.py` / **함수**: `ContrastiveAE.train()` (line 220–232)
- **방법**: 대조 손실이 패밀리 간 표현 구분을 강제하여 간접적 망각 방지
  ```python
  dist = ||z_i - z_j||₂
  contrastive_loss = is_same * dist + (1 - is_same) * max(0, margin - dist)
  total_loss = lambda_1 * contrastive_loss + reconstruction_loss  # λ=0.1
  ```
- 명시적 LwF/EWC 없음 — 스트리밍 학습 미지원

#### (5) Anomaly / Novelty Scoring
- **파일**: `cade/detect.py` / **위치**: line 89–97
  ```python
  dis_k[i] = ||z_k - centroids[i]||₂
  anomaly_k[i] = |dis_k[i] - median(dis_family[i])| / MAD[i]
  min_anomaly_score = min(anomaly_k)
  ```

### 3.4 데이터셋 형식

| 항목 | 내용 |
|------|------|
| 데이터셋 | Drebin (Android 악성코드) / IDS-2018 (네트워크 침입) |
| 파일 형식 | NumPy `.npz`: `X_train`, `y_train`, `X_test`, `y_test` |
| 입력 shape | `(N, num_features)` — Drebin: ~1300 이진 특성, IDS: 83 수치 특성 |
| 레이블 구조 | 다중 클래스 정수 (0~num_classes-1), 미지 패밀리 = 별도 레이블 |
| 학습/테스트 | 학습: 알려진 패밀리, 테스트: 알려진 + 미지 패밀리 포함 |

### 3.5 학습 루프 (Pseudo-code)

```python
# 1. 분류기 학습 (MLP or RF)
mlp.train(X_train, y_train)
y_pred = mlp.predict(X_test)

# 2. 대조 오토인코더 학습
for epoch in range(250):
    batch_x, batch_y = epoch_batches(X_train, y_train, batch_size=64, similar_ratio=0.25)
    for batch in zip(batch_x, batch_y):
        z = encoder(batch)
        dist = ||z[left] - z[right]||₂
        contrastive_loss = is_same * dist + (1-is_same) * max(0, margin - dist)
        recon_loss = MSE(batch, decoder(z))
        loss = 0.1 * contrastive_loss + recon_loss
        loss.backward(); optimizer.step()

# 3. 탐지 통계 계산
z_train = encoder(X_train)
for family in unique_families:
    centroid[f] = mean(z_train[y==f])
    dis[f] = [||z - centroid[f]||₂ for z in z_train[y==f]]
    MAD[f] = 1.4826 * median(|dis[f] - median(dis[f])|)

# 4. 테스트 샘플 이상 탐지
z_test = encoder(X_test)
for z_k in z_test:
    anomaly_score[k] = min([|||z_k - c[f]||₂ - median(dis[f])| / MAD[f] for f in families])
    is_drift[k] = anomaly_score[k] > 3.5
```

---

## 4. CND-IDS

### 4.1 파일 구조

```
CND-IDS/
├── main.py                         # 실험 진입점 (fit_and_test 오케스트레이션)
├── run_experiments.py              # 실험 설정 (feature extractor × anomaly detector × dataset)
├── datastream.py                  # 데이터셋 로딩 및 경험(experience) 분할
├── utils.py                       # 데이터셋 유틸
├── metrics.py                     # 평가 지표
├── logger_config.py
├── AutonomousDCN/
│   ├── ADCNbasic.py               # ADCN 핵심 모델 (drift 탐지, 레이어 성장, LwF)
│   ├── ADCNmainloop.py            # ADCN 메인 학습 루프
│   ├── model.py                   # simpleMPL 등 기본 신경망
│   └── utilsADCN.py               # 유틸
├── FeatureExtractors/
│   ├── CFE.py                     # Continual Feature Extractor (ADCN 기반, 메모리 포함)
│   ├── CND_IDS.py                 # CND-IDS 피처 추출기 (metric loss + LwF)
│   ├── AE_Exactor.py              # AE 피처 추출기 (LwF 포함)
│   ├── EaM.py                     # Error-aware Metric learning
│   ├── LwFRecon.py                # LwF + 재구성 손실
│   ├── PassThroughExtractor.py
│   └── modules/
│       ├── ADCNbasic.py           # ADCN 모듈 (복사본)
│       ├── memory.py              # Memory 클래스 (FIFO, Perfect, PSA)
│       ├── sampler.py             # NNBatchSampler (최근접 이웃 배치)
│       ├── loss.py                # RC_STML, KL_STML, STML_loss, Momentum_Update
│       ├── model.py               # 기본 모델
│       └── K_Means.py             # Elbow 기반 K-Means
└── AnomolyDetectors/
    ├── AE.py                      # 재구성 오류 기반
    ├── DIF.py                     # Deep Isolation Forest
    ├── DNN.py                     # DNN 이상 탐지기
    ├── ICL.py                     # Internal Contrastive Learning
    ├── K_Means.py                 # K-Means 클러스터링
    ├── PCA.py                     # PCA 기반
    ├── Random.py                  # 랜덤 베이스라인
    └── SLAD.py                    # Sparse Labeling Anomaly Detection
```

### 4.2 핵심 클래스 및 함수

| 파일 | 클래스/함수 | 역할 |
|------|------------|------|
| `AutonomousDCN/ADCNbasic.py` | `class ADCN` | 동적 계층 성장 네트워크 (drift → 레이어 추가) |
| `AutonomousDCN/ADCNbasic.py` | `ADCN.driftDetection()` | DDM 기반 drift 탐지 (3-상태: 안정/경고/drift) |
| `AutonomousDCN/ADCNbasic.py` | `ADCN.layerGrowing()` | Drift 확정 시 새 레이어 추가 |
| `AutonomousDCN/ADCNbasic.py` | `ADCN.fit()` | 재구성 손실로 전체 네트워크 학습 |
| `AutonomousDCN/ADCNbasic.py` | `ADCN.fitCL()` | LwF 기반 연속 학습 |
| `AutonomousDCN/ADCNbasic.py` | `ADCN.LwFloss()` | LwF 손실 계산 |
| `AutonomousDCN/ADCNbasic.py` | `class cluster` | K-Means 기반 클러스터링, 새 패턴 탐지 |
| `AutonomousDCN/ADCNbasic.py` | `cluster.detectNovelPattern()` | 클러스터 성장 (novelty 탐지) |
| `FeatureExtractors/CFE.py` | `class CFE` | ADCN + Memory 결합 피처 추출기 |
| `FeatureExtractors/CFE.py` | `CFE.fit()` | 배치 학습 (drift 탐지 + 메모리 리플레이 포함) |
| `FeatureExtractors/CND_IDS.py` | `class CND_IDS` | Triplet loss + LwF + 재구성 피처 추출기 |
| `FeatureExtractors/CND_IDS.py` | `CND_IDS.LwFloss()` | 이전 태스크 대비 LwF 손실 |
| `FeatureExtractors/modules/memory.py` | `class Memory` | FIFO/Perfect/PSA 메모리 버퍼 |
| `FeatureExtractors/modules/sampler.py` | `class NNBatchSampler` | 최근접 이웃 배치 샘플링 |
| `FeatureExtractors/modules/loss.py` | `class STML_loss` | RC_STML + KL_STML 복합 손실 |
| `FeatureExtractors/modules/loss.py` | `class Momentum_Update` | Teacher 모델 모멘텀 업데이트 |
| `main.py` | `fit_and_test()` | 피처 추출 → 이상 탐지기 학습 → 평가 루프 |
| `main.py` | `test_experiences()` | 모든 테스트 experience에 대해 점수 계산 |
| `datastream.py` | `class datastream` | 경험(experience) 단위 데이터 분할 및 정규화 |

### 4.3 5대 기능 구현 위치

#### (1) Drift / Distribution Shift 탐지
- **파일**: `AutonomousDCN/ADCNbasic.py` / **함수**: `ADCN.driftDetection()` (line ~547–652)
- **알고리즘**: DDM (Drift Detection Method) — CNN 피처 분포 비교
  - 3-상태 출력: `driftStatus` = 0(안정), 1(경고), 2(drift 확정)
  - 배치 간 피처 분포 평균 비교 + 통계적 경계(α_drift) 초과 시 drift 판정
- **호출 위치**: `FeatureExtractors/CFE.py::CFE.fit()` (line 49)

#### (2) Sample Selection (어떤 샘플을 학습에 쓸지 결정)
- **파일**: `FeatureExtractors/modules/sampler.py` / **클래스**: `NNBatchSampler`
  - `_build_nn_matrix()` (line 57–95): 전체 데이터의 K-최근접 이웃 행렬 계산
  - `sample_batch()` (line 98–103): 쿼리 샘플 + 그 이웃들로 배치 구성 (다양성 확보)
- **파일**: `datastream.py` / **함수**: `dataADCNLoader.createTask()`
  - `ADCN_label_mode`에 따라 레이블 샘플 선택 (random / first-experience-only)
  - `nEachClassSamples` 개수만큼 클래스별 계층적 샘플링

#### (3) Memory Buffer 관리
- **파일**: `FeatureExtractors/modules/memory.py` / **클래스**: `Memory`
  - **초기화**: `Memory(mode, capacity=1000, datastream, device)`
  - **저장/업데이트** (`update()`, line 19–40):
    - **Perfect** 모드: 공격 유형별 균등 샘플 수 저장 (capacity/num_attacks 개/유형)
    - **FIFO** 모드: 용량 초과 시 가장 오래된 샘플 제거, 새 데이터 추가
    - **PSA** 모드: 미구현
  - **조회** (`get_memory()`, line 43–45)
- **호출 위치**: `FeatureExtractors/CFE.py::CFE.fit()` (line 47, 71)
  - 배치 처리 시 메모리 데이터를 현재 배치에 concat하여 리플레이

#### (4) Catastrophic Forgetting 방지
- **파일**: `AutonomousDCN/ADCNbasic.py` / **함수**: `ADCN.LwFloss()` (line 535–545), `ADCN.fitCL()` (line 380–428)
  - 이전 태스크 네트워크(ADCNold) 출력과 현재 출력 간 BCELoss
  - 태스크 가중치: `regStr = regStrLWF * (1 - nOutputPerTask / ((iTask+1+1)*nOutputPerTask))`
- **파일**: `FeatureExtractors/CND_IDS.py` / **함수**: `CND_IDS.LwFloss()` (line 54–69)
  - 이전 모델 리스트(old_models)로부터 각 태스크의 MSE distillation loss 계산
  - `total_LwF_loss += reg_strength * MSE(current_output, old_model(input))`
- **파일**: `FeatureExtractors/CFE.py::CFE.fit()` (line 67–68)
  - `model.fitCL(x_batch, reconsLoss=True)` — 이전 태스크 존재 시 호출

#### (5) Anomaly / Novelty Scoring
- **파일**: `main.py` / **함수**: `test_experiences()` (line 324–356)
  ```python
  encoded_X_test = feature_extractor(X_test)
  scores = anomaly_detector.predict(encoded_X_test)
  ```
- 점수 계산은 선택된 이상 탐지기에 위임:
  - **PCA**: `decision_function(X)` (재구성 오류)
  - **AE**: MSE 재구성 오류
  - **DIF/SLAD/ICL**: deepod 라이브러리 `decision_function()`
  - **K-Means**: 클러스터 중심까지 거리
- **파일**: `AutonomousDCN/ADCNbasic.py` / **클래스**: `cluster.predict()` (line 195–205)
  - Allegiance(퍼지 멤버십) 가중 합산:
    `score = Σ exp(-dist(centroid_i, x)) × allegiance[i]`

### 4.4 데이터셋 형식

| 항목 | 내용 |
|------|------|
| 데이터셋 | EdgeIIoT, XIIoT, MQTT, WUST, UNSW, CICIDS17/18 등 |
| 입력 shape | `(N, nFeatures)` — 데이터셋마다 상이 (33~95차원) |
| 레이블 구조 | binary: 0=정상, 1=공격 (내부적으로 다중 클래스 사용) |
| 저장 형식 | NumPy `.npy` (`x.npy`, `y.npy`) |
| 경험 분할 | train_experiences / test_experiences 리스트 (연속학습용) |
| 정규화 | Min-Max (0~1 범위) |
| 특수 분할 | `init_normal` (90% 정상): 이상 탐지기 학습용, `init_val` (10%): 임계값 설정용 |

### 4.5 학습 루프 (Pseudo-code)

```python
# fit_and_test() 기준 (CFE + 이상 탐지기)
for i, (X, y) in enumerate(train_experiences):

    # --- 피처 추출기 학습 (CFE) ---
    feature_extractor.fit(X, device):
        model.storeOldModel(experience_num)  # LwF용 이전 모델 저장
        for batch in X:
            if memory: batch = concat(batch, memory.get_memory())
            model.driftDetection(batch, prev_batch)  # DDM drift 탐지
            if driftStatus == 2:                     # drift 확정
                model.layerGrowing()                 # 새 레이어 추가
                model.initialization(batch)          # 초기화
            if driftStatus in [0, 2]:               # 안정 or drift
                model.fit(batch)                     # 재구성 손실 학습
                if len(old_models) > 0:
                    model.fitCL(batch, reconsLoss=True)  # LwF 손실 학습
            memory.update(batch)                     # 메모리 업데이트

    # --- 이상 탐지기 학습 ---
    encoded_normal = feature_extractor(init_normal)
    anomaly_detector.fit(encoded_normal)

    # --- 모든 test experience 평가 ---
    for j, (X_test, y_test) in enumerate(test_experiences):
        encoded_test = feature_extractor(X_test)
        scores = anomaly_detector.predict(encoded_test)
        metrics = compute_metrics(y_test, scores, scores_val)
        results.append(metrics)
```

---

## 5. 기능별 비교 요약

| 기능 | AnoShift | SSF | CADE | CND-IDS |
|------|----------|-----|------|---------|
| **Drift 탐지** | 시간적 분할로 암묵적 정의 (미탐지) | KS-test (p-value < 0.05) | MAD 기반 이상 점수 (잠재 공간) | DDM (배치 간 피처 분포 비교, 3-상태) |
| **탐지 파일/함수** | `data_loader.py::load_kyoto_principal()` | `utils.py::detect_drift()` | `detect.py::detect_drift_samples()` | `ADCNbasic.py::ADCN.driftDetection()` |
| **Sample Selection** | contamination 기반 랜덤 샘플링 | M_c/M_t 마스크 KL-div 최적화 | Similar/Dissimilar 페어 동적 생성 | NNBatchSampler (최근접 이웃 배치) |
| **선택 파일/함수** | `data_loader.py` (`.sample()`) | `utils.py::optimize_old/new_mask()` | `data.py::epoch_batches()` | `modules/sampler.py::NNBatchSampler` |
| **Memory Buffer** | 미구현 | x_train_this_epoch (크기=20% 학습 데이터) | 미구현 (탐지 통계만 .npz 저장) | `modules/memory.py::Memory` (FIFO/Perfect) |
| **버퍼 업데이트** | — | `select_and_update_representative_samples_when_drift()` | — | `memory.update(new_data, curr_experience)` |
| **Catastrophic Forgetting** | Knowledge Distillation (KL) | LwF — MSE distillation (λ=0.5) | 대조 손실로 간접 보호 | LwF — BCELoss (ADCN) + MSE (CND_IDS) |
| **CF 파일/함수** | `model_utils.py::distil_model()` | `ssf.py` (line 326–331) | `autoencoder.py::ContrastiveAE.train()` | `ADCNbasic.py::ADCN.LwFloss()`, `CND_IDS.py::LwFloss()` |
| **Anomaly Scoring** | MLM 신뢰도 (BERT) / 거리 (SVDD/LOF) | 코사인 유사도 + GMM 분류 | MAD 정규화 거리: `(dist-median)/MAD` | 이상 탐지기 위임 (PCA/AE/DIF/SLAD 등) |
| **점수 파일/함수** | `evaluation_utils.py::anomaly_score_bert()` | `utils.py::evaluate()` | `detect.py::detect_drift_samples()` (line 89–97) | `main.py::test_experiences()` → `a_s_m.predict()` |
| **데이터셋** | Kyoto-2016 (14 features, binary) | NSL-KDD (121) / UNSW-NB15 (196) | Drebin (~1300) / IDS-2018 (83) | EdgeIIoT/CICIDS 등 (33~95 features) |
| **프레임워크** | TF2 / scikit-learn | PyTorch | Keras (TF1) | PyTorch |
| **학습 패러다임** | 일회성 배치 학습 | 슬라이딩 윈도우 온라인 학습 | 일회성 배치 학습 | 연속 학습 (experience 단위) |
