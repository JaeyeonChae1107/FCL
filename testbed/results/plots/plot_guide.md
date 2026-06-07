# FCL Testbed — 시각화 가이드 (Plot Guide)

생성 일시: 2026-06-07 15:26:02
포함 데이터셋: dummy

## 디렉토리 구조

```
results/plots/
├── nslkdd/          # NSL-KDD 데이터셋 단독 결과
├── unswnb15/        # UNSW-NB15 데이터셋 단독 결과
├── all/             # 두 데이터셋 합산 (비교 기준선)
├── compare_datasets_f1.png   # 두 데이터셋 나란히 비교
└── plot_guide.md             # 이 파일
```

---

## 각 플롯 설명

### 1. `heatmap_drift_x_af_f1.png` — F1 히트맵 (드리프트 탐지기 × 망각방지 전략)

| 항목 | 내용 |
|------|------|
| **X축** | Anti-Forgetting 전략 (`none` / `lwf_ssf` / `cfe` / `cndids` / `gpm`) |
| **Y축** | Drift Detector (`none` / `ssf` / `cade` / `ddm`) |
| **색상** | 평균 F1 점수 (진한 파란색 = 높은 F1 = 좋음) |
| **수치** | 각 셀 = 해당 조합의 모든 나머지 컴포넌트 평균 F1 |

**해석 방법**:
- 특정 행 전체가 밝다 → 해당 Drift Detector가 F1에 부정적 영향
- 특정 열 전체가 진하다 → 해당 Anti-Forgetting 전략이 탐지 성능에 유리
- 하나의 셀만 유독 진하다 → 해당 조합 쌍에 시너지 효과 존재

**IDS 문맥**: F1은 공격 탐지율(Recall)과 오탐율의 역수(Precision)의 조화평균.
NSL-KDD/UNSW-NB15처럼 클래스 불균형이 심한 데이터에서 단순 accuracy보다 신뢰할 수 있는 지표.

---

### 2. `heatmap_af_x_mm_bwt.png` — BWT 히트맵 (망각방지 전략 × 메모리 관리)

| 항목 | 내용 |
|------|------|
| **X축** | Memory Manager (`none` / `fixed` / `ssf` / `cndids`) |
| **Y축** | Anti-Forgetting 전략 |
| **색상** | 평균 BWT (진한 파란색 = 높은 BWT = 망각 적음) |

**해석 방법**:
- BWT = 마지막 태스크 학습 후 **이전 태스크들의 F1 변화량 평균**
  - BWT < 0 : 이전 공격 탐지 능력 저하 (catastrophic forgetting) → 빨간색 계열
  - BWT ≈ 0 : 이전 성능 유지 (이상적)
  - BWT > 0 : 새 학습이 이전 태스크도 개선 (backward plasticity) → 진한 파란색
- 특정 행(Anti-Forgetting)이 전반적으로 진하다 → 해당 전략이 망각 방지에 효과적

**IDS 문맥**: 새로운 공격 유형(e.g. 랜섬웨어)을 학습한 후 기존 공격(e.g. DDoS) 탐지 능력이
유지되는지를 측정한다. BWT가 크게 음수이면 모델 재학습마다 이전 공격에 무방비가 된다.

---

### 3. `heatmap_precision_recall.png` — Precision & Recall 2-패널 히트맵

| 항목 | 내용 |
|------|------|
| **왼쪽 패널** | Precision (정밀도): 이상 예측 중 실제 공격 비율 |
| **오른쪽 패널** | Recall (재현율, Detection Rate): 실제 공격 중 탐지한 비율 |
| **X축** | Anti-Forgetting 전략 |
| **Y축** | Drift Detector |

**해석 방법**:
- Precision 높고 Recall 낮음 → 알람은 정확하나 공격을 많이 놓침 (보수적 탐지)
- Precision 낮고 Recall 높음 → 공격 잘 잡지만 오탐 많음 (공격적 탐지)
- 두 패널 모두 진한 셀 → 오탐도 적고 탐지율도 높은 최선의 조합

**IDS 문맥**:
- **Precision**: 보안 운영자의 알람 피로도(alert fatigue)에 직결.
  낮은 precision → 오탐 알람 폭주 → 실제 위협 무시 위험
- **Recall**: 공격 미탐지(FN) 비율의 역수.
  낮은 recall → 실제 침입을 탐지 못함 → 보안 사고 발생

---

### 4. `bwt_ranking.png` — BWT 상위 10개 조합 막대 그래프

| 항목 | 내용 |
|------|------|
| **X축** | BWT 값 (오른쪽 길수록 좋음) |
| **Y축** | 조합 라벨 (`drift/selector/memory/forgetting/scorer` 순) |
| **파란색** | BWT ≥ 0 (망각 없음 또는 개선) |
| **빨간색** | BWT < 0 (이전 성능 저하, 망각 발생) |

**해석 방법**:
- 상단에 위치한 조합이 이전 태스크 성능 보존에 가장 우수
- 대부분 빨간색이면 전반적으로 망각 문제가 심각함을 의미
- 검은 수직선(BWT=0)이 망각/보존의 기준점

---

### 5. `pareto.png` — Label Efficiency vs F1 Pareto Front

| 항목 | 내용 |
|------|------|
| **X축** | Label Efficiency = 1 - (레이블 수/전체 샘플 수). 높을수록 레이블 절약 |
| **Y축** | F1 Score |
| **회색 점** | 모든 384개 조합의 (efficiency, F1) 좌표 |
| **빨간선** | Pareto 최적 경계선 (비지배 해집합) |

**해석 방법**:
- Pareto 선 위/오른쪽 점: 같은 F1이면 더 적은 레이블 사용, 또는 같은 비용으로 더 높은 F1
- 실용적 배포 후보는 Pareto 선상의 점들
- 선이 오른쪽 위로 뻗어있을수록 레이블 효율과 탐지 성능이 모두 우수한 조합 존재

**IDS 문맥**: 실제 환경에서 레이블링은 보안 전문가의 수작업 분석을 요구한다.
레이블 예산이 제한된 상황에서 최적의 탐지 성능을 달성하는 조합 선택에 활용한다.

---

### 6. `compare_datasets_f1.png` — NSL-KDD vs UNSW-NB15 F1 비교

| 항목 | 내용 |
|------|------|
| **X축** | Dataset (nslkdd / unswnb15) |
| **Y축** | Anti-Forgetting 전략 |
| **색상** | 평균 F1 (주황-빨강 계열, 진할수록 높음) |

**해석 방법**:
- 같은 Anti-Forgetting 전략이 두 데이터셋에서 일관되게 좋으면 강건한 전략
- 특정 데이터셋에서만 좋으면 데이터 특성 의존성이 있음
- NSL-KDD(121 features)와 UNSW-NB15(196 features)는 특성 차원과 공격 유형이 다름

---

## 평가 지표 요약표

| 지표 | 수식 | 이상값 | IDS 의미 |
|------|------|--------|---------|
| **F1** | 2·P·R/(P+R) | 1.0 | 탐지 성능 균형 요약 지표 |
| **Precision** | TP/(TP+FP) | 1.0 | 알람 정확도, 오탐(FP) 비용 |
| **Recall (DR)** | TP/(TP+FN) | 1.0 | 공격 탐지율, 미탐(FN) 비용 |
| **FPR** | FP/(FP+TN) | 0.0 | 정상 트래픽 오탐률 |
| **Balanced Acc** | (TPR+TNR)/2 | 1.0 | 클래스 불균형 보정 정확도 |
| **BWT** | Σ(R[T][j]−R[j][j])/(T−1) | 0.0 | 연속 학습 망각 정도 |
| **FWT** | Σ(R[i−1][i]−R_rand)/(T−1) | 양수 | 사전 지식 전이 능력 |
| **Label Efficiency** | 1−labeled/total | 1.0 | 레이블링 비용 절감률 |
| **Avg Inference ms** | mean(t)×1000 | 최소 | 실시간 탐지 지연 |

- R[i][j] = i번째 태스크까지 훈련 후 j번째 태스크의 F1
- Balanced Acc = (TPR + TNR) / 2, 클래스 불균형이 심할 때 단순 accuracy 대체
