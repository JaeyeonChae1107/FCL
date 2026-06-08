# FCL Testbed — 시각화 가이드 (Plot Guide)

생성 일시: 2026-06-08 10:18:48
포함 데이터셋: dummy

## 디렉토리 구조

```
results/plots/
├── nslkdd/                    # NSL-KDD 데이터셋 단독 결과
├── unswnb15/                  # UNSW-NB15 데이터셋 단독 결과
├── all/                       # 두 데이터셋 합산 (비교 기준선)
├── compare_datasets_f1.png    # 두 데이터셋 나란히 비교
└── plot_guide.md              # 이 파일
```

---

## 각 플롯 설명

### 1. `heatmap_drift_x_af_f1.png` — F1 히트맵 (드리프트 탐지기 × 망각방지 전략)

| 항목 | 내용 |
|------|------|
| **X축** | Anti-Forgetting 전략 (`none` / `lwf_ssf` / `cndids` / `gpm`) |
| **Y축** | Drift Detector (`none` / `ssf` / `cade`) |
| **색상** | 평균 F1 점수 (진한 파란색 = 높은 F1 = 좋음) |
| **수치** | 각 셀 = 해당 조합의 모든 나머지 컴포넌트 평균 F1 |

**해석 방법**:
- 특정 행 전체가 밝다 → 해당 Drift Detector가 F1에 부정적 영향
- 특정 열 전체가 진하다 → 해당 Anti-Forgetting 전략이 탐지 성능에 유리
- 하나의 셀만 유독 진하다 → 해당 조합 쌍에 시너지 효과 존재

**IDS 문맥**: F1은 공격 탐지율(Recall)과 알람 정확도(Precision)의 조화평균.
NSL-KDD/UNSW-NB15처럼 클래스 불균형이 심한 데이터에서 단순 accuracy보다 신뢰할 수 있는 지표.

---

### 2. `heatmap_precision_recall.png` — Precision & Recall 2-패널 히트맵

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

### 3. `heatmap_drift_x_af_fpr.png` — FPR 히트맵 (드리프트 탐지기 × 망각방지 전략)

| 항목 | 내용 |
|------|------|
| **X축** | Anti-Forgetting 전략 |
| **Y축** | Drift Detector |
| **색상** | 평균 FPR (밝을수록 낮은 FPR = 좋음) |

**해석 방법**:
- FPR = FP / (FP + TN): 정상 트래픽 중 공격으로 잘못 예측한 비율
- 셀 값이 낮을수록 오탐이 적은 조합
- Recall 히트맵과 함께 보면 탐지율-오탐률 트레이드오프 파악 가능

**IDS 문맥**: 높은 FPR은 SOC(보안 운영 센터) 분석가의 알람 피로를 유발하여
실제 공격 경보가 묻힐 위험이 있다.

---

### 4. `recall_ranking.png` — Recall 상위 10개 조합 막대 그래프

| 항목 | 내용 |
|------|------|
| **X축** | Recall (Detection Rate) 값 (오른쪽 길수록 좋음) |
| **Y축** | 조합 라벨 (`drift/selector/memory/forgetting/scorer` 순) |
| **파란색** | Recall ≥ 0.7 (권장 탐지율 이상) |
| **주황색** | Recall < 0.7 (낮은 탐지율 경고) |

**해석 방법**:
- 상단에 위치한 조합이 공격 탐지율이 가장 높은 최선의 조합
- 회색 점선(0.7 기준선)을 넘는 조합이 실용적 배포 후보
- FPR 히트맵과 함께 봐야 오탐 여부도 확인 가능

---

### 5. `recall_fpr_tradeoff.png` — Recall vs FPR 트레이드오프

| 항목 | 내용 |
|------|------|
| **X축** | FPR (낮을수록 좋음, 좌측) |
| **Y축** | Recall (높을수록 좋음, 상단) |
| **점 색상** | F1 점수 (진할수록 높음) |
| **빨간선** | Pareto 최적 경계선 (비지배 해집합) |

**해석 방법**:
- 좌상단에 위치한 점 = 오탐 적으면서 탐지율도 높은 이상적 조합
- Pareto 선상의 점들이 실용적 배포 후보
- 선이 좌상단 모서리에 가까울수록 전체 조합의 품질이 우수함

**IDS 문맥**: 보안 정책에 따라 Recall 우선(공격 놓치지 않기) 또는 FPR 우선(오탐 최소화)
중 하나를 선택할 수 있으며, 이 플롯이 그 트레이드오프를 한눈에 보여준다.

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
| **Recall** | TP/(TP+FN) | 1.0 | 공격 탐지율, 미탐(FN) 비용 |
| **FPR** | FP/(FP+TN) | 0.0 | 정상 트래픽 오탐률 |
