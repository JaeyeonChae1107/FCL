# FCL Testbed — Federated Continual Learning for IDS

클라이언트 레벨 연속 학습(Continual Learning) 실험 프레임워크.  
SSF, CADE, CND-IDS, AnoShift의 핵심 컴포넌트를 플러그인 방식으로 조합해 실험합니다.

## 아키텍처

```
[Drift Detector] → [Sample Selector] → [Memory Manager]
     → [Anti-Forgetting] → [Anomaly Scorer]
```

각 슬롯은 독립적으로 교체 가능합니다.

| 슬롯 | 구현체 | 출처 |
|------|--------|------|
| drift_detector | ssf, cade, ddm, none | SSF/CADE/CND-IDS |
| sample_selector | ssf, random | SSF |
| memory_manager | ssf, cndids, fixed, none | SSF/CND-IDS |
| anti_forgetting | lwf_ssf, cfe, cndids, gpm, none | SSF/CND-IDS/GPM |
| anomaly_scorer | pca, cade_mad, deep_svdd, lof, isoforest | CADE/AnoShift/CND-IDS |

---

## 1. 설치 방법

```bash
pip install -r testbed/requirements.txt
```

---

## 2. 데이터 준비

### NSL-KDD
```
data/
├── PKDDTrain+.csv   (또는 KDDTrain+.csv)
└── PKDDTest+.csv    (또는 KDDTest+.csv)
```
다운로드: https://www.unb.ca/cic/datasets/nsl.html

### UNSW-NB15
```
data/
├── UNSWTrain.csv
└── UNSWTest.csv
```
다운로드: https://research.unsw.edu.au/projects/unsw-nb15-dataset

---

## 3. 단일 실험 실행

```bash
# 사전 정의된 config로 실행
python testbed/run_experiment.py --config testbed/configs/exp_ssf_full.yaml

# 실제 데이터 사용
python testbed/run_experiment.py --config testbed/configs/exp_ssf_full.yaml \
    --data_dir ./data/ --dataset nslkdd
```

---

## 4. 그리드 서치 실행

```bash
# 전체 그리드 (dummy 데이터)
python testbed/run_experiment.py --grid

# 특정 슬롯만 비교
python testbed/run_experiment.py --grid \
    --subset "anti_forgetting=gpm,lwf_ssf,cndids"

# 실제 데이터로 실행
python testbed/run_experiment.py --grid \
    --dataset nslkdd --data_dir ./data/ \
    --n_tasks 5 --label_budget 50
```

---

## 5. 결과 시각화

```bash
python testbed/run_experiment.py --visualize
```

결과물:
- `testbed/results/plots/heatmap_*.png` — 슬롯 조합별 F1 히트맵
- `testbed/results/plots/bwt_ranking.png` — BWT 순위 바 차트
- `testbed/results/plots/pareto.png` — Label efficiency vs F1 파레토 프론트

---

## 6. 스모크 테스트 및 단위 테스트

```bash
# 스모크 테스트 (5개 대표 조합)
python testbed/experiments/smoke_test.py

# 단위 테스트
python testbed/run_experiment.py --test
# 또는
pytest testbed/tests/ -v
```

---

## 7. 새 컴포넌트 추가 방법

### Step 1 — 인터페이스 상속

```python
# testbed/components/my_method/my_drift.py
from testbed.base.drift_detector import BaseDriftDetector

class MyDriftDetector(BaseDriftDetector):
    def detect(self, new_data, memory_buffer):
        ...  # 구현
        return True  # or False

    def get_drift_score(self, new_data, memory_buffer):
        ...
        return 0.0
```

### Step 2 — 레지스트리 등록

```python
# testbed/pipeline/component_registry.py 의 REGISTRY 딕셔너리에 추가
from testbed.components.my_method.my_drift import MyDriftDetector

REGISTRY = {
    "drift_detector": {
        ...
        "my_method": MyDriftDetector,   # 추가
    },
    ...
}
```

### Step 3 — Config 또는 코드에서 사용

```python
client = CLClient(
    model=my_model,
    config={
        "drift_detector": {"name": "my_method"},
        ...
    }
)
```

---

## 8. 디렉토리 구조

```
testbed/
├── base/                  # 추상 인터페이스 (5개)
├── components/
│   ├── ssf/               # SSF 컴포넌트 (KS-test, LwF, mask selector)
│   ├── cade/              # CADE 컴포넌트 PyTorch 재구현 (MAD, ContrastiveAE)
│   ├── cndids/            # CND-IDS 컴포넌트 (DDM, FIFO Memory, LwF)
│   ├── baselines/         # sklearn 기반 이상탐지기 (LOF, IsoForest, DeepSVDD)
│   └── gpm/               # GPM (Gradient Projection Memory)
├── pipeline/
│   ├── component_registry.py  # 문자열 키 → 클래스 레지스트리
│   └── cl_client.py           # CLClient 오케스트레이터
├── data/
│   └── dataset_loader.py      # NSL-KDD, UNSW-NB15 로더
├── experiments/
│   ├── metrics.py             # F1, BWT, FWT, Label Efficiency 등
│   ├── grid_runner.py         # 그리드 서치 실행기
│   ├── visualizer.py          # 히트맵, 바 차트, 파레토 플롯
│   └── smoke_test.py          # 5개 대표 조합 스모크 테스트
├── configs/                   # 실험 설정 YAML
├── results/                   # 실험 결과 JSON + summary.csv
│   └── plots/
├── tests/                     # pytest 단위 + e2e 테스트
├── run_experiment.py          # CLI 진입점
└── requirements.txt
```
