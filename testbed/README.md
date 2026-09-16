# CL-NIDS-Bench

5-슬롯(drift_detector / sample_selector / memory_manager / anti_forgetting /
anomaly_scorer) 조합으로 SSF·CADE·SPIDER(+GPM)·CND-IDS 4개 논문의 연속학습
메커니즘을 통제된 조건에서 비교하는 벤치마크. 신규 방법을 제안하는 논문이
아니라 체계적 비교 연구다 — 배경·설계 결정 전체는
[`docs/design_decisions.md`](docs/design_decisions.md) 참고(코드 주석이 인용하는
"PRD"/"수정 계획" 문서는 이 저장소에 없으며, 그 자리를 이 문서가 대신한다).

## 설치

```
pip install -r testbed/requirements.txt
```

GPU를 쓰려면 위 설치 후 대상 머신의 CUDA 버전에 맞는 PyTorch를 한 번 더
설치해 덮어쓴다(`requirements.txt` 상단 주석 참고):

```
pip install torch --index-url https://download.pytorch.org/whl/cu121   # cu121은 예시
```

## 데이터 준비

원본 데이터 파일은 이 저장소에 포함돼 있지 않다 — 저장소 루트(이 README
기준 상위 폴더) 아래 다음 위치에 직접 받아 둘 것:

| 데이터셋 | 경로 | 비고 |
|---|---|---|
| NSL-KDD | `SSF-Strategic-Selection-and-Forgetting/NSL_pre_data/{PKDDTrain+.csv,PKDDTest+.csv}` | SSF 저장소 전처리 버전 |
| UNSW-NB15 | `SSF-Strategic-Selection-and-Forgetting/UNSW_pre_data/{UNSWTrain.csv,UNSWTest.csv}` | SSF 저장소 전처리 버전 |
| UNSW-NB15 공식 원본 | `UNSW-NB15-raw/{UNSW_NB15_training-set.csv,UNSW_NB15_testing-set.csv}` | `attack_cat`(다중클래스) 결합용, [UNSW 공식 페이지](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |
| X-IIoTID | `X-IIoTID-raw/X-IIoTID dataset.csv` | [IEEE DataPort](https://ieee-dataport.org/documents/x-iiotid-connectivity-agnostic-and-device-agnostic-intrusion-data-set-industrial) 원본 그대로(이 테스트베드가 직접 전처리) |

경로가 틀리면 `data/dataset_loader.py`가 정확한 위치를 알려주는
`FileNotFoundError`를 던진다. 최초 로드 시 전처리 결과를
`testbed/data/.cache/`에 캐싱한다(코드/인자가 바뀌면 자동으로 무효화).

## 실행 순서

전부 저장소 루트에서 `python -m testbed.experiments.<모듈>` 형태로 실행한다.

### 1) 스모크 테스트 (그리드 실행 전 필수 게이트)

```
python -m testbed.experiments.smoke_test --datasets nsl-kdd,unsw-nb15 --device cuda
```

`enumerate_valid_combos()`의 78개 조합 전부를 축소 설정(행 수 캡·짧은
epoch)으로 검증해 `testbed/experiments/smoke_test_results.json`에 저장한다.
`--datasets`에 `x-iiotid`를 추가하려면 명시적으로 넣을 것(기본값은
NSL-KDD/UNSW-NB15만). 코드를 고치면 캐시된 통과 판정이 자동으로 무효화된다
(code_version 불일치 검사). 여러 GPU/프로세스로 나눠 돌리려면
`--shard i/n`을 각 프로세스에 다르게 주고, 전부 끝난 뒤
`--merge-shards n`으로 합친다.

### 2) 정식 그리드 실행

```
python -m testbed.experiments.grid_runner --datasets nsl-kdd,unsw-nb15 --device cuda
```

스모크를 통과한 조합만 전체 데이터·정식 epoch(Track A 200/Track B 20)으로
실행해 `testbed/results/{combo_id}__{dataset}.json`을 생성한다. 데이터셋당
Joint/Offline 상한선(`JOINT_BASELINE__{dataset}.json`)도 함께 자동 계산된다
(끄려면 `--skip-joint-baseline`). 주요 플래그:

- `--track A|B` — 특정 Track만 재실행.
- `--shard i/n` — 여러 프로세스로 조합을 나눠 동시 실행(스모크와 별도 개념 —
  merge 불필요, 결과 파일이 조합별로 이미 분리돼 있음). **멀티 GPU
  노드에서는 프로세스마다 `CUDA_VISIBLE_DEVICES`를 GPU 하나로 고정할 것**
  (예: `CUDA_VISIBLE_DEVICES=0 python -m testbed.experiments.grid_runner
  --shard 0/4 --device cuda ...`) — 한 프로세스에 여러 GPU가 동시에 보이면
  `torch.random.fork_rng()`(RNG 격리에 전역적으로 사용, `docs/design_
  decisions.md` 2절/12절 참고)가 보이는 GPU 전부의 RNG 상태를 매번
  저장·복원해 불필요한 오버헤드가 생기고, `PYTHONWARNINGS=error` 같은
  엄격한 설정에서는 첫 호출에서 경고가 예외로 승격돼 그리드가 아예
  시작하지 못할 수 있다 — 결과값 자체에는 영향 없음, 순수 운영상 주의사항.
- `--ignore-smoke` — 스모크 필터링을 무시하고 78개 전체 강행.
- `--blurry-ratio 0.3` — i-Blurry 재등장 스트림(기본 0.0=disjoint). Disjoint와
  Blurry를 둘 다 보려면 이 스크립트를 두 번(`--blurry-ratio 0`,
  `--blurry-ratio 0.3`) 실행한다 — 결과 파일명이 자동으로 구분된다.

이미 있는 결과는 `code_version`이 지금 코드와 일치하면 건너뛴다(재개 가능) —
컴포넌트를 고쳤으면 자동으로 재계산된다.

### 3) 리더보드/리포트 생성

```
python -m testbed.experiments.leaderboard_builder
```

`results/*.json`을 모아 데이터셋별로 `reports/leaderboard_{dataset}.csv`,
`summary_{dataset}.csv`, `drift_summary_{dataset}.csv`,
`no_cl_comparison_{dataset}.csv`, `per_category_{dataset}.csv` 등을 생성하고
콘솔에 요약을 출력한다. `results/*.json`에 낡은(현재 코드와 다른)
`code_version`만 섞여 있으면 경고를 낸다.

## 테스트

```
python -m pytest testbed/tests/
```

`test_grid_invariants.py`가 이 벤치마크의 핵심 전제(RNG 격리, 그리드 조합
수, 라벨예산 면제, kwargs 배선 등)를 회귀 테스트로 고정해둔다 — 컴포넌트를
고친 뒤에는 항상 먼저 돌려볼 것.

## 디렉토리 구조

```
testbed/
  base/            5개 슬롯의 추상 base class + 공유 backbone(FCLAutoEncoder)
  components/      슬롯별 실제 구현(ssf/, cade/, spider_gpm/, cndids/, novelty_baselines/)
  pipeline/        CLClient(8단계 오케스트레이터), component_registry, 공통 baseline
  common/          평가지표, 결과 스키마, 그리드 호환성 정의, RNG 유틸
  data/            데이터셋 로더(+ .cache/)
  configs/         전역/컴포넌트별 하이퍼파라미터 YAML(전부 원 논문 근거 주석 포함)
  experiments/     smoke_test.py / grid_runner.py / leaderboard_builder.py
  results/         그리드 실행 결과(조합별 JSON) — git 미추적
  reports/         리더보드/분석 CSV·JSON
  tests/           pytest 회귀 테스트
  docs/            design_decisions.md(현재 유효한 설계 결정) + 과거 작업 일지
```
