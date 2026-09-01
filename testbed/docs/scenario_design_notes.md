# 시나리오 설계 노트

**추기(2026-07-29)**: 여기서 설명하는 `scenarios/*.yaml`과
`common/scenario_loader.py`는 전체 코드베이스 최종 검토 중 실제 실행
경로(`grid_runner.py`/`smoke_test.py`) 어디에서도 호출되지 않는 죽은
코드임이 확인되어 삭제했다(사용자 지시) — 실제 구현은 이 문서 아래에 설명된
값들(n_experiences=5, labeling_budget=0.1 등)을 `configs/global_hparams.yaml`과
`grid_runner.py`/`smoke_test.py`에 직접 반영하는 방식으로 진행되었다. 이
문서가 설명하는 설계 결정(왜 task_incremental만인지, 왜 이 값들인지, Track
B에서 labeling_budget의 의미가 왜 축소되는지) 자체는 여전히 유효하고 실제
구현과 일치하므로 그대로 남겨두되, "scenarios/*.yaml이 어떻게 검증되는지"에
대한 서술만 죽은 코드를 가리키지 않도록 아래에서 정정한다.

## 구현 범위: task_incremental만 구현, streaming_window는 미구현

PRD 9절은 `experience_definition.type`으로 `task_incremental`과
`streaming_window` 두 값을 정의하지만, 이 구현은 **`task_incremental`만
구현했다.**

이유:
- PRD 7절 미해결 이슈 1이 "anti_forgetting=gpm을 streaming_window 시나리오와
  결합할지 여부"를 명시적으로 미해결 상태로 남겨두었다. GPM(SVD 기반 gradient
  projection)은 "태스크 종료" 시점에 activation을 수집해 기저를 갱신하는
  구조라(components/spider_gpm/gpm_anti_forgetting.py 참고) task_incremental
  처럼 태스크 경계가 명확한 시나리오에는 자연스럽게 맞지만, streaming_window
  처럼 연속적인 윈도우 흐름에서 "태스크 경계"를 어디로 정의할지는 이 PRD가
  결정하지 않았다.
- SSF 원 논문 자체는 streaming_window(sample_interval 기반 동적 윈도우)로
  동작하지만(SSF/ssf.py), 이 테스트베드는 4개 논문 공통의 experience 구조(9.2절
  — 병합 후 원본 순서로 n_experiences 균등 분할, experience당 stratified
  80/20)를 모든 컴포넌트에 동일하게 적용해야 공정한 비교가 가능하다(PRD 0절).
  SSF의 실제 스트리밍 윈도우 방식을 그대로 쓰면 다른 세 논문 유래 컴포넌트와
  같은 experience 구조를 공유할 수 없다.

따라서 실제 구현(`configs/global_hparams.yaml`의 `n_experiences`,
`grid_runner.py`/`smoke_test.py`에 하드코딩된 `labeling_budget`)은
`task_incremental` 하나만 반영한다. streaming_window 지원은 이 구현의
범위 밖이며, 추가하려면 먼저 PRD 7절 이슈1을 해결해야 한다.

## n_experiences=5, labeling_budget=0.1 선택 근거

- `n_experiences=5`: CND-IDS 원 논문이 X-IIoTID/CICIDS2017/UNSW-NB15를 5개
  experience로 분할한다고 명시(부록A, `CND-IDS/utils.py`의 여러
  `create_split_experiences` 호출부에서 확인). 10.1절 global_hparams 기본값.
- `labeling_budget.value=0.1`(10%): 특정 논문 수치가 아니라 이 테스트베드의
  기본값이다(9절 scenario 스펙 자체가 "테스트베드 기본값"이라고 명시). SSF는
  실측으로 라벨링 비용을 20~50배 절감했다고 보고하지만, 그 절감률 자체를
  이 테스트베드가 재현 목표로 삼지 않는다(0절 — 재현이 아니라 재조합·비교).

## Track B는 labeling_budget을 아예 적용하지 않는다 (2026-08 수정, 이전 서술 정정)

**이 절의 이전 버전은 더 이상 사실이 아니다** — "Track B도 `sample_selector=
random`이 `labeling_budget`만큼 샘플을 뽑되 라벨만 무시한다"고 서술했는데,
CND-IDS 원 논문(Algorithm 1: "Get Xtrain from experience data Ei" ->
"Fit CFE to Xtrain")을 다시 대조한 결과 label_budget 개념 자체가 없이
experience 전체를 그대로 학습에 쓴다는 걸 확인해 `pipeline/cl_client.py`의
Step 3를 고쳤다(`combo["track"] == "B"`일 때 `sample_selector.select()`를
아예 호출하지 않고 `selected_data = new_data` 그대로 사용 — 라벨링 비용
"절감"이 아니라 애초에 label_budget 게이트 자체가 없다). `TRACK_B_GRID`가
여전히 `sample_selector: ["random"]`을 슬롯에 남겨두는 건 combo 딕셔너리
스키마를 모든 조합에서 통일하기 위한 것일 뿐, Track B 실행에서는 이 값이
빌드만 되고 실제로 쓰이지는 않는다(`RandomSelector` 인스턴스는 생성되지만
`select()`가 호출되지 않음) — 리더보드/summary에 `sample_selector=random`
으로 표시되는 Track B 행을 볼 때 이 점을 감안해야 한다.
`CNDIDSAntiForgetting.compute_loss()`가 라벨을 인자로 받되 의도적으로
무시하는 라벨-프리 설계라는 점은 그대로 유효하다(components/cndids/
cndids_anti_forgetting.py 참고).
