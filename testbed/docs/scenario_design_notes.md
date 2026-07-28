# 시나리오 설계 노트

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

따라서 `scenarios/*.yaml`은 `type: task_incremental`만 정의하며,
`common/scenario_loader.py`도 이 형식만 검증한다. streaming_window 지원은
이 구현의 범위 밖이며, 추가하려면 먼저 PRD 7절 이슈1을 해결해야 한다.

## n_experiences=5, labeling_budget=0.1 선택 근거

- `n_experiences=5`: CND-IDS 원 논문이 X-IIoTID/CICIDS2017/UNSW-NB15를 5개
  experience로 분할한다고 명시(부록A, `CND-IDS/utils.py`의 여러
  `create_split_experiences` 호출부에서 확인). 10.1절 global_hparams 기본값.
- `labeling_budget.value=0.1`(10%): 특정 논문 수치가 아니라 이 테스트베드의
  기본값이다(9절 scenario 스펙 자체가 "테스트베드 기본값"이라고 명시). SSF는
  실측으로 라벨링 비용을 20~50배 절감했다고 보고하지만, 그 절감률 자체를
  이 테스트베드가 재현 목표로 삼지 않는다(0절 — 재현이 아니라 재조합·비교).

## Track B에서 labeling_budget의 의미 축소 (9.2절 참고)

Track B(`supervision: label_free`에 대응하는 컴포넌트 조합, 즉 anti_forgetting
='cndids')에서는 `sample_selector=random`이 `labeling_budget`만큼 샘플을
무작위로 고르지만, 그 결과(`selected_labels`)는 `CNDIDSAntiForgetting.
compute_loss()`에서 전혀 쓰이지 않는다(components/cndids/cndids_anti_forgetting.py
참고 — 라벨을 인자로 받되 의도적으로 무시). 따라서 Track B에서
`labeling_budget`은 "선택 개수"로만 기능하고 "라벨링 비용" 절감이라는 원래
의미는 없다 — 별도 시나리오 파일을 두지 않고 같은 scenarios/*.yaml을
공유하는 이유이기도 하다.
