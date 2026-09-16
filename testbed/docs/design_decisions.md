# 설계 결정 기록 (2026-09-14 통합)

## 이 문서의 존재 이유

코드 곳곳의 주석이 "PRD 3.1절", "PRD 13절", "수정 계획 1.3절/2.2절/3절/3.2절/6.2절" 같은
문서를 근거로 인용한다. 그 문서들은 이 저장소 어디에도 없다 — 여러 세션에 걸친 대화와
Claude Code의 개인 플랜 파일(저장소 밖, 세션 로컬)에만 존재했다. 그 결과 코드 주석이
참조 불가능한 문서를 계속 인용하는 구조가 됐고, 이건 실제로 반복된 혼란의 원인 중
하나였다(2026-09-14 감사에서 발견).

**이 문서가 그 자리를 대신한다.** 앞으로 "PRD X절"/"수정 계획 X절"이라는 표현이 코드에
남아있다면 그건 이 문서가 갱신되기 전의 낡은 인용이다 — 새로 코드를 고칠 때 그런 표현을
발견하면 이 문서의 해당 절을 가리키도록 정정할 것. 컴포넌트별 세부 근거(원 논문 몇 줄이
무엇을 하는지, A/B 수치 등)는 각 컴포넌트 파일 자신의 docstring에 있다 — 이 문서는
**여러 컴포넌트에 걸친(cross-cutting) 결정**과 **그 결정들이 왜 필요했는지**만 다룬다.

## 1. 프로젝트 목적 — 벤치마크 논문, 신규 방법 제안 아님

CL-NIDS-Bench는 4개 논문(SSF/INFOCOM 2025, CADE/USENIX Security 2021, SPIDER/
INFOCOM 2024, CND-IDS/DAC 2025) + GPM(SPIDER가 공식 결합하는 구성요소, Saha et al.
ICLR 2021)의 메커니즘을 5-슬롯 플러그인 파이프라인(drift_detector/sample_selector/
memory_manager/anti_forgetting/anomaly_scorer)으로 분해해 공통 인프라 위에서
재조합·비교하는 **체계적 비교 연구(벤치마크) 논문**이다. 신규 continual-learning
방법을 제안하는 논문이 **아니다** — 그건 명시적으로 future work다.

**이 결정이 다른 모든 결정에 미치는 영향**: 각 컴포넌트가 원 논문과 100% 동일하게
재현되는 것 자체는 목표가 아니다. 오히려 "동일한 조건 아래서 슬롯 값만 바꿨을 때
성능이 어떻게 달라지는가"를 깨끗하게 비교하는 것이 핵심이다. 그래서:
- 컴포넌트가 원문과 다르면(의도적이든 실측으로 발견됐든) **그 자체가 결함이 아니다** —
  다만 왜 다른지, 그 차이가 결과에 미치는 영향을 실측하고 정직하게 기록해야 한다.
- 반대로 "슬롯 값만 바꿔도 통제된 비교가 성립한다"는 전제를 깨는 문제(2/3절 참고)는
  개별 컴포넌트의 논문 충실도 문제보다 **더 근본적인 우선순위**를 가진다 — 벤치마크의
  존재 이유 자체를 위협하기 때문이다.

## 2. 전역 RNG 오염과 그 격리 (2026-09-14)

**문제**: `torch.manual_seed(seed)`는 콤보 실행당 1회만(`grid_runner.py`의
`run_combo_full()`/`run_joint_baseline()`, `smoke_test.py`의
`run_smoke_test_for_combo()`) 호출되고 라운드마다 재시드하지 않는다. 이 상태에서
일부 컴포넌트(CADE의 사설 encoder 초기화·매 라운드 학습, SSF 마스크 최적화의 `M_c`/
`M_t` 초기화, GPM의 자기학습 배치 추출, SPIDER/SSF/CND-IDS 메모리 매니저의 버퍼
교체·리플레이 샘플링, `RandomSelector`/`SSFSampleSelector`의 표본 선택)이 **조건부로**
(어떤 슬롯 값이 선택됐을 때만) 전역 RNG를 소비한다. 그 결과 예를 들어 `dd=cade`가
켜져 있는지 여부에 따라 그 뒤에 실행되는 메인 공유 모델의 셔플·표본선택이 보는 난수
시퀀스 자체가 달라진다 — 즉 리더보드에서 조합 A와 B의 성능 차이 중 일부가 "메커니즘
차이"가 아니라 "그 컴포넌트가 전역 RNG를 얼마나 소비했는가"에서 올 수 있었다.

**해결**: `torch.random.fork_rng()`(진입 시 RNG 상태를 저장, 블록을 빠져나오면 그대로
복원하는 PyTorch 내장 컨텍스트매니저)로 위 컴포넌트들의 사설 랜덤성 소비를 전부
격리했다. 블록 안에서는 `testbed/common/rng_utils.py`의 `derived_seed(seed, tag,
counter)`(SHA-256 해시 기반)로 파생 시드를 걸어, 그 컴포넌트 자신의 결과는 재현
가능하게 유지하면서 블록 밖의 전역 RNG 스트림은 그 컴포넌트가 있든 없든 동일하게
만든다. **처음엔 정수 오프셋(예: `seed + 90300 + counter`, 오프셋끼리 100 간격)
방식을 썼다가 해시 기반으로 교체했다** — GPM의 자기학습 카운터(`_self_training_
call_count`)가 라운드가 아니라 미니배치마다 증가해 Track A 실전 설정(epoch=200)
에서 수천까지 쉽게 도달, 100 간격으로 떼어둔 다른 컴포넌트의 오프셋 구간을 실제로
침범하는 걸 발견했기 때문이다(2026-09-14 2차 재검토). 해시 기반은 `tag` 문자열이
다르면 카운터가 아무리 커져도 실질적으로 충돌하지 않는다(직접 검증: 11개 tag ×
카운터 0~20000 범위에서 충돌 0건).

격리 대상 8개 파일과 각자의 `tag`: `cade_drift_detector.py`(생성자→
`"cade_encoder_init"`, `_fit_impl`→`"cade_fit"`), `cl_client.py`
(`_compute_ssf_masks`→`"ssf_masks"`), `gpm_anti_forgetting.py`
(`get_self_training_batch`→`"gpm_self_training"`), `spider_memory_manager.py`
(`update`→`"spider_update"`, `get_replay_batch`→`"spider_replay"`),
`ssf_memory_manager.py`(`"ssf_memory_update"`/`"ssf_memory_replay"`),
`cndids_memory_manager.py`(`"cndids_memory_update"`/`"cndids_memory_replay"`),
`common_baselines.py`(`RandomSelector.select`→`"random_selector_select"`),
`ssf_sample_selector.py`(`_quota_select`→`"ssf_quota_select"`).

**실측 검증**(NSL-KDD, 격리 이후): `(ss=random,mm=none,af=none,as=none)`에서
`dd=none`과 `dd=cade`가 f1/bwt/pr_auc 소수점 8자리까지 완전히 동일함을 확인 —
`(ss=ssf,mm=spider,af=gpm,as=none)`에서도 동일하게 확인. `as=cade_mad`로 바꾸면
(CADEMADScorer가 CADE의 사설 encoder에 연결되는 구조적 이유로) 여전히 다름도 함께
확인 — 즉 격리가 "신호가 안 쓰이는 경우"만 정확히 무력화하고 "구조적으로 진짜 다른
경우"는 건드리지 않는다. 이 항등성은 `testbed/tests/test_grid_invariants.py`의
`test_dd_cade_identical_to_dd_none_when_signal_unused`/
`test_dd_cade_differs_from_dd_none_when_wired_to_cade_mad`로 고정돼 있다 — 앞으로
누군가 이 격리를 깨거나 실수로 되돌리면 테스트가 즉시 실패한다.

**아직 격리되지 않은 것(2026-09-14 확인, 실제 그리드에는 영향 없음)**:
`cndids_anti_forgetting.py`의 `on_experience_start()`가 정상 참조 풀을 캡할 때 쓰는
`torch.randperm`(`held_out_reference`가 없을 때만 타는 폴백 분기)은 격리돼 있지 않다.
다만 실제 그리드/스모크 실행에서는 `CLClient`가 항상 `held_out_normal_reference`를
채워 넘겨 이 분기 자체가 발동하지 않는다(죽은 코드) — `CLClient` 없이 이 컴포넌트를
독립적으로 쓰는 코드(단위 테스트 등)를 새로 작성할 때만 주의가 필요하다.

## 3. `dd=cade`+`as=cade_mad`에서 af/ss/mm이 정확도 지표에 아무 영향을 못 준다
   (2026-09-14 발견 — RNG 오염만큼 핵심 전제를 위협하는 문제)

**발견**: 78개 그리드에서 무작위 20개 콤보를 스모크 규모로 돌려보다가, `dd=cade`+
`as=cade_mad`가 함께 켜진 서로 다른 4개 콤보(af/ss/mm이 전부 다름 — `ssf/ssf/gpm`,
`ssf/spider/none`, `random/none/gpm`, `random/none/lwf_ssf`)가 f1/bwt/pr_auc
**소수점 10자리까지 완전히 동일**함을 확인했다(f1=0.4401247900,
bwt=-0.1231364344, pr_auc=0.7517598561 — 넷 다 정확히 같음).

**원인(코드로 확인)**: `CLClient.__init__`이 `dd=cade`+`as=cade_mad`를
`CADEMADScorer.set_private_encoder()`로 연결하면 `uses_shared_representation`
이 `False`로 바뀐다. 그 뒤 Step 6/7의 평가 경로(`cl_client.py`)는
`uses_shared_representation=False`이면 공유 backbone(`self.model`)을 아예 거치지
않고 원본 데이터를 그대로 `CADEMADScorer.score()`에 넘기며, 이 메서드는
`self._private_detector.min_anomaly_score(data)`(CADE 자신의 사설 encoder +
카테고리별 centroid/median/MAD)로만 점수를 낸다. 이 사설 encoder는
`CADEDriftDetector.fit_with_category(new_data, new_labels, train_category)`
(Step 3, `consumes_full_round_data=True`라 라운드 전체를 씀)로만 학습되는데, 이
호출은 `af`/`sample_selector`/`memory_manager`가 무엇이든 완전히 동일하게
일어난다 — 즉 **공유 모델(af/ss/mm이 실제로 훈련시키는 대상)이 최종 정확도
지표 계산에 단 한 번도 관여하지 않는다.** 공유 모델은 여전히 Step 4에서
학습되긴 하지만(계산 자원은 쓴다) 그 출력은 버려진다.

**영향 범위**: `dd=cade`이면서 `as=cade_mad`인 모든 조합 — 활성 (ss,mm) 쌍
2개(`("random","ssf")`,`("ssf","ssf")`) × af(3) = 6개, 비활성 (ss,mm) 쌍
4개 × af(3) = 12개, 합계 **72개 Track A 콤보 중 18개(25%)**. 이 18개 안에서는
af/ss/mm을 무엇으로 바꿔도 f1/precision/recall/pr_auc/bwt/`per_category_final`
(전부 같은 `eval_scores`/`threshold`에서 파생됨)이 전부 동일하다.

**이게 "버그"가 아니라 "그리드 해석의 함정"인 이유**: `dd=cade`+`as=cade_mad`
연결 자체는 CADE 원 논문의 실제 방식(사설 대조학습 encoder로 판정)을 충실히
재현한 의도된 설계다(`cade_anomaly_scorer.py` 참고) — 고칠 대상이 아니다.
문제는 이 18개 셀을 "af가 dd=cade+as=cade_mad 조합에서도 성능에 영향을 주는가"
를 보여주는 18개의 독립된 관측치로 해석하면 안 된다는 점이다 — 실제로는 전부
"CADE 방법 자체가 얼마나 잘 작동하는가"라는 **하나의** 관측치일 뿐이다.

**정확도 지표와 자원 지표를 구분해야 한다**: `training_time_sec`/
`memory_footprint*`/`labeling_cost`는 공유 모델이 실제로 학습되며 쓴 자원을
반영하므로 af/ss/mm에 따라 여전히 달라질 수 있다(확인 안 됨, 추후 확인 필요) —
즉 이 18개 콤보를 그리드에서 완전히 제거하는 건 잘못이다(Edge Pareto 같은
자원 효율 분석에서는 여전히 서로 다른 데이터 포인트일 수 있음). **제거하지
않고 그대로 두되, 슬롯별 F1 기여도 분석(7절 "슬롯별 F1 기여도 집계 코드",
아직 미구현)을 구현할 때
반드시 이 18개를 특별 처리해야 한다**: (a) 정확도 기반 슬롯 기여도 분석에서는
이 18개를 하나의 대표값으로 묶거나 제외하고, (b) 자원 기반 분석(Edge Pareto
등)에서는 그대로 포함하되 "이 구간의 정확도는 af/ss/mm과 무관하다"는 각주를
반드시 남긴다. 이 처리를 안 하면 af/ss/mm의 측정된 효과가 이 18개의
무정보(non-informative) 중복 데이터로 인해 희석·왜곡된다 — 심사위원이 "왜
af=gpm의 분산이 유독 이 구간에서만 0인가"를 물으면 답할 근거가 이 절이다.

**숨은 전제(회귀 테스트 작성 중 발견, 2026-09-14)**: 이 항등성은 `train_category`가
실제로 전달될 때만 성립한다 — `cl_client.py:342`의
`elif hasattr(self.drift_detector, "fit_with_category") and train_category is not None:`가
`train_category is None`이면 `consumes_full_round_data` 면제 분기 전체를 건너뛰고
`self.drift_detector.fit(selected_data, selected_labels)`(라벨예산 서브셋, 즉
`sample_selector`에 따라 달라지는 데이터)로 조용히 폴백하기 때문이다. `grid_runner.py`/
`smoke_test.py`는 둘 다 `dataset_loader.py`가 무조건 채워주는 `e["train_category"]`를
그대로 넘기므로 실제 그리드 실행에서는 이 폴백이 절대 발생하지 않는다 — 하지만 이
항등성을 확인하는 임의의 스크립트나 단위 테스트가 `run_experience()`를 `train_category`
없이 직접 호출하면(실제로 `test_grid_invariants.py`의 이 절 회귀 테스트를 처음 작성할 때
바로 이 폴백에 걸려 항등성이 깨진 것처럼 보이는 거짓 실패가 났었다) 이 불변조건이
성립하지 않는 다른 코드 경로를 테스트하게 된다. 프로덕션 코드를 방어적으로 바꾸지는
않았다 — 실제 호출부 2곳(`grid_runner.py`, `smoke_test.py`) 전부 항상 카테고리를
넘기므로 지금 존재하지 않는 시나리오에 대한 방어 코드를 추가하지 않는다는 이
프로젝트의 원칙(6절)을 그대로 따른다. 대신 `test_grid_invariants.py`가
`train_category`를 명시적으로 채워 넘기는 전용 헬퍼(`_run_and_collect_with_category`)로
이 전제를 문서화해뒀다.

**회귀 테스트**: `test_grid_invariants.py`의
`test_dd_cade_as_cade_mad_scoring_independent_of_af_ss_mm`(위 4개 콤보 조합으로 실측
재현, threshold/eval_scores 완전 일치 확인)과
`test_dd_cade_as_cade_mad_training_loss_still_varies_by_af`(반대로 공유 backbone의
`avg_train_loss`는 af가 손실항을 실제로 바꾸면(`af=lwf_ssf`) 여전히 달라야 함을 확인 —
`af=gpm`은 GPM이 손실 값이 아니라 `project_gradients()`로 역전파 후 그래디언트만
투영하므로 `avg_train_loss` 자체는 `af=none`과 항등인 게 정상이라 이 두 번째 테스트의
반례로 못 쓴다는 것도 작성 중 실제로 걸려서 발견).

## 4. Track A 그리드 재구성 — 90개 → 72개 (2026-09-14)

`common/compatibility.py`의 `TRACK_A_DD_ACTIVE_SS_MM`/
`TRACK_A_DD_INERT_VALUES_BY_AS`가 "drift_detector의 출력이 실제로 파이프라인에
영향을 주는가"를 기준으로 중복 조합을 제거한다. 2절의 RNG 격리 이후 이 분류가
바뀌었다:

1. **`("ssf","spider")`를 활성에서 제외**: `SSFSampleSelector` 재작성(별도 결정,
   `ssf_sample_selector.py` 참고) 이후 `drift_score`를 전혀 안 쓰고, `SPIDERMemoryManager.update()`도
   `drift_detected`를 안 읽는다 — 실측으로 `dd=none`과 `dd=cade`가 완전히 동일함을
   확인(2절 참고).
2. **비활성 조합에서 `dd=cade`가 순회할지는 `anomaly_scorer`에 따라 갈린다**:
   `as=cade_mad`면 `dd=cade`가 `CADEMADScorer`와 사설 encoder로 연결돼(구조적 이유,
   RNG와 무관) 여전히 `dd=none`과 다르므로 유지. `as=none`이면 RNG 격리 이후
   완전히 동일해져 제외.

결과: Track A 활성 (ss,mm) 쌍 2개(`("random","ssf")`, `("ssf","ssf")`) × dd(3) ×
af(3) × as(2) = 36, 비활성 4개 쌍 × [af(3) × (as=cade_mad: dd 2 + as=none: dd 1)]
= 4×9 = 36, 합계 72. Track B는 그대로 6개. **총 78개**(이전 96개).

`enumerate_valid_combos()`가 `assert len(combos) == 78`로 이 수를 스스로 검증하고,
`testbed/tests/test_grid_invariants.py`의 `test_enumerate_valid_combos_shape_and_
membership`이 회귀 테스트로 고정한다.

## 5. 컴포넌트별 라벨예산 면제(consumes_full_round_data) — 완전한 목록

Track A는 기본적으로 라벨 예산(10%, `labeling_budget`)으로 골라진 서브셋만 학습에
쓴다(SSF 원 논문 근거). 그런데 CADE/SPIDER/GPM 세 컴포넌트는 **원 논문 자체에 라벨
예산이라는 개념이 없다** — 이 경우 그 컴포넌트에 한해 라운드 전체(`new_data`)를 보게
면제해주는 것이 각자의 원 논문에 더 충실하다:

| 컴포넌트 | 클래스 | 면제 대상 | 근거 |
|---|---|---|---|
| CADE 드리프트 감지 | `CADEDriftDetector` | family 단위 pairing/centroid 학습(`fit_with_category`) | `cade/utils.py` argparse 전체에 라벨예산 개념 없음 |
| CADE MAD 채점 | `CADEMADScorer` | median/MAD 재보정 참조 | 위와 동일(단, `dd=cade`와 연결된 경우 `fit()`이 no-op이라 실질 영향 없음) |
| SPIDER 버퍼 | `SPIDERMemoryManager` | 버퍼 갱신(`update`) | SPIDER 원문의 버퍼는 "이전 태스크 전체에서 무작위" — 라벨 예산 서브셋이 아님 |
| GPM 기저 | `GPMAntiForgetting` | SVD 기저 계산용 activation 표본(`set_full_round_data`) | GPM 공식 코드 `get_representation_matrix(net, device, x, y=None)`는 라벨(y)조차 안 씀 |

**2026-09-14 재검토로 발견된 누락**: GPM만 이 면제를 못 받고 있었다 —
`compute_loss()`가 라벨예산 서브셋의 미니배치에서만 activation을 누적해, SVD 기저가
실제 논문보다 훨씬 좁은 풀에서 계산되고 있었다. `set_full_round_data(new_data,
new_labels)` 훅을 추가하고 `cl_client.py` Step 3이 라운드당 1회 호출하도록 고쳤다 —
분류기 학습 자체(task loss)는 여전히 라벨예산 서브셋만 쓰고, GPM이 기저 계산에 쓰는
표본 풀만 바뀐다. CADE/SPIDER와 같은 근거의 구조적 일관성 수정이라 A/B 방향과
무관하게 채택했다(SSF 표본선택 수정과 같은 원칙, 6절 참고).

**`mm=spider` 클래스 균형 로직의 재검증 필요성(2026-09-14 2차 재검토로 발견)**:
위 수정으로 activation 표본의 출처가 근본적으로 바뀌면서(라벨예산 서브셋의
미니배치 → 라운드 전체), `mm=spider`일 때 클래스당 균형을 맞추는 기존 로직
(Algorithm 1 line 10 근거, 예전 데이터 출처로 A/B 검증됐던 것)이 **새 데이터
출처 기준으로는 재검증된 적이 없다**는 걸 재감사에서 지적받았다. 즉시
재A/B(NSL-KDD smoke)한 결과: 균형 ON f1=0.8003/bwt=+0.0028 vs OFF
f1=0.7950/bwt=+0.0045 — 차이가 작고 방향도 지표마다 갈려, 예전만큼 뚜렷한
근거는 아니다. 구조적 근거(Algorithm 1)가 있는 쪽(ON)을 잠정 유지하되, 7절에
멀티시드 단계에서 재확인할 항목으로 남긴다.

**이 목록은 사람이 수동으로 유지해야 한다**: 새 컴포넌트를 추가할 때 그 원 논문에도
라벨예산 개념이 없다면 같은 근거로 면제를 검토할 것. `testbed/tests/
test_grid_invariants.py`의 `test_no_label_budget_components_declare_full_round_
data_exemption`이 위 4개 클래스가 실제로 플래그를 선언하는지만 고정한다 — **새
컴포넌트를 이 목록에 넣는 것 자체는 테스트가 자동으로 알아내 주지 않는다.**

## 6. "실증적 검증 우선"과 "구조적 버그는 결과와 무관하게 고친다"의 구분

이 프로젝트에는 두 가지 다른 원칙이 있고, 어느 쪽을 적용할지 매번 판단해야 한다:

- **하이퍼파라미터/설계 선택**(예: `target_replay_fraction`, `activation_sample_size`,
  블러링 비율): 그리드 전체에 영향을 주는 값을 인용 근거만으로 바꾸지 않는다 — 반드시
  A/B 실측 후, 개선되는 방향으로만 반영한다.
- **구조적 충실도 버그**(원문의 실제 로직과 다르게 구현된 것으로 확인된 경우, 예:
  SSF 표본선택의 3분기 로직 누락, GPM의 라벨예산 면제 누락): 원문과 다르게 동작하던
  실제 결함을 원문대로 고친 것이므로, A/B 결과가 개선/악화 어느 쪽이든 **채택한다**
  (사용자 결정, 2026-09-14 SSF 표본선택 수정 때 명시). 다만 결과가 나빠졌다면 그
  수치와 원인 추정을 정직하게 문서에 남긴다.

이 둘을 혼동하면 안 된다 — "이미 검증됐다"는 이전 기록이 있어도 재검토 때는 항상
실제 코드를 다시 대조하고, 두 원칙 중 어느 쪽에 해당하는지부터 판단한다.

## 7. 알려진 한계 (논문 한계/방법론 절에 명시할 것)

- **`mm=cndids`의 FIFO 버퍼가 사실상 "직전 라운드만 기억"으로 퇴화**(이미
  `cndids_memory_manager.py`에 상세 기록돼 있었으나 2026-09-15 Round-9
  감사에서 이 중앙 목록에 빠져있음을 발견해 추가): `max_size`(3개
  memory_manager가 공유하는 순수 테스트베드 값, 1000)가 Track B 한 라운드의
  정상 표본 수(NSL-KDD 13,468건+)보다 훨씬 작아, 매 라운드 `update()` 직후
  버퍼가 거의 전부 "이번 라운드 것"으로 채워진다 — FIFO 자체는 정의대로
  동작하지만 "여러 라운드에 걸친 리플레이로 망각을 완화한다"는
  memory_manager 슬롯의 취지에서 보면 사실상 한 라운드만 리플레이하는
  것과 비슷해진다. 꼬리 슬라이싱 대신 무작위 표본 교체로 바꿔 여러 라운드
  표본이 비율대로 섞여 남도록 완화했지만(코드로 구현 확인됨), 버퍼 크기
  자체가 라운드 규모 대비 작다는 근본 한계는 남는다. `mm=cndids`가 쓰이는
  Track B 콤보(anomaly_scorer∈{pca,cade_mad}, 6개 중 2개)에 해당.
- **Track A/B 학습 조건 비대칭**: Track A는 라벨예산 10%, epoch=200(SSF 관례). Track
  B(CND-IDS 전용)는 라벨-프리 전체 experience, epoch=20(CND-IDS 관례). 리더보드는
  두 트랙을 같은 지표로 함께 순위 매기지만 학습 조건 자체는 다르다 — 감추지 않고
  명시한다(`global_hparams.yaml` 주석 참고).
- **CADE 라벨예산 면제가 만드는 셀 내부 비대칭**: `dd=cade`/`as=cade_mad`가 라운드
  전체를, 같은 조합 안의 SSF/GPM 계열은 여전히 10% 라벨예산만 보게 된다 — 컴포넌트별
  로는 각자 원 논문에 충실한 선택이지만, 조합 단위로는 "같은 셀 안에서도 컴포넌트마다
  받는 데이터량이 다르다"는 공정성 문제가 생긴다. 고치지 않는다(각자 원문 충실이
  맞음) — 논문에 명시.
- **공유 backbone 대 원 논문 아키텍처 재현의 근본적 긴장**: "공정 비교"와 "각 논문
  그대로 재현"은 원래 동시에 만족할 수 없다. 방안: (A, 권장) 메인 결과는 공유
  backbone, 부록에 각 컴포넌트를 원 논문 네이티브 아키텍처로 단독 실행한 이중 검증표
  추가(미구현). (C) 방법론 절에 "아키텍처를 통제변수로 고정해 메커니즘 효과만
  분리했다" 명시.
- ~~BWT와 forgetting의 부호 규약이 반대 방향~~ **해소됨(2026-09-15)**: 예전엔
  `common/metrics.py`의 `bwt()`가 (마지막 − 최초, 음수=망각)인데
  `leaderboard_builder.py`의 `forgetting_exp{j}`와 `grid_runner.py`의
  per-category `forgetting`은 (최초 − 마지막, 양수=망각)으로 반대 방향이었다.
  사용자 지적으로 재검토해 "그냥 하나로 통일해 출력할 수 있는지"를 확인한
  결과 — 이 세 값 모두 실제 실험 계산(모델 학습·평가)에는 전혀 관여하지 않는
  순수 리포팅 파생값이라 A/B 없이 안전하게 통일 가능하다고 판단했다.
  `bwt()`의 학계 표준 부호(Lopez-Paz & Ranzato, 음수=망각)로 전부 맞췄다 —
  `forgetting_exp{j}`→`bwt_exp{j}`, `no_cl_forgetting_exp{j}`→`no_cl_bwt_exp{j}`,
  per-category `forgetting`→`bwt`로 필드명도 함께 바꿔 이름과 부호가 항상
  일치하게 했다. 이제 이 저장소 어디서나 "음수=망각, 양수=개선"이 예외 없이
  성립한다 — 방향을 캡션에 따로 명시할 필요가 없어졌다.
- **CADE 리플레이 오버샘플링 비율이 라운드마다 다르게 실현될 수 있음**:
  `target_replay_fraction=0.15` 스윕으로 채택한 값이지만, 누적 참조 풀이 이미
  목표치를 넘으면 오버샘플링 자체가 발동하지 않아 실제 비율이 15%를 초과하는 라운드가
  이론상 있을 수 있다(NSL-KDD 실제 규모에서는 거의 항상 발동해 큰 문제는 아님으로
  보이나, 정밀 확인은 안 됨).
- **X-IIoTID 정식 78콤보 스모크 배터리**: 아직 재확인 필요 — **스코프 자체는
  이미 확정**(2026-09-13, 사용자가 3번째 데이터셋으로 결정, CND-IDS 원 논문이
  실제로 쓴 데이터셋 중 하나라는 근거). "재확인 필요"는 순전히 **스모크 테스트
  실행이 사용자 지시로 전체 데이터셋 공통으로 보류 중**이라는 뜻이지(NSL-KDD/
  UNSW-NB15도 마찬가지 — 아래 참고), X-IIoTID를 넣을지 뺄지가 아직 안 정해졌다는
  뜻이 아니다(2026-09-15, 이 문구가 "스코프 미정"으로 오독될 수 있어 명확화).
  `experiments/smoke_test_results.json`의 x-iiotid 96건은 전부 현재 code_version과
  다른 낡은 판정(Round-3 감사에서 직접 확인). 다만
  **데이터 로딩 계층만은 Round-4에서 직접 재검증 완료**(2026-09-14, 모델 학습
  없이 `load_dataset()`만 실행): `blurry_ratio=0`/`0.3` 둘 다 정상 로드(input_dim=78,
  9라운드, NaN/Inf 없음, 라벨 {0,1}만 존재), i-Blurry 블러링 대상 판정이 X-IIoTID
  에도 올바르게 일반화됨을 확인 — 9개 공격 카테고리 중 `crypto-ransomware`(최소
  카테고리, train ~366건)만 `min_per_round_share=30` 기준에 못 미쳐 disjoint로
  남고, 나머지 8개는 모든 비-주 라운드에 고르게 블러링됐으며 총 행 수(787,292)는
  disjoint/blurry 양쪽에서 정확히 동일(중복·유실 없음) — NSL-KDD(U2R)/UNSW-NB15
  (Worms)에서 이미 검증된 것과 같은 패턴이 세 번째 데이터셋에서도 재현됨.
  이건 파이프라인/모델 학습이 아니라 순수 데이터 로딩 검증이라 "스모크 테스트
  금지" 지시와 무관하다. 실제 78콤보 학습 스모크는 여전히 미실행.
- **CND-IDS K-Means의 seed가 멀티시드에서 실제로 바뀌는지는 확인됨**(2026-09-14,
  `seed=42` vs `seed=99`로 `cluster_centers_`가 실제로 다름을 실측 확인) — 이 자체는
  더 이상 한계가 아니라 해결된 항목.
- **멀티시드**는 아직 미실행 — 단일 시드(42) 결과만 있다. 벤치마크/비교 논문의
  핵심 주장(순위, 슬롯 기여도)이 전부 조합 간 비교이므로, 멀티시드 없이는 통계적으로
  방어하기 어렵다. Phase 0~6(설계) 완료 후의 필수 최종 게이트.
- **슬롯별 F1 기여도 집계 코드**가 아직 없다 — 벤치마크 논문의 핵심 결과 그림이 될
  가능성이 높은데(어느 슬롯이 성능에 가장 큰 영향을 주는가), 그리드 실행 후 순수
  후처리로 계산 가능해 지금은 보류.
- **GPM `mm=spider` 클래스 균형 로직의 정식 규모 재검증 필요**(5절 참고): activation
  표본 출처가 라벨예산 서브셋에서 라운드 전체로 바뀌면서, smoke 규모(epoch=5)
  재A/B에서는 균형 ON/OFF 차이가 작고 방향도 지표마다 갈렸다(f1 0.8003 vs 0.7950,
  bwt +0.0028 vs +0.0045). 정식 epoch(200)·멀티시드 단계에서 반드시 재확인할 것 —
  뚜렷한 이득이 없다고 최종 확인되면 더 단순한 "등장 순서대로 누적"으로 통일하는
  것도 고려.
- **자기학습(self-training) 피드백 루프가 두 군데 독립적으로 존재하며 둘 다
  별도 분석된 적 없음**(2026-09-14, Round-4 교차-슬롯 감사 발견): (a)
  `mm=spider`+`af=lwf_ssf`/`af=none` — SPIDER 버퍼의 pseudo-label(직전 라운드
  스냅샷)을 실측 라벨처럼 BCE에 그대로 씀(`spider_memory_manager.py`에 이미
  정직하게 기록됨). (b) `ss=ssf`+`mm=ssf`+`af=lwf_ssf` — SSF 드리프트 backfill의
  pseudo-label(현재 모델)을 BCE뿐 아니라 InfoNCE anchor 마스크에도 구분 없이 씀
  (`ssf_memory_manager.py` 참고, 이번에 새로 문서화). 크래시·눈에 띄는 붕괴는
  둘 다 없지만, "모델이 방금 낸 예측을 스스로 정답처럼 다시 학습한다"는 구조
  자체의 장단점은 미분석. 논문 한계 절에 명시할 것.
- **`mm=spider`+`af=gpm` 결합배치가 SPIDER Algorithm 1의 `r`(라벨:pseudo-label
  비례) 크기를 구현하지 않고, 서로 다른 시점(SPIDER=직전 라운드 스냅샷,
  GPM=현재 모델)의 pseudo-label을 동등 가중치로 섞는다**(2026-09-14, Round-4
  발견, `gpm_anti_forgetting.py`에 상세 기록): concat 방식 채택·GPM 자기학습의
  현재모델 채택·클래스 균형 채택은 각각 개별적으로 A/B됐지만, 이 셋이 합쳐진
  결과로 생기는 "두 pseudo-label 소스를 동등 가중치·시점 불일치인 채로 섞는다"는
  조합 효과 자체는 분석된 적 없다. 현재 채택안(f1=0.8462/bwt=+0.0285)은 건강하게
  작동하지만, `r` 비례 구현이나 두 pseudo-label 소스의 시점 통일은 지금과는 다른
  새 A/B가 필요해 이번 라운드에서는 손대지 않았다.

## 8. 이 문서를 갱신해야 하는 시점

- 컴포넌트 하나가 새로 추가/제거될 때(특히 5절의 라벨예산 면제 목록, 7절의 한계 목록).
- `common/compatibility.py`의 그리드 구성 로직이 다시 바뀔 때(4절).
- 새로운 cross-cutting 버그(한 컴포넌트가 아니라 여러 컴포넌트/파이프라인 전체에
  걸친 문제)를 발견하고 고쳤을 때(3절처럼).
- 멀티시드 실행 결과가 나와 7절의 "단일 시드" 한계가 해소될 때.
- **`base/models.py`(공유 backbone)를 고칠 때는 무조건**(2026-09-14 Round-6
  감사로 추가 — 11절 참고) — 78개 콤보 전체가 이 파일 하나를 공유하므로,
  겉보기엔 "SSF fidelity 수정" 하나처럼 보여도 실제로는 이 문서가 정의하는
  cross-cutting 변경 그 자체다. "어느 한 컴포넌트 파일을 고쳤다"가 아니라
  "몇 개 콤보에 영향을 주는가"로 판단할 것 — `base/`, `pipeline/`, `common/`
  아래 파일을 고칠 때는 특히 이 기준을 의식적으로 먼저 물을 것.

## 9. Round-2 감사 로그 (2026-09-14) — Round 1(위 1~8절) 완료 이후 재검증

Round 1(RNG 격리, 그리드 재구성, 라벨예산 면제, CADE-MAD 항등성 발견 등)을 마친 뒤,
"AI 분야 전문 교수가 봐도 흠 잡을 데 없는 수준"이라는 기준으로 재검증을 반복하라는
사용자 지시에 따라 진행한 2차 감사. 이전 결론을 신뢰하지 않고 원 논문 소스와 이
저장소 코드를 처음부터 다시 대조하는 독립된 서브에이전트 3개(SSF+CADE / GPM+SPIDER+
CND-IDS / 파이프라인 코어+base 계약)를 병렬로 투입하고, 직접 `experiments/grid_runner.py`,
`experiments/smoke_test.py`, `data/dataset_loader.py`, `experiments/leaderboard_builder.py`,
`configs/*.yaml` 전체를 다시 읽었다.

**실제로 발견·수정한 것**:

1. **Joint/Offline 상한선(6.2절)의 blurry_ratio 캐시 버그(직접 발견)**: `run_grid()`가
   `run_joint_baseline()`에 넘기는 `dataset`이 이번 실행의 `--blurry-ratio` 값을 그대로
   반영한 것인데, 캐시 파일명(`JOINT_BASELINE__{dataset}.json`)은 blurry_ratio를 구분하지
   않는다 — `--blurry-ratio 0`과 `--blurry-ratio 0.3`을 CLI 도움말이 권장하는 대로 두 번
   실행하면(i-Blurry, 4절 아래 Phase 4) 먼저 실행한 쪽의 데이터 순서로 계산된 Joint
   baseline이 이후 실행에 조용히 고정되는 순서 의존 버그였다. `run_grid()`가 이제 항상
   별도로 `blurry_ratio=0.0` 데이터셋을 로드해 Joint baseline을 계산하도록 고쳤다
   (`grid_runner.py` 참고). i-Blurry가 표본을 라운드 간에 "이동"만 하고 중복·삭제는
   안 하므로(`dataset_loader.py`의 `_class_incremental_split` 블러링 로직) 최종 수치
   자체가 크게 틀리지는 않았을 것이나, 같은 파일명이 실행 순서에 따라 다른 데이터로
   계산된 결과를 담는 건 이 프로젝트의 핵심 원칙("코드가 바뀌면 캐시를 못 믿는다")과
   어긋난다.

2. **`CNDIDSAntiForgetting.on_experience_start()`의 격리 안 된 `torch.randperm`(GPM/SPIDER
   담당 서브에이전트와 파이프라인 담당 서브에이전트 둘 다 독립적으로 발견 — 교차 검증됨)**:
   정상 참조 저수지 캡 로직(`max_normal_ref` 초과 시 무작위 서브샘플)이 이 저장소의
   다른 모든 randperm/randn/randint 호출과 달리 `torch.random.fork_rng()`로 격리돼
   있지 않았다. `set_held_out_reference()`가 항상 채워지는 실제 `grid_runner.py`/
   `smoke_test.py` 실행에서는 이 분기 자체가 도달되지 않아(held_out_normal_reference가
   비어있는 경우가 없음) 지금까지 실제 결과에 영향은 없었지만, `CLClient` 없이
   직접 이 컴포넌트를 쓰는 호출(단위 테스트 등)에서는 실제로 전역 RNG를 오염시킬 수
   있는 구조였다 — 2절의 원칙을 그대로 적용해 `derived_seed(seed, "cndids_normal_ref_cap",
   round_counter)`로 격리했다(`cndids_anti_forgetting.py` 참고).

3. **문서 정합성 자잘한 보완**: `grid_runner.py`/`smoke_test.py`의 `--datasets` 도움말이
   X-IIoTID를 언급하지 않던 것 보완, `cl_client.py`의 `run_experience()` docstring이
   실제로 반환하는 필드(`n_selected`/`label_budget_int`/`n_optimizer_steps`/
   `first_epoch_avg_loss`/`last_epoch_avg_loss` 등, 15절 스모크 게이트가 읽는 진단
   필드들)를 다 나열하지 않던 것 보완, X-IIoTID의 `random_state=42` train/test 분할이
   `load_dataset()`의 `seed` 인자와 독립적으로 고정된 이유(NSL-KDD/UNSW-NB15의 "공식
   분리 파일"과 같은 역할 — 멀티시드에서도 안 바뀌어야 다른 두 데이터셋과 "무엇이
   시드에 따라 변하는가"가 일관됨) 명확화.

4. **`dd=cade`+`as=cade_mad` 항등성(3절)의 숨은 전제 발견**: 이 항등성 회귀 테스트
   (`test_grid_invariants.py`)를 작성하던 중, `train_category`가 실제로 전달돼야만
   `cl_client.py`의 `consumes_full_round_data` 면제 분기가 타는 것을 발견했다 — 안
   넘기면 CADE가 `selected_data`(sample_selector에 따라 달라지는 라벨예산 서브셋)로
   조용히 폴백해 항등성 자체가 깨진다. `grid_runner.py`/`smoke_test.py`는 둘 다
   `dataset_loader.py`가 무조건 채워주는 `train_category`를 넘기므로 실제 그리드
   실행에서는 문제없이 성립하지만, 이 전제가 코드 어디에도 명시돼 있지 않았다(3절에
   상세 기록 추가).

**검증했지만 문제 없음으로 확인된 것들(재확인 근거 있음, 각 서브에이전트 보고서 참고)**:
SSF/CADE의 원문 대조(마스크 최적화, InfoNCE, contrastive loss, pairing 로직) 전부 일치,
GPM의 SVD 기저 계산·bias 증강 수학·클래스 균형 엣지 케이스 전부 정상, CND-IDS의
n_init 고정·seed 배선·LwF 가중치 전부 정상, `enumerate_valid_combos()`의 78개 조합
구성과 `validate_combo()`의 유효/무효 판정 전부 정상, `cl_client.py`의 8단계 순서와
Step4→5 순서(자기 리플레이 방지) 정상(다만 회귀 테스트가 없어 위 항목과 별개로
`test_memory_update_happens_after_training_not_before` 신규 추가), 5개 슬롯 전부
`build()`에 kwargs가 정상 전달됨(과거 memory_manager 버그의 재발 없음), base 클래스
4개의 `consumes_full_round_data`/`uses_shared_representation`/`threshold_needs_labels`/
`consumes_held_out_reference` 플래그 선언 일관성 정상.

**신규 회귀 테스트(`test_grid_invariants.py`, 12개 추가돼 총 38개)**: CADE-MAD 항등성
2개(채점 결과 항등 + 학습 손실은 여전히 달라야 함), Step4→5 순서 1개, CND-IDS 저수지
캡 RNG 격리 2개, 기존 seed-kwargs 프로브 2개에 `anomaly_scorer` 슬롯 추가(이전엔
4개 슬롯만 검사해 5번째 슬롯의 향후 회귀를 못 잡는 공백이 있었음).

**결론**: Round 2에서 실제 동작에 영향을 주는 버그는 발견하지 못했다(Joint baseline
캐시 문제와 CND-IDS randperm 둘 다 지금까지의 실제 실행 결과에는 영향이 없었음 —
전자는 실제로 두 blurry_ratio 실행을 아직 안 했고, 후자는 폴백 분기가 도달된 적이
없음). 다만 둘 다 "지금 우연히 안전한" 상태였을 뿐 구조적으로는 실제 버그였고, 향후
멀티시드·i-Blurry 정식 실행 단계에서 실제로 발현될 수 있었던 것들이라 Round 2가 아니었으면
그 시점에 처음 발견됐을 것이다. 이 정도 수준(3개 독립 서브에이전트 + 직접 재검토
전부가 원 소스 재대조를 완료하고 실제 코드로 반례를 재현/검증)에서 새로운 cross-cutting
문제가 나오지 않는다면, 남은 작업은 GPU 서버에서의 정식 그리드 실행(및 6.8절 멀티시드)뿐이라고
판단한다.

## 10. Round-3/4 감사 로그 (2026-09-14) — 계속

**Round 3(메트릭/레지스트리/베이스라인 + 문서 정합성)**: `common/metrics.py`의
F1/precision/recall(직접 손계산 일치)·PR-AUC(`sklearn.average_precision_score`와
소수점까지 정확히 일치)·BWT(부호 규약이 7절에 이미 기록된 한계와 일치, 새 문제
아님) 전부 정상. `component_registry.py`의 REGISTRY가 `compatibility.py`의 모든
슬롯 값을 빠짐없이 커버. `RandomSelector`/`NoAnomalyScorer`에 숨은 편향 없음.
`PCAScorer`가 CND-IDS 원문(`CND-IDS/AnomolyDetectors/PCA.py`)과 재구성오차·
`init_normal`·threshold 계산까지 정확히 일치(오히려 원문의 잠재적 out-of-bounds
위험을 이 테스트베드가 회피하는 방향으로 안전하게 개선돼 있음을 확인). 유일한
실제 발견 — **`validate_result()`가 타입만 검사하고 범위는 전혀 검사하지 않음**
(f1=1.5, pr_auc=-0.3 같은 값을 조용히 통과시킴) — `common/result_schema.py`에
범위 검사를 추가했다(f1/precision/recall/pr_auc/labeling_cost∈[0,1],
bwt∈[-1,1], 자원 지표≥0, roc_auc/best_f1_reference/normal_fpr은 NaN 허용).
기존 results/*.json 96개 전부로 드라이런해 위반 0건 확인 후 반영. 문서 정합성
감사(전담 서브에이전트가 두 번 stall되어 직접 수행)는 5절/4절/7절의 모든 수치
claim이 실제 코드와 정확히 일치함을 확인 — 남아있는 다수의 "PRD X절" 인용은
`design_decisions.md` 서두의 "발견 시 opportunistic하게 정정" 원칙에 따라 의도적
방치(대부분 실제 내용은 인용된 파일 자신의 docstring에 자기완결적으로 이미 담겨
있어 정보 손실 위험은 낮음을 표본 확인).

**Round 4(교차-슬롯 상호작용)**: 개별 컴포넌트가 아니라 "두 슬롯이 동시에 특정
값일 때"만 나타나는 문제를 전담으로 훑었다. X-IIoTID 데이터 로딩 계층(모델 학습
없이 `load_dataset()`만 직접 실행)도 i-Blurry 포함 재검증 — 정상(NaN/Inf 없음,
crypto-ransomware만 disjoint 유지, 총 행 수 보존, 위 "X-IIoTID 78콤보 스모크"
항목에 반영). SSF mask 공유 타이밍, SPIDER 버퍼 vs GPM 자기학습 풀의 데이터
중복 여부, `dd=cade`가 만든 drift 신호가 다른 memory_manager로 건너갈 때의
의미 불일치, `as=cade_mad` 활성 시 다른 컴포넌트가 공유 backbone의 `z`를
착각해서 쓰는지 — 넷 다 문제없음 확인. 실제 발견 2건은 모두 "버그"가 아니라
"각자 원 논문/이전 A/B에 충실하게 구현된 두 메커니즘이 합쳐질 때 생기는, 지금껏
별도로 분석된 적 없는 조합 효과"였다 — 위 7절에 추가한 두 항목(자기학습 피드백
루프 2건, GPM+SPIDER 결합배치의 `r`-비례 미구현 및 시점 불일치)과
`ssf_memory_manager.py`/`gpm_anti_forgetting.py`의 상세 기록 참고. 코드는
바꾸지 않았다 — 둘 다 이미 건강하게 작동함이 실측으로 확인된 상태고, "고치는"
쪽으로 가려면 지금과는 다른 새 A/B가 필요해 논문 한계 절에 정직하게 남기는 쪽을
택했다(6절의 "실증적 검증 우선" 원칙).

**테스트 스위트**: Round 3/4는 코드 로직을 바꾸지 않았고(result_schema.py의
범위검사 추가만 예외 — 기존 결과 96개 드라이런으로 별도 검증 완료) 새 회귀
테스트도 추가하지 않았다 — 발견된 두 항목이 "고쳐야 할 버그"가 아니라 "정직하게
문서화해야 할 한계"라 회귀 테스트 대상이 아니기 때문이다. `python -m pytest
testbed/tests/` 38/38 통과 유지 확인(테스트 개수는 이후 라운드에서 계속
늘었다 — 11절 참고, 이 문서의 각 라운드 로그가 "그 라운드 종료 시점"의
스냅샷이라 이후 숫자와 안 맞는 건 정상이다).

## 11. Round-5/6 감사 로그 (2026-09-14) — 자기 편향 대응 + 소급 문서화

**Round 5(순수 인프라 위생)**: `testbed/archive/`가 `results/` glob과 겹치지
않음을 직접 실행 확인, `requirements.txt`에 `joblib` 누락 발견·추가(`cndids_
anti_forgetting.py`가 `elbow_n_jobs>1`일 때 직접 import하는데 지금까지
scikit-learn의 전이 의존성으로 우연히 설치돼 있어 안 드러났었음), `common_
baselines.py`/`thresholding.py`/`base/sample_selector.py` 직접 재확인(문제
없음), `result_schema.py`의 범위검사(10절)에 전용 회귀 테스트(`test_result_
schema.py`, 22개) 신규 작성.

**Round 6(자기 편향 대응 — 지금까지의 판단 자체를 의심하는 전담 감사)**:
사용자가 "Claude가 자기 작업을 리뷰하면 확증편향이 생길 수 있다"고 명시적으로
우려해, 이전 라운드들의 결론에 호의를 베풀지 말고 전부 처음부터 재검증하라는
전담 서브에이전트 2개를 투입했다.

1. **소급 발견 — `base/models.py`의 ReLU 구조 수정(Round 1, 이 대화 시작
   이전)이 cross-cutting 결정 로그에 전혀 기록되지 않았음**: 78개 콤보 전체가
   공유하는 backbone의 forward pass 자체를 바꾼, 이 세션에서 가장 영향
   범위가 넓은 변경인데도(SSF 원문의 decoder/classifier 선행 ReLU 누락을
   발견해 추가 — `models.py` 자체 docstring에는 기록돼 있었음), 8절이 스스로
   요구하는 "cross-cutting 버그를 고쳤을 때 이 문서를 갱신"이 지켜지지 않고
   있었다(`design_decisions.md`에 "ReLU"/"models.py" 언급이 0건이었음을
   grep으로 직접 확인). 원인 추정: 이 세션이 "고칠 버그 찾기"에 집중하다
   개별 컴포넌트 fidelity 수정으로 분류해버려, 실제로는 5절/6절이 말하는
   "그리드 전체에 영향 주는 공유 파라미터/인프라 변경"에 해당한다는 걸
   놓쳤다 — 자기 검토가 놓치기 쉬운 바로 그 종류의 맹점이었다. **소급
   기록**: SSF 원문(`utils.py:28-57`, `AE_classifier`)의 decoder/classifier는
   둘 다 입력에 먼저 `ReLU()`를 적용한 뒤 첫 `Linear`를 통과하는 구조인데,
   이 테스트베드는 그 선행 ReLU를 빠뜨리고 z/x_hat을 그대로 첫 Linear에
   통과시키고 있었다 — 단순 스케일 차이가 아니라 음수 성분을 0으로 클리핑
   하는 비선형 변환 자체의 누락이라 "구조적 충실도 버그"(6절 원칙)로
   분류해 A/B 방향과 무관하게 채택했다. sanity 확인(NSL-KDD, epoch=5, 붕괴
   없음): naive 기준선 f1=0.1060/bwt=-0.6399, SSF 풀조합 f1=0.6542/
   bwt=+0.0102, CADE f1=0.4493/bwt=-0.1513 — 전부 정상 범위였다(정식
   epoch=200·전체 그리드 재검증은 GPU 실행 시 자동으로 이뤄짐, 기존
   results/*.json 96개는 code_version 불일치로 이미 자동 폐기 대상임을
   확인). **이 발견 자체가 보여주는 것**: 서브에이전트가 검증한 다른
   claim들(3절의 CADE-MAD 독립성, 4절의 그리드 축소, Round-2의 두 수정)은
   전부 소스 코드로 재추적해 "확증됨(CONFIRM)"이었다 — 즉 이 세션의 개별
   판단 자체가 틀렸던 게 아니라, "무엇을 이 문서에 기록해야 하는가"라는
   메타 수준의 판단이 한 번 빠졌던 것이다.
2. **Joint baseline 재로드가 `min_per_round_share`를 안 넘겨 캐시 키만
   중복될 수 있던 사소한 흠**(정확성 문제는 아니었음, `grid_runner.py`에서
   수정) — blurry_ratio=0.0에서는 이 값이 실제 분할에 전혀 안 쓰이지만
   `_dataset_cache_key()`가 캐시 키에는 포함시켜, `--min-per-round-share`를
   기본값 아닌 값으로 준 실행에서 내용은 같은데 디스크 캐시 파일만 하나 더
   생기는 낭비가 있었다. 이번 실행의 실제 인자를 그대로 넘기도록 고쳤다.
3. **테스트 스위트 자체의 공허함(vacuous test) 점검 — `test_grid_invariants.
   py`/`test_result_schema.py`의 모든 테스트를 실제로 깨보며 재확인**: 대부분
   실제로 그 불변조건이 깨지면 실패함을 monkeypatch로 직접 확인(예:
   `fork_rng()`를 no-op으로 만들면 RNG 격리 테스트가 실제로 실패, `derived_
   seed`가 seed를 무시하게 만들면 시드 민감도 테스트가 실제로 실패). 다만
   `test_gpm_class_balance_branch_actually_exercised_for_mm_spider`가
   `gpm._self_training_pool`을 직접 대입해 "mm=spider가 채운 것처럼"
   흉내만 내고 `cl_client.py` Step 3의 실제 배선(`combo["memory_manager"]
   == "spider"`일 때만 `set_self_training_pool()`을 호출하는 조건)은 한
   번도 거치지 않는다는 걸 발견 — 그 배선 자체가 깨지는 회귀(오타, 조건
   반전 등)는 어떤 기존 테스트도 못 잡았다. `test_mm_spider_actually_
   populates_gpm_self_training_pool_via_real_wiring`을 새로 추가해
   `run_experience()`를 실제로 호출하는 통합 테스트로 이 배선을 직접
   검증하고, 배선을 몽키패치로 깨뜨려 새 테스트가 실제로 실패하는 것까지
   확인 후 반영했다.

**테스트 스위트**: 61/61 통과(`test_result_schema.py` 22개 + `test_grid_
invariants.py`의 신규 통합 테스트 1개 포함).

**결론**: Round 6은 "이전 판단이 틀렸다"를 찾는 데는 실패했다 — 서브에이전트가
소스로 재추적한 모든 구체적 claim(3/4절, Round-2의 두 수정)이 확증됐다. 대신
"판단은 맞았지만 기록이 이 문서의 자기 정책을 못 지킨 사례"(ReLU 수정)와
"테스트는 있지만 진짜 배선을 안 거치는 사례"(GPM self-training pool)를
찾았다 — 둘 다 코드 로직이 아니라 "검증의 검증" 계층에서 나온 결과로, 자기
편향에 맞서는 이런 전담 라운드가 헛되지 않았음을 보여준다.

## 12. Round-7 감사 로그 (2026-09-14) — GPU 실행 준비성

이 테스트베드는 지금까지 GPU 없는 로컬 환경에서만 실행·검증됐다 — 남은
유일한 단계가 GPU 서버 실행이라, `--device cuda`에서만 드러날 수 있는 문제를
전담으로 훑었다(`.to(device)` 누락, sklearn/numpy 경계에서 GPU 텐서를 못
바꾼 경우, `torch.Generator()`의 암묵적 CPU 기본값 등). 결과: 크래시 유발
버그는 없음(정확히는, 이전에 이미 한 번 8건을 찾아 고친 적이 있는 "디바이스
이식성" 감사를 재확인한 것 — `docs/metric_justification.md` 참고 — 그 8건이
전부 여전히 올바르게 고쳐진 채로 남아있음을 이번에 독립적으로 재확인했다).
성능/신뢰성 관점의 발견 2건은 코드를 고치지 않고 아래처럼 처리한다:

1. **`torch.random.fork_rng()`가 13곳 호출부 전부 `devices=`를 명시하지
   않음**: 기본값(`devices=None`)은 "보이는 모든 CUDA 디바이스"를 fork한다
   (PyTorch 소스 직접 확인 — `torch/random.py`의 `if num_devices > 1 and not
   _fork_rng_warned_already:`가 정확한 트리거 조건). 이게 만드는 결과:
   (a) 여러 GPU가 동시에 보이는 프로세스에서 첫 호출 시 `UserWarning` 1회
   (기본은 무해하지만 `PYTHONWARNINGS=error` 같은 엄격한 설정에서는 예외로
   승격돼 그리드 전체가 시작하자마자 죽을 수 있음), (b) `GPMAntiForgetting.
   get_self_training_batch()`처럼 **미니배치마다**(mm=spider, epochs=200
   기준 라운드당 수백 회) 호출되는 곳에서는 안 쓰는 GPU까지 매번 상태
   저장/복원하는 순수 오버헤드가 누적됨. **결과값 자체는 안 바뀐다** —
   fork/restore되는 여분의 디바이스 RNG 상태는 실제로 소비되지 않으므로
   단일/다중 GPU 간 재현성이나 정확도에는 영향이 없다(직접 확인). 코드
   수정 대신 **운영상 완화책**을 택한다: `--shard`로 여러 GPU를 나눠 쓸 때
   프로세스마다 `CUDA_VISIBLE_DEVICES`를 하나로 고정하면(원래 멀티-GPU
   샤딩의 자연스러운 방식) 그 프로세스에서 `device_count()==1`이라 위
   조건이 아예 안 걸린다 — `README.md`의 `--shard` 안내에 추가했다. 13곳을
   전부 고치는 리팩터링은 이 코드베이스에서 가장 안전-critical한 메커니즘
   (RNG 격리, 이 세션 전체의 핵심 발견)을 마감 직전에 건드리는 것이라 —
   심각도(경고/오버헤드일 뿐, 결과 비정확성 없음) 대비 위험이 안 맞다고
   판단해 하지 않는다.
2. **`components/ssf/ssf_masks.py`의 마스크 최적화가 라운드당 파이썬
   루프로 작은 텐서 연산 약 2000개를 순차 실행**(10 bin × 100 SGD step ×
   2회 호출) — GPU에서 커널 launch 오버헤드로 상대적으로 느릴 수 있는
   패턴이지만 미니배치가 아니라 라운드당 1회(`ss=ssf`/`mm=ssf`일 때만)라
   전체 학습 시간을 지배할 정도는 아니다. SSF 원문 로직을 그대로 이식한
   코드를 성능 이유로 재작성하는 건 이 테스트베드의 목적(재현 충실도)과
   맞지 않아 손대지 않는다 — SSF 계열 조합이 예상보다 느리면 참고할
   메모로만 남긴다.

**테스트 스위트**: 코드 변경이 없어(문서만 추가) 61/61 유지.

**부록 — git 저장소 자체의 문제(테스트베드 코드 정확성과는 무관, 사용자
결정 필요, 2026-09-15 `.gitignore` 점검 중 발견)**: 원본 데이터셋 raw
CSV(`SSF-Strategic-Selection-and-Forgetting/{NSL_pre_data,UNSW_pre_data}/`,
`UNSW-NB15-raw/`, 합계 약 230MB)가 이미 git 히스토리에 커밋돼 있고, 이
저장소에는 실제 원격(`origin = github.com/JaeyeonChae1107/FCL`)이 연결돼
있어 이미 push된 상태로 보인다(로컬 `main`이 `[origin/main]`을 추적 중).
`X-IIoTID-raw/`(339MB, 아직 미추적)도 실수로 `git add -A` 한 번이면 같이
커밋될 뻔했다 — `.gitignore`에 이 디렉토리들과 `CADE/data/`,`CADE/models/`
를 추가해 **앞으로의** 추가 커밋만 막았다(이미 커밋된 과거 이력은
그대로임 — 되돌리려면 히스토리 재작성 + force-push가 필요한 되돌리기 어려운
작업이라 사용자 승인 없이 하지 않는다). 이미 커밋된 데이터셋 재배포가
각 데이터셋의 이용 약관과 맞는지(NSL-KDD/UNSW-NB15/X-IIoTID 전부 각자 배포
조건이 있음)는 이 세션이 판단할 문제가 아니다 — 사용자가 직접 확인하고
필요하면 히스토리 정리를 결정할 것(2026-09-15, 사용자 확인 — 실험용 데이터는
따로 관리하고 git에는 안 올리기로 결정. `.gitignore`는 이미 그 방향으로
반영돼 있음).

## 13. Round-8 감사 로그 (2026-09-15) — 하이퍼파라미터 수치 원문 대조

지금까지 컴포넌트 YAML의 키 이름이 생성자 파라미터와 일치하는지(3절/Round-3),
알고리즘 로직이 원문과 일치하는지(Round 1/2)는 검증했지만, **개별 수치
하나하나가 인용된 원문 소스와 실제로 일치하는지**는 전담으로 검증한 적이
없었다. `cade.yaml`/`ssf.yaml`/`cndids.yaml`/`gpm.yaml`/`global_hparams.yaml`의
모든 수치와 `base/models.py`의 `ssf_backbone_dims()` 공식을 원문 파일을 직접
읽어 하나씩 대조했다. 결과: 전부 확증(값이 정확히 일치하거나, "원문 근거
없음"이라는 자기 서술이 정직함을 확인) — 단 하나, `cndids.yaml`의
`triplet_margin: 2.0` 주석이 인용한 줄 번호(`CND_IDS.py:76-78`)가 틀렸다
(값 자체는 맞음, 실제 위치는 `:38-39`) — `cade_drift_detector.py`의
`run_cade_exp_ids_infiltration.sh:9→13` 정정과 같은 패턴의 사소한 줄번호
오타. 정정했다.

**테스트 스위트**: 코드 로직 변경 없음(YAML 주석만 수정), 61/61 유지.

## 14. Round-9 감사 로그 (2026-09-15) — 넓은 범위 자유 감사(수렴 확인용)

지금까지 8라운드가 점점 좁고 깊은 주제로 수렴해온 것과 달리, 이번엔 제약
없이 "뭐든 이상한 걸 찾으라"는 열린 지시로 전담 서브에이전트를 투입했다 —
그동안의 감사가 정말 수렴했는지(더 넓게 봐도 새로운 게 없는지) 확인하기
위함. 결과: 코드 정확성 관점에서는 새로운 문제를 못 찾았다(cl_client.py
8단계, dataset_loader.py의 X-IIoTID 전처리·누수 방지, leaderboard_builder.py
집계 전부 재확인해도 기존 문서와 일치). 유일한 발견은 이 저장소의 git
커밋 상태(이번 세션 전체가 아직 커밋 안 됨)였는데, 사용자가 "수정이 완벽히
끝난 뒤에만 커밋할 것"이라는 의도된 선택임을 확인해줘 — 코드 문제가
아니었다.

**직접 이어서 발견한 것(3개 컴포넌트 파라미터가 yaml에 안 보임)**: 서브에이전트
보고를 받은 뒤 "모든 생성자 파라미터가 어떤 component_hparams yaml에는
있는가"를 역방향으로(Round-8은 "yaml 값이 원문과 맞는가"였다면 이건 "생성자
기본값이 yaml에 노출돼 있는가") 직접 스크립트로 전수 점검했다.
`GPMAntiForgetting.activation_sample_size`(2000), `PCAScorer.
variance_threshold`(0.95), `SPIDERMemoryManager`/`SSFMemoryManager`/
`CNDIDSMemoryManager`가 공유하는 `max_size`(1000) 세 값이 생성자 기본값
으로만 존재하고 어떤 yaml에도 없었다 — cade.yaml의 `max_category_ref`/
`target_replay_fraction`처럼 "원문 근거 없는 테스트베드 값"도 전부 yaml에
명시해온 이 프로젝트의 일관된 관례에서 벗어난 사각지대였다. 세 값 모두
**기존 기본값과 정확히 같은 값**으로 yaml에 명시해(동작 변화 없음 —
`build()`로 실제 생성한 인스턴스의 속성값이 기존과 동일함을 직접 실행
확인) `gpm.yaml`/`cndids.yaml`에 추가했다. 이 과정에서 `max_size`(1000)가
Track B의 `mm=cndids`에서 만드는 실제 한계("FIFO가 직전 라운드만 기억"으로
퇴화 — 이미 `cndids_memory_manager.py`에 상세 기록돼 있었음)가 7절
중앙 목록에는 없었다는 것도 같이 발견해 추가했다.

**결론**: 코드 로직 자체는 Round 9에서도 확증 외에 새로 나온 게 없어 수렴
신호가 유지된다. 다만 "yaml 노출 완결성"이라는, Round 8까지 다루지 않았던
축을 하나 더 찾아 메웠다 — 이런 종류의(로직은 안 바뀌지만 투명성/일관성이
빠진) 발견이 나올 여지가 완전히 사라졌다고 단언하기는 여전히 어렵다.

**테스트 스위트**: `build()`로 세 값의 런타임 결과가 기존과 동일함을 직접
확인 후 반영, 61/61 유지.

## 15. Round-10 (2026-09-15) — 사용자 재확인으로 발견한 잔여 항목 3개

사용자가 이전에 보고한 "3개는 의도적 보류, 1개(수정 계획 인용)는 성격이
다르다"는 판정에 이의를 제기하며 4개를 다시 짚었다 — 재확인 결과:

1. **`cade_drift_detector.py`/`cade_anomaly_scorer.py`의 "수정 계획 2.2절"
   인용 6곳**: 사용자 말대로 순수 텍스트 정정이라 A/B·실행이 전혀 필요
   없는데도 Round 3에서 "이번 세션에 직접 편집한 파일만 정정"이라는 잘못
   적용된 기준으로 방치했었다 — 지금 전부 정정했다(design_decisions.md
   5절을 가리키거나, 대응 절이 없으면 팬텀 인용만 제거). 저장소 전체에
   ".py/.yaml 파일 안의 수정 계획 인용"은 이제 0건.
2. **CADE `_update_category_refs`의 꼬리 슬라이싱**: target_replay_fraction과
   같은 "A/B 필요해서 보류" 사유가 **아님**을 확인했다 — 재검토 결과 공격
   category는 class-incremental 분할상 평생 한 라운드에만 등장해 이 캡이
   "이전 라운드 참조를 밀어낸다"는 상황 자체가 성립하지 않고, 매 라운드
   재등장하는 정상(normal) category만 영향받는데 정상 행은 애초에 라운드
   배정 전 전역 셔플돼 있어(카테고리/피처와 무관, 분포 이동 없음) "편향
   없이 표본 크기만 줄어드는" 효과일 뿐이다 — CND-IDS의 FIFO 버퍼(같은
   category가 매 라운드 재등장하고 라운드마다 새 정보가 있어 소실이 실제
   정보 손실인 경우)와 근본적으로 다른 상황이라 같은 수정이 필요 없다고
   판단한다. 이 논증 자체는 코드 재실행 없이 순수하게 데이터 흐름을
   추적해 검증했다(`_fit_impl`이 `_update_category_refs`를 라운드당 정확히
   1회, 이미 셔플된 전체 라운드 데이터로 호출함을 직접 확인). 다만 이
   "필요 없다"는 판단 자체는 정성적 논증이라 실제 그리드 스케일 A/B로
   검증된 적은 없다는 점을 `cade_drift_detector.py` docstring에 정직하게
   남겼다. **이전 응답에서 "아마 같은 이유일 것"이라고 확인 없이 추정
   답했던 것은 부정확했다 — 실제로는 다른 이유(검증이 아니라 재검토 후
   기각)였다.**
3. **X-IIoTID 스코프**: 사용자가 "보류 중"이라는 문구를 스코프 자체가
   미정인 것으로 읽을 수 있다고 지적 — 확인 결과 스코프는 이미 2026-09-13에
   사용자가 확정했고(3번째 데이터셋), 7절의 "재확인 필요"는 **스모크 실행이
   전체 데이터셋 공통으로(NSL-KDD/UNSW-NB15도 마찬가지) 사용자 지시로
   보류 중**이라는 뜻일 뿐이었다 — 오독 소지가 있는 문구라 7절 자체를
   명확히 정정했다. 결정을 기다리는 게 아니라 이미 결정된 항목이 실행
   대기 중인 것뿐이다.

**결론**: 4개 중 3개(수정 계획 인용, X-IIoTID 문구, CADE 꼬리 슬라이싱의
근거 정리)는 실제로 손을 안 대고 있었거나 불명확하게 남아있던 것들이었고,
사용자의 재확인이 없었다면 "의도적 보류"로 뭉뚱그려 넘어갔을 것들이다 —
자기 검토만으로는 이런 걸 못 잡았을 가능성이 있다는 걸 보여주는 사례로
남긴다.

**테스트 스위트**: 코드 로직 변경 없음(문서/docstring만), 61/61 유지.

## 16. Round-11 (2026-09-15) — BWT 부호 통일

사용자가 개요 문서(아티팩트)를 보고 "BWT 정의를 하나로 정해서 출력할 수는
없냐"고 질문 — 7절이 "한계"로만 기록해두고 방치했던 BWT 부호 불일치를
실제로 없앨 수 있는지 재검토했다. 결론: **가능하고, 안전하다** — 이 세 값
(`common/metrics.py`의 `bwt()`, `leaderboard_builder.py`의
`forgetting_exp{j}`, `grid_runner.py`의 per-category `forgetting`) 전부
**이미 계산된 결과를 다르게 표시만** 하는 순수 리포팅 파생값이라 A/B 없이도
안전하게 통일할 수 있다(6절의 "실증적 검증 우선"은 실험 자체를 바꾸는
변경에 적용되는 원칙이지, 이미 계산된 R-matrix를 어떤 부호로 표시할지는
해당하지 않는다).

`bwt()`의 학계 표준 부호(Lopez-Paz & Ranzato, 음수=망각/양수=개선)로 전부
맞췄다 — `forgetting_exp{j}`→`bwt_exp{j}`, `no_cl_forgetting_exp{j}`→
`no_cl_bwt_exp{j}`, per-category `forgetting`→`bwt`로 필드명도 함께
바꿔 이름과 부호가 항상 일치하게 했다(`grid_runner.py`/
`leaderboard_builder.py`/`test_leaderboard_builder.py` 수정). 7절의 해당
항목은 "해소됨"으로 갱신했다 — 이제 이 저장소 어디서나 "음수=망각"이
예외 없이 성립한다.

**같은 대화에서 함께 확인한 것(코드 변경 없음)**:
- **96→78 그리드 축소의 이유**: RNG 오염 격리가 먼저 끝나야 "이 두 콤보가
  정말 완전히 같은 결과를 낸다"를 실행으로 증명할 수 있었고, 그렇게 확인된
  진짜 중복만 제거한 것 — 임의 축소가 아니라 순서상 RNG 격리 이후에나
  가능했던 정리였다는 걸 아티팩트에 추가 설명했다.
- **i-Blurry가 이미 구현돼 있다는 걸 아티팩트가 안 보여주고 있었음**:
  `blurry_ratio` 옵션(Koh et al. ICLR 2022, 기본 0.0=순수 class-incremental,
  0.3=일부 재등장)이 이미 코드에 있는데 개요 문서가 순수
  class-incremental만 설명해 마치 아직 안 된 것처럼 보였다 — 아티팩트에
  전용 카드로 추가해 바로잡았다. 데이터셋마다 라운드 수가 다른 이유(공통
  고정값 대신 실제 공격 유형 수로 자동 결정 — 2절/5절 근거와 동일)도 같이
  명확히 했다.

**테스트 스위트**: `test_leaderboard_builder.py`의 필드명·부호를 함께
갱신, 61/61 유지.
