"""Backbone-Type 정합성 — PRD 4절. 유일하게 유효한 슬롯 값 목록.

| 슬롯 | Track A(classifier) 전용 | Track B(autoencoder) 전용 | 공통 |
|---|---|---|---|
| drift_detector | ssf, cade | — | none |
| sample_selector | ssf | — | random |
| memory_manager | ssf | cndids | none, spider |
| anti_forgetting | lwf_ssf, gpm | cndids | none |
| anomaly_scorer | none | pca | cade_mad |

- Track A `anomaly_scorer=none`: 분류기 sigmoid(logit) 직접 판정(SSF/SPIDER
  원 논문 방식). cade_mad와 비교용.
- `memory_manager=spider`: SPIDER 원 논문의 유한 버퍼 메커니즘(라벨 없는
  무작위 샘플, experience마다 전체 교체, 직전 태스크 스냅샷 모델로
  pseudo-labeling). Track A/B 공통.
- Track B `anomaly_scorer`: pca만. dif/lof는 CND-IDS가 비교용으로 인용한
  제3자 baseline(부록A)이라 제외.
- Track B `cade_mad` 추가(93→96개 조합): Track A/B가 같은
  `FCLAutoEncoder`를 공유해 z를 만든다(`base/models.py`) — CADEMADScorer가
  소비하는 z는 Track A/B에서 구조적으로 동일. threshold 계산은
  `threshold_needs_labels` 플래그(scorer 속성, `base/anomaly_scorer.py`)로
  분기하므로 Track B의 cade_mad도 s_ref 기반 median+MAD를 그대로 쓴다.

## drift_detector의 (sample_selector, memory_manager, anomaly_scorer) 조건부 제약

Track A에서 `drift_detector` 출력(get_drift_score/detect)의 소비처는 두
곳뿐이다 — `SSFSampleSelector`(drift_score), `SSFMemoryManager`
(drift_detected). `memory_manager='none'`이면 버퍼가 없어 비교 대상 자체가
없다.

`sample_selector='random'`+`memory_manager`가 `'none'`/`'spider'`, 또는
`memory_manager='none'`(sample_selector 무관) — 이 조합들에서 drift_detector
출력은 파이프라인 어디에도 영향을 주지 않는다. 실측(leaderboard_for_chart.json):
이 조건의 216개 dd-비교 쌍 중 36쌍에서 `dd=none`과 `dd=ssf`가 f1/precision/
recall까지 완전 동일(SSFDriftDetector의 K-S 검정이 비교 표본 없이 항상
"drift 없음"으로 퇴화).

**2026-09-14 재검토로 두 가지 정정(전역 RNG 오염 격리 이후 재실측)**:

1. **`("ssf", "spider")`를 활성에서 제외**: 이 쌍이 활성으로 분류된 근거는
   "SSFSampleSelector가 drift_score를 소비한다"였는데, `ssf_sample_selector.py`
   재작성(2026-09-14, SSFMaskContext 기반으로 전면 재작성) 이후 실제 코드를
   다시 보니 `drift_score`는 시그니처에만 남아있고 전혀 쓰이지 않는다
   (`_quota_select()`가 `M_t_cont`만 랭킹에 씀). `SPIDERMemoryManager.update()`
   도 `drift_detected` 인자를 받기만 하고 본문에서 안 읽는다(단순 무작위
   교체). 즉 이 쌍의 두 컴포넌트 다 drift 신호를 안 쓴다 — 실측(NSL-KDD,
   `ss=ssf/mm=spider/af=gpm/as=none`)으로 `dd=none`과 `dd=cade`가 f1/bwt/
   pr_auc 소수점 8자리까지 완전히 동일함을 확인했다.

2. **`dd=cade`의 "비활성 조합에서도 유지" 근거가 `anomaly_scorer`에 따라
   갈린다**: 예전엔 "CADE의 사설 encoder 학습이 전역 RNG를 오염시켜(신호
   자체는 안 쓰여도) 결과값이 달라진다"는 이유로 `dd=cade`를 모든 비활성
   조합에 유지했다. 이번 세션에 `torch.random.fork_rng()`로 그 RNG 오염
   자체를 격리하면서(`cade_drift_detector.py` 참고) 이 근거가 무효화됐다 —
   실측(NSL-KDD, `ss=random/mm=none/af=none/as=none` 및 `ss=ssf/mm=spider/
   af=gpm/as=none` 둘 다) `dd=none`과 `dd=cade`가 완전히 동일함을 확인.
   **단, `as=cade_mad`일 때는 여전히 다르다** — `CLClient.__init__`이
   `dd=cade`+`as=cade_mad`를 `CADEMADScorer.set_private_encoder()`로
   연결해(`uses_shared_representation`을 False로 전환) 공유 backbone
   대신 CADE의 대조학습 표현공간을 쓰게 만들기 때문이다(이건 RNG 오염과
   무관한, 의도된 구조적 연결). 실측: 같은 조합에 `as=cade_mad`만 바꾸면
   `dd=none` f1=0.7992 vs `dd=cade` f1=0.4493로 뚜렷이 다름.
   따라서 비활성 (ss,mm) 조합에서 `drift_detector`가 순회할 값은
   `anomaly_scorer`에 따라 갈린다 — `as=cade_mad`면 `['none','cade']`
   (여전히 서로 다른 결과), 그 외(`as=none`)면 `['none']`만(`'cade'`는
   이제 `'none'`과 완전 중복이라 제외 — `'ssf'`가 이미 그랬던 것과 같은
   이유).
"""

import itertools
from typing import Dict, List, Tuple


class IncompatibleComboError(Exception):
    pass


TRACK_A_GRID: Dict[str, List[str]] = {
    "anti_forgetting": ["none", "lwf_ssf", "gpm"],
    "drift_detector": ["none", "ssf", "cade"],
    "sample_selector": ["random", "ssf"],
    "memory_manager": ["none", "spider", "ssf"],
    "anomaly_scorer": ["cade_mad", "none"],
}  # 슬롯별 허용값 카탈로그. 실제 조합 수는 72개(단순 곱 108이 아님, 2026-09-14
   # 재검토로 90→72) — TRACK_A_DD_ACTIVE_SS_MM/TRACK_A_DD_INERT_VALUES_BY_AS,
   # enumerate_valid_combos() 참고.

# drift_detector 3개 값을 전부 순회하는 (sample_selector, memory_manager).
# 그 외는 TRACK_A_DD_INERT_VALUES_BY_AS만 순회 — 모듈 docstring 참고.
TRACK_A_DD_ACTIVE_SS_MM: List[Tuple[str, str]] = [
    ("random", "ssf"),   # SSFMemoryManager가 drift_detected를 소비
    ("ssf", "ssf"),       # 위와 동일 이유(SSFSampleSelector 자체는 drift_score
                           # 미소비 — 모듈 docstring 2026-09-14 정정 1번 참고)
]

# 비활성 (sample_selector, memory_manager)에서 순회할 drift_detector 값 —
# anomaly_scorer에 따라 갈린다(모듈 docstring 2026-09-14 정정 2번 참고).
# as=cade_mad면 dd=cade가 사설 encoder 연결로 여전히 dd=none과 다른 결과를
# 내므로 유지, 그 외(as=none)는 RNG 오염 격리 이후 dd=cade≡dd=none이 실측
# 확인돼 완전 중복이라 제외.
TRACK_A_DD_INERT_VALUES_BY_AS: Dict[str, List[str]] = {
    "cade_mad": ["none", "cade"],
    "none": ["none"],
}

TRACK_B_GRID: Dict[str, List[str]] = {
    "anti_forgetting": ["cndids"],
    "drift_detector": ["none"],
    "sample_selector": ["random"],
    "memory_manager": ["none", "spider", "cndids"],
    "anomaly_scorer": ["pca", "cade_mad"],
}  # itertools.product -> 6개 (cade_mad 추가)

SLOTS = ["drift_detector", "sample_selector", "memory_manager",
         "anti_forgetting", "anomaly_scorer"]

# naive fine-tuning 기준선(사용자 결정) — backbone이 매
# experience BCE로 계속 학습, drift/메모리/망각방지/별도 scorer 없음, 판정은
# 분류기 sigmoid 0.5. 그리드에 이미 포함된 조합. Track A 공통 조건(라벨 예산
# 10%, epoch 200) 그대로 받음 — Track B(라벨 없이 experience 전체 사용)와는
# 학습 조건이 다르다는 점을 리포트에서 별도 표시.
NO_CL_BASELINE_COMBO: Dict[str, str] = {
    "track": "A",
    "drift_detector": "none",
    "sample_selector": "random",
    "memory_manager": "none",
    "anti_forgetting": "none",
    "anomaly_scorer": "none",
}


def enumerate_valid_combos() -> List[dict]:
    """Track A(72개)와 Track B(6개)를 직접 구성해 concat한 78개만 생성한다
    (2026-09-14 재검토로 90→72, 96→78 — 모듈 docstring 정정 1/2번 참고).
    활성 (ss,mm)은 drift_detector 3개 값 전부, 비활성은
    `TRACK_A_DD_INERT_VALUES_BY_AS[anomaly_scorer]`만 순회한다 — dd가
    anomaly_scorer에 의존하므로 as_ 루프를 dd 루프보다 바깥에 둔다."""
    combos = []
    ss_mm_pairs = list(itertools.product(
        TRACK_A_GRID["sample_selector"], TRACK_A_GRID["memory_manager"]))
    for ss, mm in ss_mm_pairs:
        is_active = (ss, mm) in TRACK_A_DD_ACTIVE_SS_MM
        for af in TRACK_A_GRID["anti_forgetting"]:
            for as_ in TRACK_A_GRID["anomaly_scorer"]:
                dd_values = (TRACK_A_GRID["drift_detector"] if is_active
                             else TRACK_A_DD_INERT_VALUES_BY_AS[as_])
                for dd in dd_values:
                    combos.append({
                        "drift_detector": dd,
                        "sample_selector": ss,
                        "memory_manager": mm,
                        "anti_forgetting": af,
                        "anomaly_scorer": as_,
                        "track": "A",
                    })
    for values in itertools.product(*TRACK_B_GRID.values()):
        combo = dict(zip(TRACK_B_GRID.keys(), values))
        combo["track"] = "B"
        combos.append(combo)
    assert len(combos) == 78, f"expected 78 valid combos, got {len(combos)}"
    assert NO_CL_BASELINE_COMBO in combos, "NO_CL_BASELINE_COMBO가 유효 조합에 없음"
    return combos


def validate_combo(combo: dict) -> None:
    """설정 파일 등에서 사람이 직접 지정한 combo(디버깅 목적의 단일 조합 실행
    등)를 검증한다. 위반 시 IncompatibleComboError를 발생시킨다."""
    track = combo.get("track")
    if track == "A":
        grid = TRACK_A_GRID
    elif track == "B":
        grid = TRACK_B_GRID
    else:
        raise IncompatibleComboError(f"Unknown track: {track!r} (must be 'A' or 'B')")

    for slot in SLOTS:
        allowed = grid[slot]
        value = combo.get(slot)
        if value not in allowed:
            raise IncompatibleComboError(
                f"Track {track}: slot {slot!r} value {value!r} not compatible. "
                f"Allowed: {allowed}")

    if track == "A":
        ss = combo["sample_selector"]
        mm = combo["memory_manager"]
        dd = combo["drift_detector"]
        as_ = combo["anomaly_scorer"]
        is_active = (ss, mm) in TRACK_A_DD_ACTIVE_SS_MM
        allowed_inert_dd = TRACK_A_DD_INERT_VALUES_BY_AS[as_]
        if not is_active and dd not in allowed_inert_dd:
            raise IncompatibleComboError(
                f"Track A: drift_detector={dd!r}는 sample_selector={ss!r} + "
                f"memory_manager={mm!r} + anomaly_scorer={as_!r} 조합에서 "
                f"'none'과 완전히 동일한 결과를 내는 중복이라 무효 조합이다"
                f"(중복 방지 — 모듈 docstring 2026-09-14 정정 참고). 허용값: "
                f"{allowed_inert_dd}.")
