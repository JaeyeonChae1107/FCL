"""Backbone-Type 정합성 — PRD 4절.

이 표가 이 프로젝트 전체에서 유일하게 유효한 슬롯 값 목록이다.

| 슬롯 | Track A(classifier) 전용 | Track B(autoencoder) 전용 | 공통 |
|---|---|---|---|
| drift_detector | ssf, cade | — | none |
| sample_selector | ssf | — | random |
| memory_manager | ssf | cndids | none, spider |
| anti_forgetting | lwf_ssf, gpm | cndids | none |
| anomaly_scorer | cade_mad, none | pca | — |

Track A의 anomaly_scorer에 `none`(분류기 자체 sigmoid(logit) 판정 — SSF/SPIDER
원 논문의 실제 방식)을 추가했다(사용자 지시) — CADE의 MAD 채점 방식(cade_mad)과
비교해 "CADE 채점을 빌려오는 게 실제로 도움이 되는가"를 검증하기 위함이다.

memory_manager의 `fifo`(특정 논문 근거 없는 공통 baseline)는 SPIDER 원 논문의
실제 "유한 버퍼 메모리(M)" 메커니즘(`spider` — 라벨 없는 무작위 샘플, 매
experience마다 전체 교체(No MRP), 직전 태스크 스냅샷 모델로 pseudo-labeling)
으로 대체했다. Track A/B 양쪽에서 쓸 수 있다(사용자 지시).

Track B의 anomaly_scorer는 `pca`만 남겼다 — `dif`/`lof`는 CND-IDS 원 논문
자체의 제안 방법이 아니라 CND-IDS가 비교를 위해 인용한 제3자 baseline이라
(부록A), 삭제했다(사용자 지시).

## drift_detector의 (sample_selector, memory_manager) 조건부 제약

Track A에서 `drift_detector`의 출력(get_drift_score()/detect())은 두 소비처만
있다 — `SSFSampleSelector`(drift_score로 extremity 블렌딩), `SSFMemoryManager`
(drift_detected로 eviction 비율 결정). 게다가 SSF/CADE 두 detector 모두
"비교할 과거 표본"이 있어야 의미가 있는데, `memory_manager='none'`이면
버퍼(`buf_ref`) 자체가 없어 비교 대상이 없다.

즉 `sample_selector='random'`이면서 `memory_manager`가 `'none'` 또는
`'spider'`인 경우(SPIDERMemoryManager도 drift_detected를 소비하지 않음),
그리고 `memory_manager='none'`인 경우는 `sample_selector`가 `'ssf'`여도
(버퍼가 없어 SSFSampleSelector에 넘어가는 drift_score 자체가 항상 0으로
퇴화) — 이 셋 다 `drift_detector`의 출력이 파이프라인 어디에도 영향을 줄 수
없다. 실측(leaderboard_for_chart.json)으로 확인한 결과, 이 조건에 해당하는
216개 dd-비교 쌍 중 36쌍에서 `dd=none`과 `dd=ssf`가 f1/precision/recall까지
완전히 동일했다 — **이 둘만** 진짜 무의미한 중복 비교였다(SSFDriftDetector의
K-S 검정은 비교할 과거 표본이 없으면 항상 "drift 없음"으로 퇴화하므로,
`dd=none`과 계산 결과가 항상 같아진다).

`dd=cade`는 같은 조건에서도 단 한 번도 `dd=none`과 동일해지지 않았다(0/…쌍) —
CADE의 `fit()`이 사설 contrastive AE를 실제로 학습시키며 전역 RNG를 소비해
이후 랜덤 시드 흐름을 바꿔놓기 때문에, 신호 자체는 안 쓰여도 결과값은
달라진다. 이건 "완전히 동일해지는 중복"이 아니라 "잡음으로 인한 차이"라
성격이 다르다 — 사용자 지시로, 소수점까지 완전히 일치하는 진짜 중복(`ssf`가
`none`과 겹치는 경우)만 제거하고, 값 자체가 다른 `cade`는 계속 별도
조합으로 남긴다. 그래서 `TRACK_A_DD_ACTIVE_SS_MM`에 없는(=drift_detector가
비활성인) (sample_selector, memory_manager) 조합에서는 `drift_detector`가
`'none'`과 `'cade'` 두 값만 순회하고 `'ssf'`(=`'none'`의 완전한 중복)만
제외한다 — `enumerate_valid_combos()`와 `docs/metric_justification.md` 참고.
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
}  # 슬롯별 허용값 카탈로그 (validate_combo의 값 검증에 사용). 실제 조합 수는
   # drift_detector의 조건부 제약(TRACK_A_DD_ACTIVE_SS_MM, TRACK_A_DD_INERT_VALUES)
   # 때문에 단순 itertools.product(3*3*2*3*2=108)가 아니라 90개다 —
   # enumerate_valid_combos() 참고.

# drift_detector가 실제로 소비되는 (sample_selector, memory_manager) 조합만
# drift_detector 3개 값을 전부 순회한다. 그 외(비활성) 조합에서는
# TRACK_A_DD_INERT_VALUES만 순회한다 — 위 모듈 docstring의 근거 참고.
TRACK_A_DD_ACTIVE_SS_MM: List[Tuple[str, str]] = [
    ("random", "ssf"),   # SSFMemoryManager가 drift_detected를 소비
    ("ssf", "spider"),   # SSFSampleSelector가 drift_score를 소비 (버퍼 있음)
    ("ssf", "ssf"),       # 둘 다 소비
]

# 비활성 (sample_selector, memory_manager) 조합에서 순회할 drift_detector
# 값 — 'ssf'는 이런 조합에서 'none'과 f1/precision/recall이 소수점까지
# 완전히 동일해지는 진짜 중복이라 제외한다. 'cade'는 값 자체가 'none'과
# 다르므로(RNG 오염 때문이긴 하지만 "완전히 동일"은 아님) 남긴다 — 모듈
# docstring 참고.
TRACK_A_DD_INERT_VALUES: List[str] = ["none", "cade"]

TRACK_B_GRID: Dict[str, List[str]] = {
    "anti_forgetting": ["cndids"],
    "drift_detector": ["none"],
    "sample_selector": ["random"],
    "memory_manager": ["none", "spider", "cndids"],
    "anomaly_scorer": ["pca"],
}  # itertools.product -> 1*1*1*3*1 = 3개

SLOTS = ["drift_detector", "sample_selector", "memory_manager",
         "anti_forgetting", "anomaly_scorer"]


def enumerate_valid_combos() -> List[dict]:
    """Track A(90개)와 Track B(3개)를 각각 직접 구성해 concat한 93개만
    생성한다 — "생성 후 제외" 로직은 쓰지 않는다 (PRD 4.2절).

    Track A는 (sample_selector, memory_manager)별로 drift_detector가 실제로
    소비되는 조합에서만 3개 값을 전부 순회하고, 그 외(비활성) 조합에서는
    TRACK_A_DD_INERT_VALUES(['none', 'cade'])만 순회한다 — 'ssf'는 비활성
    조합에서 'none'과 완전히 동일한 결과를 내는 진짜 중복이라 제외하되,
    'cade'는 값 자체가 다르므로(모듈 docstring 참고) 남긴다.
    """
    combos = []
    ss_mm_pairs = list(itertools.product(
        TRACK_A_GRID["sample_selector"], TRACK_A_GRID["memory_manager"]))
    for ss, mm in ss_mm_pairs:
        dd_values = (TRACK_A_GRID["drift_detector"]
                     if (ss, mm) in TRACK_A_DD_ACTIVE_SS_MM else TRACK_A_DD_INERT_VALUES)
        for dd in dd_values:
            for af in TRACK_A_GRID["anti_forgetting"]:
                for as_ in TRACK_A_GRID["anomaly_scorer"]:
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
    assert len(combos) == 93, f"expected 93 valid combos, got {len(combos)}"
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
        is_active = (ss, mm) in TRACK_A_DD_ACTIVE_SS_MM
        if not is_active and dd not in TRACK_A_DD_INERT_VALUES:
            raise IncompatibleComboError(
                f"Track A: drift_detector={dd!r}는 sample_selector={ss!r} + "
                f"memory_manager={mm!r} 조합에서 'none'과 완전히 동일한 결과를 "
                f"내는 중복이라 무효 조합이다(중복 방지 — "
                f"docs/metric_justification.md 참고). 허용값: "
                f"{TRACK_A_DD_INERT_VALUES}.")
