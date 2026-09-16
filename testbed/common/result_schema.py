"""결과 스키마 — PRD 14.1절.

results/{combo_id}.json은 REQUIRED_RESULT_FIELDS 전체를 포함해야 한다.
combo_id는 5개 슬롯 값으로 결정론적으로 구성한다(같은 조합이면 항상 같은
ID가 나온다).
"""

import math
from typing import Any, Dict

REQUIRED_RESULT_FIELDS = {
    "combo_id": str,
    "exp_name": str, "dataset": str,
    "drift_detector": str, "sample_selector": str, "memory_manager": str,
    "anti_forgetting": str, "anomaly_scorer": str,
    "track": str,
    "f1": float, "precision": float, "recall": float, "pr_auc": float,
    "bwt": float, "perf_matrix": list,
    "roc_auc": float, "labeling_cost": float, "training_time_sec": float,
    "memory_footprint": int, "memory_footprint_peak": int,
    "memory_footprint_avg": float, "avg_inference_latency_ms": float,
    "best_f1_reference": float, "seed": int,
    "code_version": str,
    # 부가 분석 필드(리더보드 정렬에는 쓰지 않음, 학습 경로 무변경 —
    # experiments/grid_runner.py 참고).
    "drift_detected_per_round": list, "drift_score_per_round": list,
    "n_drift_detected": int,
    "per_category_final": dict, "per_category_recall_history": dict,
    "normal_fpr": float, "normal_fpr_history": list,
}
# memory_footprint_peak/avg, code_version: SPIDER 등 라운드마다 버퍼가
# 요동치는 memory_manager는 마지막 라운드 스냅샷만으론 오해를 살 수 있고,
# 결과 캐싱이 코드 버전을 검사하지 않으면 낡은 결과가 재사용될 위험이
# 있다(grid_runner.py 참고).
# drift_*/n_drift_detected/per_category_*/normal_fpr*: (1) 데이터셋·조합별
# drift 감지 횟수, (2) 공격 category별 recall과 라운드에 따른 망각. 둘 다
# CLClient.run_experience()가 이미 돌려주던 값(drift_detected/eval_scores)을
# grid_runner가 집계만 한 것이라 학습 과정에는 영향이 없다.


def make_combo_id(combo: Dict[str, Any]) -> str:
    return (
        f"{combo['track']}_dd={combo['drift_detector']}_ss={combo['sample_selector']}"
        f"_mm={combo['memory_manager']}_af={combo['anti_forgetting']}"
        f"_as={combo['anomaly_scorer']}"
    )


def validate_result(result: Dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_RESULT_FIELDS if k not in result]
    if missing:
        raise ValueError(f"Result missing required fields: {missing}")
    # 2026-09-14 재검토 전까지는 REQUIRED_RESULT_FIELDS의 타입이 키 존재만
    # 확인될 뿐 실제로 검사되지 않는 장식적 문서였다 — 실제 타입 검사를
    # 추가했다. bool은 int의 서브클래스라 isinstance(True, int)가 True를
    # 반환하는 파이썬 함정을 피하려 명시적으로 제외했고, float 필드는
    # JSON 왕복 후 정수로 직렬화될 수 있어(예: bwt=0.0이 아니라 실수로
    # bwt=0로 저장된 경우) int도 허용한다. 기존 결과 96개(results/*.json,
    # NSL-KDD 전체 그리드)로 드라이런해 위반 0건을 확인한 뒤 반영했다.
    type_errors = []
    for k, expected in REQUIRED_RESULT_FIELDS.items():
        v = result[k]
        if expected is float:
            ok = isinstance(v, (int, float)) and not isinstance(v, bool)
        elif expected is int:
            ok = isinstance(v, int) and not isinstance(v, bool)
        else:
            ok = isinstance(v, expected)
        if not ok:
            type_errors.append(f"{k}: expected {expected.__name__}, got {type(v).__name__}")
    if type_errors:
        raise ValueError(f"Result has fields with wrong type: {type_errors}")

    # 2026-09-14 Round-3 감사로 발견·추가 — 위 타입 검사는 f1=1.5나
    # pr_auc=-0.3처럼 "타입은 float이지만 그 지표의 정의상 불가능한 값"을
    # 전혀 못 잡는다("검증"이라는 함수 이름이 실제로 구현 못 하는 것을
    # 약속하고 있었다). `common/metrics.py`의 현재 구현은 이런 값을 낼 수
    # 없지만(직접 확인됨), 이 함수의 존재 이유 자체가 "미래에 metrics.py나
    # 다른 계산 경로에 회귀가 생겨도 results/*.json·리더보드까지 조용히
    # 새지 않게 막는 마지막 방어선"이므로 지금 값이 안전하다는 사실이 이
    # 검사를 생략할 근거가 되지 않는다. NaN은 "정의되지 않음"을 나타내는
    # 이 프로젝트의 명시적 관례(roc_auc/best_f1_reference/normal_fpr —
    # 단일 클래스뿐인 평가 세트, Track B, category 정보 없음)라 해당
    # 필드에서만 허용한다.
    _RANGE_CHECKS = {
        "f1": (0.0, 1.0, False), "precision": (0.0, 1.0, False),
        "recall": (0.0, 1.0, False), "pr_auc": (0.0, 1.0, False),
        "labeling_cost": (0.0, 1.0, False),
        "bwt": (-1.0, 1.0, False),
        "roc_auc": (0.0, 1.0, True),
        "best_f1_reference": (0.0, 1.0, True),
        "normal_fpr": (0.0, 1.0, True),
        "training_time_sec": (0.0, math.inf, False),
        "avg_inference_latency_ms": (0.0, math.inf, False),
        "memory_footprint": (0, math.inf, False),
        "memory_footprint_peak": (0, math.inf, False),
        "memory_footprint_avg": (0.0, math.inf, False),
        "n_drift_detected": (0, math.inf, False),
    }
    range_errors = []
    for k, (low, high, nan_ok) in _RANGE_CHECKS.items():
        v = result[k]
        if isinstance(v, float) and math.isnan(v):
            if not nan_ok:
                range_errors.append(f"{k}: NaN이 허용되지 않는 필드인데 NaN임")
            continue
        if not (low <= v <= high):
            range_errors.append(f"{k}: [{low}, {high}] 범위를 벗어남 (실제값: {v})")
    if range_errors:
        raise ValueError(f"Result has fields with out-of-range values: {range_errors}")
