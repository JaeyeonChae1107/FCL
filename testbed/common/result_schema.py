"""결과 스키마 — PRD 14.1절.

results/{combo_id}.json은 REQUIRED_RESULT_FIELDS 전체를 포함해야 한다.
combo_id는 5개 슬롯 값으로 결정론적으로 구성한다(같은 조합이면 항상 같은
ID가 나온다).
"""

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
}
# memory_footprint_peak/avg, code_version은 2026-08-14 추가(구조 전수 감사에서
# 발견: SPIDER 등 라운드마다 버퍼가 요동치는 memory_manager는 마지막 라운드
# 스냅샷만으론 오해를 살 수 있고, 결과 캐싱이 코드 버전을 검사하지 않아 낡은
# 결과가 재사용될 위험이 있었다 — grid_runner.py 참고).


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
