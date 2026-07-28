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
    "memory_footprint": int, "avg_inference_latency_ms": float,
    "best_f1_reference": float, "seed": int,
}


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
