"""`validate_result()`가 실제로 "마지막 방어선"으로 기능하는지 고정한다 —
2026-09-14 Round-3 감사로 발견: 타입만 검사하고 값의 범위(f1>1.0 등 그
지표의 정의상 불가능한 값)는 전혀 검사하지 않던 공백이 있었다. `common/
metrics.py`의 현재 구현은 이런 값을 낼 수 없지만, 이 함수의 존재 이유
자체가 "미래에 계산 경로 어딘가에 회귀가 생겨도 results/*.json·리더보드까지
조용히 새지 않게 막는 것"이므로 그 계약을 회귀 테스트로 고정해둔다."""

import pytest

from testbed.common.result_schema import REQUIRED_RESULT_FIELDS, validate_result


def _minimal_valid_result():
    """REQUIRED_RESULT_FIELDS 전부를 채운, 모든 검사를 통과하는 최소 결과."""
    result = {}
    for key, expected_type in REQUIRED_RESULT_FIELDS.items():
        if expected_type is str:
            result[key] = "x"
        elif expected_type is int:
            result[key] = 0
        elif expected_type is float:
            result[key] = 0.0
        elif expected_type is list:
            result[key] = []
        elif expected_type is dict:
            result[key] = {}
        else:  # pragma: no cover - REQUIRED_RESULT_FIELDS에 새 타입이 추가되면 여기도 갱신
            raise AssertionError(f"알 수 없는 필드 타입: {key}={expected_type}")
    return result


def test_minimal_valid_result_passes():
    validate_result(_minimal_valid_result())  # 예외 없이 통과해야 함


def test_missing_field_rejected():
    result = _minimal_valid_result()
    del result["f1"]
    with pytest.raises(ValueError, match="missing required fields"):
        validate_result(result)


def test_wrong_type_rejected():
    result = _minimal_valid_result()
    result["f1"] = "0.5"  # str이 아니라 float이어야 함
    with pytest.raises(ValueError, match="wrong type"):
        validate_result(result)


@pytest.mark.parametrize("field,bad_value", [
    ("f1", 1.5), ("f1", -0.1),
    ("precision", 1.0001), ("recall", -0.0001),
    ("pr_auc", 2.0), ("labeling_cost", 1.5),
    ("bwt", -1.5), ("bwt", 1.5),
    ("training_time_sec", -1.0), ("avg_inference_latency_ms", -0.1),
    ("memory_footprint", -1), ("memory_footprint_peak", -1),
    ("memory_footprint_avg", -0.1), ("n_drift_detected", -1),
])
def test_out_of_range_value_rejected(field, bad_value):
    result = _minimal_valid_result()
    result[field] = bad_value
    with pytest.raises(ValueError, match="out-of-range"):
        validate_result(result)


@pytest.mark.parametrize("field", ["roc_auc", "best_f1_reference", "normal_fpr"])
def test_nan_allowed_only_for_undefined_marker_fields(field):
    """roc_auc/best_f1_reference/normal_fpr은 "정의되지 않음"을 뜻하는 NaN이
    이 프로젝트의 명시적 관례(단일 클래스뿐인 평가 세트, Track B, category
    정보 없음)라 허용해야 한다."""
    result = _minimal_valid_result()
    result[field] = float("nan")
    validate_result(result)  # 예외 없이 통과해야 함


def test_nan_rejected_for_fields_without_undefined_convention():
    """f1처럼 NaN이 "정의되지 않음"을 뜻하는 관례가 없는 필드는 NaN도
    타입은 float이라 타입 검사만으로는 못 잡는다 — 범위 검사가 잡아야 한다."""
    result = _minimal_valid_result()
    result["f1"] = float("nan")
    with pytest.raises(ValueError, match="out-of-range"):
        validate_result(result)


def test_boundary_values_accepted():
    """0.0/1.0(또는 -1.0/1.0) 경계값 자체는 거부하면 안 된다 — 부등식이
    엄격 부등호로 잘못 바뀌는 회귀를 잡는다."""
    result = _minimal_valid_result()
    result["f1"] = 1.0
    result["precision"] = 0.0
    result["bwt"] = -1.0
    validate_result(result)
