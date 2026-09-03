"""부가 분석 리포트(drift 요약·no-CL 기준선 비교·category별 recall) 검증
— 2026-09-03 추가. toy 결과 dict로 집계 로직만 확인하고, results/를 실제로
읽거나 쓰지 않는다(REPORTS_DIR을 tmp_path로 monkeypatch)."""

import pandas as pd
import pytest

from testbed.common.compatibility import NO_CL_BASELINE_COMBO
from testbed.common.result_schema import make_combo_id
from testbed.experiments import leaderboard_builder as lb

BASELINE_ID = make_combo_id(NO_CL_BASELINE_COMBO)


def _toy_result(combo_id, f1, bwt, perf_matrix, drift_detected_per_round=None,
                 drift_score_per_round=None, drift_detector="none",
                 sample_selector="random", memory_manager="none", track="A",
                 per_category_final=None, normal_fpr=0.1):
    n = len(perf_matrix)
    drift_detected_per_round = drift_detected_per_round or [False] * n
    drift_score_per_round = drift_score_per_round or [0.0] * n
    return {
        "combo_id": combo_id, "exp_name": combo_id, "dataset": "toy-ds",
        "drift_detector": drift_detector, "sample_selector": sample_selector,
        "memory_manager": memory_manager, "anti_forgetting": "none",
        "anomaly_scorer": "none", "track": track,
        "f1": f1, "precision": f1, "recall": f1, "pr_auc": f1,
        "bwt": bwt, "perf_matrix": perf_matrix, "roc_auc": 0.5,
        "labeling_cost": 0.1, "training_time_sec": 1.0,
        "memory_footprint": 0, "memory_footprint_peak": 0, "memory_footprint_avg": 0.0,
        "avg_inference_latency_ms": 0.1, "best_f1_reference": 0.0, "seed": 42,
        "code_version": "toytoytoytoy",
        "drift_detected_per_round": drift_detected_per_round,
        "drift_score_per_round": drift_score_per_round,
        "n_drift_detected": sum(drift_detected_per_round),
        "per_category_final": per_category_final or {},
        "per_category_recall_history": {},
        "normal_fpr": normal_fpr, "normal_fpr_history": [normal_fpr] * n,
    }


@pytest.fixture
def toy_results():
    baseline_perf = [[0.5, 0.0], [0.3, 0.6]]  # exp0 F1 0.5->0.3(망각), exp1 0.6
    other_perf = [[0.7, 0.0], [0.65, 0.85]]
    return [
        _toy_result(BASELINE_ID, f1=0.5, bwt=-0.2, perf_matrix=baseline_perf,
                    per_category_final={
                        "DoS": {"n_test": 100, "first_seen_round": 0,
                                "recall_at_first_seen": 0.5, "recall_final": 0.3,
                                "forgetting": 0.2},
                    }),
        _toy_result("A_dd=ssf_ss=random_mm=ssf_af=lwf_ssf_as=none", f1=0.8, bwt=0.05,
                    perf_matrix=other_perf, drift_detected_per_round=[False, True],
                    drift_score_per_round=[0.0, 0.9], drift_detector="ssf",
                    memory_manager="ssf",
                    per_category_final={
                        "DoS": {"n_test": 100, "first_seen_round": 0,
                                "recall_at_first_seen": 0.7, "recall_final": 0.85,
                                "forgetting": -0.15},
                    }),
    ]


def test_attach_no_cl_deltas_computes_signed_differences(toy_results):
    df = pd.DataFrame(toy_results)
    out = lb.attach_no_cl_deltas(df, "toy-ds")
    baseline_row = out[out["combo_id"] == BASELINE_ID].iloc[0]
    assert baseline_row["is_no_cl_baseline"]
    assert baseline_row["f1_delta_vs_no_cl"] == pytest.approx(0.0)
    assert baseline_row["bwt_delta_vs_no_cl"] == pytest.approx(0.0)

    other_row = out[out["combo_id"] != BASELINE_ID].iloc[0]
    assert not other_row["is_no_cl_baseline"]
    assert other_row["f1_delta_vs_no_cl"] == pytest.approx(0.8 - 0.5)
    assert other_row["bwt_delta_vs_no_cl"] == pytest.approx(0.05 - (-0.2))


def test_attach_no_cl_deltas_nan_when_baseline_missing():
    df = pd.DataFrame([_toy_result("A_dd=ssf_ss=random_mm=ssf_af=lwf_ssf_as=none",
                                    f1=0.8, bwt=0.05, perf_matrix=[[0.7, 0.0], [0.65, 0.85]])])
    out = lb.attach_no_cl_deltas(df, "toy-ds")
    assert out["f1_delta_vs_no_cl"].isna().all()
    assert not out["is_no_cl_baseline"].any()


def test_write_drift_summary_detect_rate_per_round(tmp_path, monkeypatch, toy_results):
    monkeypatch.setattr(lb, "REPORTS_DIR", str(tmp_path))
    df = pd.DataFrame(toy_results)
    lb.write_drift_summary(df, "toy-ds")
    out_path = tmp_path / "drift_summary_toy-ds.csv"
    assert out_path.exists()
    out_df = pd.read_csv(out_path, comment="#")
    ssf_row = out_df[out_df["drift_detector"] == "ssf"].iloc[0]
    assert ssf_row["detect_rate_exp0"] == pytest.approx(0.0)
    assert ssf_row["detect_rate_exp1"] == pytest.approx(1.0)
    none_row = out_df[out_df["drift_detector"] == "none"].iloc[0]
    assert none_row["detect_rate_exp0"] == pytest.approx(0.0)
    assert none_row["detect_rate_exp1"] == pytest.approx(0.0)


def test_write_no_cl_comparison_forgetting_columns(tmp_path, monkeypatch, toy_results):
    monkeypatch.setattr(lb, "REPORTS_DIR", str(tmp_path))
    df = lb.attach_no_cl_deltas(pd.DataFrame(toy_results), "toy-ds")
    lb.write_no_cl_comparison(df, "toy-ds")
    out_path = tmp_path / "no_cl_comparison_toy-ds.csv"
    assert out_path.exists()
    out_df = pd.read_csv(out_path, comment="#")
    baseline_row = out_df[out_df["combo_id"] == BASELINE_ID].iloc[0]
    # exp0: R[0][0]=0.5, 마지막 라운드 R[-1][0]=0.3 -> forgetting 0.2
    assert baseline_row["forgetting_exp0"] == pytest.approx(0.2)
    other_row = out_df[out_df["combo_id"] != BASELINE_ID].iloc[0]
    assert other_row["no_cl_forgetting_exp0"] == pytest.approx(0.2)


def test_write_no_cl_comparison_skipped_without_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(lb, "REPORTS_DIR", str(tmp_path))
    df = lb.attach_no_cl_deltas(
        pd.DataFrame([_toy_result("A_dd=ssf_ss=random_mm=ssf_af=lwf_ssf_as=none",
                                   f1=0.8, bwt=0.05, perf_matrix=[[0.7, 0.0], [0.65, 0.85]])]),
        "toy-ds")
    lb.write_no_cl_comparison(df, "toy-ds")
    assert not (tmp_path / "no_cl_comparison_toy-ds.csv").exists()


def test_write_per_category_reports(tmp_path, monkeypatch, toy_results):
    monkeypatch.setattr(lb, "REPORTS_DIR", str(tmp_path))
    lb.write_per_category_reports(toy_results, "toy-ds")
    long_df = pd.read_csv(tmp_path / "per_category_toy-ds.csv")
    assert set(long_df["combo_id"]) == {r["combo_id"] for r in toy_results}
    dos_baseline = long_df[(long_df["combo_id"] == BASELINE_ID) & (long_df["category"] == "DoS")].iloc[0]
    assert dos_baseline["forgetting"] == pytest.approx(0.2)

    difficulty = pd.read_csv(tmp_path / "category_difficulty_toy-ds.csv")
    dos_row = difficulty[difficulty["category"] == "DoS"].iloc[0]
    assert dos_row["recall_final_mean"] == pytest.approx((0.3 + 0.85) / 2)
    assert dos_row["best_combo_id"] == "A_dd=ssf_ss=random_mm=ssf_af=lwf_ssf_as=none"

    pivot = pd.read_csv(tmp_path / "per_category_pivot_toy-ds.csv")
    assert "DoS" in pivot.columns and "normal_fpr" in pivot.columns


def _old_style_result(combo_id, f1, bwt, perf_matrix):
    """2026-09-03 이전(부가 분석 필드 도입 전) 형태의 결과 — drift_detected_
    per_round/per_category_final 등 신규 필드가 아예 없다. write_reports()가
    SUMMARY_COLS/CHART_COLS에 이 필드들을 직접 인덱싱하면 KeyError로 죽는
    회귀가 있었다(2026-09-03 발견·수정) — reindex(columns=...)로 바꿔 없는
    컬럼은 NaN으로 채우도록 고쳤다. 이 테스트는 그 회귀가 다시 생기지
    않는지 확인한다."""
    return {
        "combo_id": combo_id, "exp_name": combo_id, "dataset": "toy-ds",
        "drift_detector": "none", "sample_selector": "random",
        "memory_manager": "none", "anti_forgetting": "none",
        "anomaly_scorer": "none", "track": "A",
        "f1": f1, "precision": f1, "recall": f1, "pr_auc": f1,
        "bwt": bwt, "perf_matrix": perf_matrix, "roc_auc": 0.5,
        "labeling_cost": 0.1, "training_time_sec": 1.0,
        "memory_footprint": 0, "memory_footprint_peak": 0, "memory_footprint_avg": 0.0,
        "avg_inference_latency_ms": 0.1, "best_f1_reference": 0.0, "seed": 42,
        "code_version": "oldoldoldold",
    }


def test_write_reports_and_print_report_survive_missing_new_fields(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(lb, "REPORTS_DIR", str(tmp_path))
    old_results = [_old_style_result("A_dd=none_ss=random_mm=none_af=none_as=none",
                                      f1=0.5, bwt=-0.1, perf_matrix=[[0.5]])]
    df = lb.build_leaderboard(old_results)
    df = lb.attach_no_cl_deltas(df, "toy-ds")
    lb.write_reports(df, "toy-ds")  # 회귀 전에는 여기서 KeyError
    lb.print_report(df, "toy-ds")  # 회귀 전에는 여기서 KeyError

    summary = pd.read_csv(tmp_path / "summary_toy-ds.csv")
    assert "n_drift_detected" in summary.columns
    assert summary["n_drift_detected"].isna().all()
    assert summary["f1_delta_vs_no_cl"].iloc[0] == pytest.approx(0.0)  # 자기 자신이 기준선
    captured = capsys.readouterr()
    assert "KeyError" not in captured.out
