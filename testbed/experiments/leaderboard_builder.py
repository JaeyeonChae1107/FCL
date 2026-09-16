"""리더보드 생성 — PRD Phase 4.

스모크 테스트를 통과한 조합의 results/*.json을 모아 F1 내림차순(동률 시
PR-AUC 내림차순)으로 정렬한 reports/leaderboard_{dataset}.csv를 **데이터셋별로
따로** 만든다. Track A/B/전체 1위를 명시하고, 상위 5개는 BWT(+Track A는
best_f1_reference)도 함께 표시한다.

**데이터셋을 섞지 않는 이유**: NSL-KDD/UNSW-NB15는 규모·난이도가
서로 크게 달라(예: 같은 조합이라도 F1이 데이터셋마다 0.6대~0.9대까지 벌어짐,
docs/metric_justification.md 참고) 두 데이터셋 결과를 하나의 표에 섞어 정렬하면
"어느 조합이 최선인가"가 아니라 "어느 조합이 우연히 가장 쉬운 데이터셋에
배정됐는가"를 보게 된다. 그래서 데이터셋마다 독립적으로 리더보드를 만들고,
비교는 반드시 같은 데이터셋 내에서만 한다.
"""

import glob
import io
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# 2026-09-14 추가 — Windows 콘솔(cp949 등)에서 print()가 em-dash(—) 같은
# 문자를 만나면 UnicodeEncodeError로 죽는다(experiments/smoke_test.py 모듈
# docstring 참고, X-IIoTID 스모크에서 실제로 재현됨). print_report()도 한글
# 메시지를 많이 출력하므로 동일하게 방어한다.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")

from testbed.common.compatibility import NO_CL_BASELINE_COMBO, enumerate_valid_combos
from testbed.common.result_schema import make_combo_id

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTBED_ROOT = os.path.dirname(_HERE)
RESULTS_DIR = os.path.join(_TESTBED_ROOT, "results")
REPORTS_DIR = os.path.join(_TESTBED_ROOT, "reports")

SUMMARY_COLS = [
    "combo_id", "exp_name", "dataset", "drift_detector", "sample_selector",
    "memory_manager", "anti_forgetting", "anomaly_scorer", "track",
    "f1", "precision", "recall", "pr_auc", "bwt", "n_drift_detected",
    "f1_delta_vs_no_cl", "bwt_delta_vs_no_cl",
]
# summary_*.csv가 "한 화면에서 훑어볼 수 있는" 요약이라는 목적을 지키려면
# drift 횟수와 no-CL 대비 차이도 별도 파일을 열지 않고 여기서 보여야 한다.


def load_all_results() -> List[Dict[str, Any]]:
    """results/*.json 중 `enumerate_valid_combos()`가 지금 이 순간 유효하다고
    보는 조합만 읽어온다. 그리드가 바뀌면(compatibility.py) 예전에 생성된
    무효 조합 결과 파일이 results/에 그대로 남아있을 수 있는데, 그걸 걸러내지
    않으면 더 이상 유효하지 않은 조합이 리더보드에 다시 섞여 들어간다 —
    실제로 drift_detector 그리드 축소 때 두 번 발생한 문제라 여기서 근본적으로
    막는다."""
    valid_ids = {make_combo_id(c) for c in enumerate_valid_combos()}
    results = []
    for path in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        with io.open(path, encoding="utf-8") as f:
            result = json.load(f)
        if result.get("combo_id") in valid_ids:
            results.append(result)
    return results


def load_joint_baseline(dataset_name: str) -> Optional[Dict[str, Any]]:
    """6.2 Joint/Offline 상한선(`grid_runner.py run_joint_baseline()`)
    결과를 읽는다. `combo_id="JOINT_BASELINE"`이 `enumerate_valid_combos()`
    의 유효 조합(78개)에 속하지 않아 `load_all_results()`가 걸러내므로(의도된
    동작 — 그리드와 섞이면 안 됨: perf_matrix가 1×T로 다른 조합의
    T×T와 모양이 다르고 5-슬롯 값도 전부 "joint" 마커라 정렬/비교 대상이
    아님), 별도 경로로 직접 읽는다. 2026-09-14 재검토 전까지는 계산만 되고
    어디에도 리포트되지 않던 항목 — main()/print_report()에서 참고용으로만
    표시한다."""
    path = os.path.join(RESULTS_DIR, f"JOINT_BASELINE__{dataset_name}.json")
    if not os.path.exists(path):
        return None
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def build_leaderboard(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """F1 내림차순, 동률 시 PR-AUC 내림차순 (PRD 3.1절 leaderboard 정렬 기준)."""
    df = pd.DataFrame(results)
    df = df.sort_values(by=["f1", "pr_auc"], ascending=[False, False]).reset_index(drop=True)
    return df


CHART_COLS = [
    "combo_id", "dataset", "track", "drift_detector", "sample_selector",
    "memory_manager", "anti_forgetting", "anomaly_scorer",
    "f1", "precision", "recall", "pr_auc", "bwt",
    "n_drift_detected", "is_no_cl_baseline", "f1_delta_vs_no_cl", "bwt_delta_vs_no_cl",
]


def attach_no_cl_deltas(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """"지속학습을 전혀 쓰지 않은" 기준선(common/compatibility.py
    NO_CL_BASELINE_COMBO, 사용자 결정) 대비 각 조합의 f1/pr_auc/bwt 차이를
    계산한다. 새로 실행하지 않는다 — 그 기준선도
    enumerate_valid_combos()에 포함된 그리드의 일부
    (A_dd=none_ss=random_mm=none_af=none_as=none)라 results/에 이미 있는 값을
    그대로 찾아 쓴다.
    """
    df = df.copy()
    baseline_id = make_combo_id(NO_CL_BASELINE_COMBO)
    baseline_rows = df[df["combo_id"] == baseline_id]
    df["is_no_cl_baseline"] = df["combo_id"] == baseline_id
    if len(baseline_rows) == 0:
        print(f"경고: [{dataset_name}] no-CL 기준선({baseline_id})의 결과가 "
              f"results/에 없습니다 - f1_delta_vs_no_cl 등은 NaN으로 둡니다. "
              f"grid_runner.py로 이 조합을 먼저 실행하세요.")
        df["f1_delta_vs_no_cl"] = float("nan")
        df["pr_auc_delta_vs_no_cl"] = float("nan")
        df["bwt_delta_vs_no_cl"] = float("nan")
        return df
    baseline = baseline_rows.iloc[0]
    df["f1_delta_vs_no_cl"] = df["f1"] - baseline["f1"]
    df["pr_auc_delta_vs_no_cl"] = df["pr_auc"] - baseline["pr_auc"]
    df["bwt_delta_vs_no_cl"] = df["bwt"] - baseline["bwt"]
    return df


def write_drift_summary(df: pd.DataFrame, dataset_name: str) -> None:
    """(drift_detector, sample_selector, memory_manager)별 drift 감지 횟수와
    라운드별 감지율을 집계한다 — "데이터셋별로 drift detection을 몇 번
    했는지" 질문에 답하기 위한 리포트.
    drift_detected_per_round는 CLClient.run_experience()가 이미 매 라운드
    돌려주던 값을 grid_runner.py가 기록만 한 것이다(학습 경로 무변경).

    해석 주의(CSV에도 주석으로 남긴다): 라운드 0은 비교할 이전 버퍼가 없어
    구조적으로 감지 False, memory_manager='none'이면 buf_ref가 항상 None이라
    전체 라운드가 False(base/drift_detector.py 계약), drift_detector='none'은
    애초에 항상 False를 반환한다. 실질적으로 의미 있는 신호는 Track A의
    drift_detector∈{ssf,cade} × memory_manager∈{spider,ssf} 조합에서만 나온다
    (common/compatibility.py TRACK_A_DD_ACTIVE_SS_MM 참고).
    """
    if "drift_detected_per_round" not in df.columns:
        print(f"[{dataset_name}] drift_detected_per_round 필드가 없어 "
              f"drift_summary를 건너뜁니다(옛 code_version 결과로 보입니다).")
        return
    rows = []
    group_cols = ["drift_detector", "sample_selector", "memory_manager"]
    n_rounds = 0
    for keys, group in df.groupby(group_cols):
        histories = [h for h in group["drift_detected_per_round"] if isinstance(h, list) and h]
        scores = [h for h in group["drift_score_per_round"] if isinstance(h, list) and h]
        n_rounds = max(n_rounds, max((len(h) for h in histories), default=0))
        row = {
            "drift_detector": keys[0], "sample_selector": keys[1], "memory_manager": keys[2],
            "n_combos": len(group),
            "n_drift_detected_mean": float(group["n_drift_detected"].mean()),
            "n_drift_detected_max": int(group["n_drift_detected"].max()),
        }
        for t in range(n_rounds):
            rates = [h[t] for h in histories if len(h) > t]
            row[f"detect_rate_exp{t}"] = float(np.mean(rates)) if rates else float("nan")
            round_scores = [sc[t] for sc in scores if len(sc) > t]
            row[f"avg_drift_score_exp{t}"] = float(np.mean(round_scores)) if round_scores else float("nan")
        rows.append(row)
    out_df = pd.DataFrame(rows).sort_values(
        by=["drift_detector", "sample_selector", "memory_manager"]).reset_index(drop=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, f"drift_summary_{dataset_name}.csv")
    with io.open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        f.write("# detect_rate_exp0/mm=none 조합은 구조적으로 항상 0 "
                "(write_drift_summary() docstring 참고)\n")
        out_df.to_csv(f, index=False)


def write_no_cl_comparison(df: pd.DataFrame, dataset_name: str) -> None:
    """no-CL 기준선(NO_CL_BASELINE_COMBO) 대비 f1/pr_auc/bwt 차이와, 라운드별
    BWT(초반 라운드 성능 대비 마지막 모델의 그 라운드 성능 변화)를 조합별로
    나란히 보여준다. perf_matrix(R행렬)는 grid_runner.py가
    이미 계산해 저장한 값을 그대로 쓴다 — 추가 학습/평가 없음.

    **부호 규약(2026-09-15 통일)**: `bwt_exp{j}` = R[마지막 라운드][j] -
    R[j][j] — `common/metrics.py`의 `bwt()`와 정확히 같은 부호(음수=망각,
    양수=개선). 예전엔 이 열 이름이 `forgetting_exp{j}`이고 부호가 반대
    (R[j][j] - R[마지막][j], 양수=망각)였는데, `bwt()`·리더보드의 `bwt`
    컬럼과 방향이 어긋나 있었다(design_decisions.md 7절이 지적한 "BWT 부호
    규약이 파일마다 다르다"는 문제). 프로젝트 전체에서 "음수=망각"으로
    통일했다 — 더 이상 방향을 따로 캡션에 명시할 필요가 없다.

    Track B(라벨-프리, experience 전체 사용)는 Track A 기준선과 학습 조건
    자체가 다르므로(configs/global_hparams.yaml 참고) f1/bwt 차이는
    "지속학습 유무"뿐 아니라 "라벨 예산 유무"까지 함께 반영된 값이다 — CSV
    주석과 print_report()에서 이 점을 명시한다.
    """
    if "is_no_cl_baseline" not in df.columns or not df["is_no_cl_baseline"].any():
        print(f"[{dataset_name}] no-CL 기준선 결과가 없어 no_cl_comparison을 건너뜁니다.")
        return
    baseline = df[df["is_no_cl_baseline"]].iloc[0]
    baseline_R = np.asarray(baseline["perf_matrix"], dtype=float)
    n_rounds = baseline_R.shape[0]

    rows = []
    for _, r in df.iterrows():
        R = np.asarray(r["perf_matrix"], dtype=float)
        row = {
            "combo_id": r["combo_id"], "track": r["track"],
            "f1": r["f1"], "bwt": r["bwt"],
            "f1_delta_vs_no_cl": r["f1_delta_vs_no_cl"],
            "pr_auc_delta_vs_no_cl": r["pr_auc_delta_vs_no_cl"],
            "bwt_delta_vs_no_cl": r["bwt_delta_vs_no_cl"],
        }
        for j in range(min(n_rounds, R.shape[0])):
            row[f"final_f1_exp{j}"] = float(R[-1][j])
            row[f"bwt_exp{j}"] = float(R[-1][j] - R[j][j])
            row[f"no_cl_final_f1_exp{j}"] = float(baseline_R[-1][j])
            row[f"no_cl_bwt_exp{j}"] = float(baseline_R[-1][j] - baseline_R[j][j])
        rows.append(row)
    out_df = pd.DataFrame(rows).sort_values(by="f1", ascending=False).reset_index(drop=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, f"no_cl_comparison_{dataset_name}.csv")
    with io.open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(f"# 기준선: {baseline['combo_id']} (backbone만 BCE로 계속 학습, "
                "drift/메모리/망각방지/별도 scorer 없음). Track B 조합은 라벨 예산 "
                "없이 experience 전체를 쓰므로 학습 조건 자체가 다르다. "
                "bwt_exp{j}는 음수=망각/양수=개선(common/metrics.py의 bwt()와 "
                "동일 부호) "
                "(write_no_cl_comparison() docstring 참고)\n")
        out_df.to_csv(f, index=False)


def write_per_category_reports(results: List[Dict[str, Any]], dataset_name: str) -> None:
    """공격 category별 recall/망각과, category별 난이도(전 조합 평균)를
    리포트한다. data/dataset_loader.py가 보존하는
    test_category와, grid_runner.py가 이미 계산해 둔 eval_scores/threshold만
    재사용한다 — 추가 forward/학습 없음. per_category_final이 없는(구
    code_version) 결과는 건너뛴다.
    """
    long_rows = []
    for r in results:
        pcf = r.get("per_category_final")
        if not isinstance(pcf, dict) or not pcf:
            continue
        for cat, stats in pcf.items():
            long_rows.append({
                "combo_id": r["combo_id"], "track": r["track"], "category": cat,
                **stats,
            })
    if not long_rows:
        print(f"[{dataset_name}] per_category_final 필드가 있는 결과가 없어 "
              f"per_category 리포트를 건너뜁니다(옛 code_version 결과로 보입니다).")
        return
    os.makedirs(REPORTS_DIR, exist_ok=True)
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(os.path.join(REPORTS_DIR, f"per_category_{dataset_name}.csv"),
                    index=False, encoding="utf-8-sig")

    f1_by_combo = {r["combo_id"]: r["f1"] for r in results}
    normal_fpr_by_combo = {r["combo_id"]: r.get("normal_fpr", float("nan")) for r in results}
    pivot = long_df.pivot_table(index="combo_id", columns="category", values="recall_final")
    pivot["normal_fpr"] = pivot.index.map(normal_fpr_by_combo)
    pivot["f1"] = pivot.index.map(f1_by_combo)
    pivot = pivot.sort_values(by="f1", ascending=False).drop(columns=["f1"])
    pivot.to_csv(os.path.join(REPORTS_DIR, f"per_category_pivot_{dataset_name}.csv"),
                 encoding="utf-8-sig")

    difficulty = long_df.groupby("category")["recall_final"].agg(
        recall_final_mean="mean", recall_final_min="min", recall_final_max="max",
        n_combos="count").reset_index()
    best_combo = long_df.loc[long_df.groupby("category")["recall_final"].idxmax(),
                              ["category", "combo_id", "recall_final"]].rename(
        columns={"combo_id": "best_combo_id", "recall_final": "best_recall_final"})
    difficulty = difficulty.merge(best_combo, on="category").sort_values(
        by="recall_final_mean").reset_index(drop=True)
    difficulty.to_csv(os.path.join(REPORTS_DIR, f"category_difficulty_{dataset_name}.csv"),
                       index=False, encoding="utf-8-sig")


def write_reports(df: pd.DataFrame, dataset_name: str) -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(REPORTS_DIR, f"leaderboard_{dataset_name}.csv"),
              index=False, encoding="utf-8-sig")
    # df[SUMMARY_COLS]/df[CHART_COLS] 대신 df.reindex(columns=...) 사용 —
    # results/에 이 필드들이 생기기 전(구 code_version) 결과만 섞여 있으면
    # 그 컬럼이 df에 없어 df[cols]가 KeyError로 죽는다. reindex는 없는
    # 컬럼을 NaN으로 채울 뿐 죽지 않는다.
    df.reindex(columns=SUMMARY_COLS).to_csv(
        os.path.join(REPORTS_DIR, f"summary_{dataset_name}.csv"),
        index=False, encoding="utf-8-sig")
    # HTML 아티팩트(리더보드 대시보드)가 그대로 읽어들이는 경량 JSON — DATA 배열.
    chart_records = df.reindex(columns=CHART_COLS).round(4).to_dict(orient="records")
    with io.open(os.path.join(REPORTS_DIR, f"leaderboard_for_chart_{dataset_name}.json"), "w",
                 encoding="utf-8") as f:
        json.dump(chart_records, f, ensure_ascii=False, indent=2)


def print_report(df: pd.DataFrame, dataset_name: str,
                  joint_baseline: Optional[Dict[str, Any]] = None) -> None:
    print("=" * 70)
    print(f"CL-NIDS-Bench v2.5 리더보드 - {dataset_name}")
    print("=" * 70)
    print("주의: 이 결과는 SSF/SPIDER/CADE/CND-IDS 각 논문의 원 보고 수치를")
    print("재현한 것이 아니라, 공통 인프라 위에서 구성요소를 재조합해 비교한")
    print("것이다(PRD 0절). 각 논문의 원 수치와 달라도 그 자체는 결함이 아니다.")
    print("데이터셋마다 규모/난이도가 달라 다른 데이터셋 결과와 직접 비교하지")
    print("않는다 - 비교는 항상 이 데이터셋 안에서만 유효하다.")
    print()

    overall_top = df.iloc[0]
    print(f"[전체 1위] {overall_top['combo_id']} "
          f"F1={overall_top['f1']:.4f} PR-AUC={overall_top['pr_auc']:.4f}")

    for track in ["A", "B"]:
        track_df = df[df["track"] == track]
        if len(track_df) == 0:
            continue
        top = track_df.iloc[0]
        print(f"[Track {track} 1위] {top['combo_id']} "
              f"F1={top['f1']:.4f} PR-AUC={top['pr_auc']:.4f}")

    print()
    print("참고: Track A(semi-supervised discriminative)는 라벨 예산(10%) 안의")
    print("데이터만, Track B(label-free novelty detection)는 experience 전체를")
    print("라벨 없이 학습에 쓴다 - 학습 조건은 다르지만 평가지표(F1/PR-AUC/BWT)는")
    print("동일한 프로토콜로 채점되므로 위 순위는 두 Track을 함께 비교한 것이다.")
    print()

    # F1/PR-AUC/BWT 한 줄에 drift 감지 횟수와 no-CL 기준선 대비 ΔF1/ΔBWT까지
    # 한 번에 보여준다. 값이 없는(구 code_version) 결과는 그 항목만 생략한다.
    print("상위 5개 조합:")
    for _, row in df.head(5).iterrows():
        parts = [f"F1={row['f1']:.4f}", f"PR-AUC={row['pr_auc']:.4f}", f"BWT={row['bwt']:.4f}"]
        if row["track"] == "A" and pd.notna(row.get("best_f1_reference")):
            parts.append(f"best_f1_reference={row['best_f1_reference']:.4f}")
        if pd.notna(row.get("n_drift_detected")):
            parts.append(f"drift={row['n_drift_detected']:.0f}회")
        if pd.notna(row.get("f1_delta_vs_no_cl")):
            parts.append(f"ΔF1vsNoCL={row['f1_delta_vs_no_cl']:+.4f}")
            parts.append(f"ΔBWTvsNoCL={row['bwt_delta_vs_no_cl']:+.4f}")
        print(f"  {row['combo_id']}: " + " ".join(parts))
    print()

    # "어떤 공격을 잘/못 감지하는지"를 전체 1위와 no-CL 기준선에 대해 콘솔에서
    # 바로 보여준다(상세는 reports/per_category_*.csv).
    def _print_category_extremes(label: str, row: "pd.Series") -> None:
        pcf = row.get("per_category_final")
        if not isinstance(pcf, dict) or not pcf:
            return
        ranked = sorted(pcf.items(), key=lambda kv: kv[1]["recall_final"], reverse=True)
        best = ", ".join(f"{c}(recall={s['recall_final']:.2f})" for c, s in ranked[:2])
        worst = ", ".join(f"{c}(recall={s['recall_final']:.2f})" for c, s in ranked[-2:])
        print(f"  [{label}] 잘 감지: {best} / 못 감지: {worst}")

    if "per_category_final" in df.columns:
        _print_category_extremes(f"1위 {overall_top['combo_id']}", overall_top)
        if "is_no_cl_baseline" in df.columns and df["is_no_cl_baseline"].any():
            _print_category_extremes(
                f"no-CL 기준선 {df[df['is_no_cl_baseline']].iloc[0]['combo_id']}",
                df[df["is_no_cl_baseline"]].iloc[0])
        print()

    # no-CL 기준선은 top-5 밖에 있는 경우가 많아 별도로 명시한다.
    if "is_no_cl_baseline" in df.columns and df["is_no_cl_baseline"].any():
        baseline = df[df["is_no_cl_baseline"]].iloc[0]
        baseline_rank = int(df.index[df["is_no_cl_baseline"]][0]) + 1
        print(f"[지속학습 없음 기준선] {baseline['combo_id']} "
              f"F1={baseline['f1']:.4f} BWT={baseline['bwt']:.4f} "
              f"(전체 {len(df)}개 중 {baseline_rank}위) "
              f"- 상세 라운드별 망각 비교는 reports/no_cl_comparison_{dataset_name}.csv 참고")
        print()

    # 6.2 Joint/Offline 상한선 — 그리드 밖의 참고용 기준선이라 순위표
    # (leaderboard/summary)에는 섞지 않고 여기서만 별도로 보여준다
    # (load_joint_baseline() docstring 참고).
    if joint_baseline is not None:
        f1_gap = joint_baseline["f1"] - overall_top["f1"]
        print(f"[Joint/Offline 상한선] F1={joint_baseline['f1']:.4f} "
              f"PR-AUC={joint_baseline['pr_auc']:.4f} "
              f"(5-슬롯 구조 없이 전체 데이터를 한 번에 학습 — CL 없이 도달 "
              f"가능한 참고 상한. 전체 1위 대비 F1 격차={f1_gap:+.4f}, 양수면 "
              f"Joint가 더 높음)")
        print()

    # (drift_detector, sample_selector, memory_manager)별 drift 평균 감지
    # 횟수. 해석 주의는 write_drift_summary() docstring 참고.
    if "drift_detector" in df.columns and "n_drift_detected" in df.columns:
        print("drift 감지 평균 (dd × ss × mm):")
        group_cols = ["drift_detector", "sample_selector", "memory_manager"]
        agg = df.groupby(group_cols)["n_drift_detected"].mean().reset_index()
        for _, g in agg.sort_values(by="n_drift_detected", ascending=False).iterrows():
            print(f"  dd={g['drift_detector']} ss={g['sample_selector']} mm={g['memory_manager']}: "
                  f"{g['n_drift_detected']:.2f}회")
        print("  (mm=none 등 버퍼 없는 조합, drift_detector=none은 구조적으로 항상 0회 "
              "- 상세는 reports/drift_summary_*.csv 참고)")
        print()


def main() -> None:
    results = load_all_results()
    if not results:
        print("results/ 에 결과 파일이 없습니다. 먼저 grid_runner.py를 실행하세요.")
        return
    # 2026-09-14 재검토로 발견·수정 — 이전엔 "한 데이터셋 안에 서로 다른
    # code_version이 섞여 있는가"만 봤다. results/*.json 전부가 **하나의
    # 낡은** 버전으로 내부적으로 일관돼 있으면(예: RNG 격리·GPM 라벨예산
    # 면제 반영 이전 결과로만 채워진 상태) 이 검사는 통과해버려 아무
    # 경고 없이 낡은 결과로 리더보드를 만들었다. 현재 코드의
    # `compute_code_version()`과 직접 비교해 이 구멍을 막는다.
    try:
        from testbed.experiments.grid_runner import compute_code_version
        current_code_version = compute_code_version()
    except Exception as exc:  # pragma: no cover - 방어적, 리더보드 생성 자체는 막지 않음
        print(f"경고: 현재 code_version을 계산할 수 없어({exc}) 낡은 결과 검사를 건너뜁니다.")
        current_code_version = None
    dataset_names = sorted({r["dataset"] for r in results})
    for dataset_name in dataset_names:
        dataset_results = [r for r in results if r["dataset"] == dataset_name]
        # grid_runner.py의 run_grid()는 중간에 끊겨도 이어서 돌릴 수 있다
        # (code_version이 일치할 때만 건너뜀) — 재실행이 안 끝난 상태로 이
        # 스크립트를 돌리면 results/에 새/옛 코드 결과가 섞여 있을 수 있다.
        # 코드 버전이 섞여 있으면 경고만 낸다(자동으로 막지는 않음 — 부분
        # 결과라도 봐야 할 때가 있으므로).
        code_versions = {r.get("code_version") for r in dataset_results}
        if len(code_versions) > 1:
            print(f"경고: [{dataset_name}] 결과에 서로 다른 code_version이 섞여 "
                  f"있습니다({sorted(v for v in code_versions if v)}) - grid_runner.py "
                  f"재실행이 아직 끝나지 않았을 수 있습니다. 전체 재실행 완료 후 "
                  f"다시 생성하는 것을 권장합니다.")
        elif current_code_version is not None and code_versions != {current_code_version}:
            stale = next(iter(code_versions))
            print(f"경고: [{dataset_name}] 결과가 전부 낡은 code_version({stale!r})으로 "
                  f"계산됐습니다 — 현재 코드({current_code_version!r})와 다릅니다. "
                  "코드를 고친 뒤 grid_runner.py 재실행을 깜빡했을 수 있습니다. "
                  "`python -m testbed.experiments.grid_runner`로 재실행 후 이 리더보드를 "
                  "다시 생성하세요.")
        df = build_leaderboard(dataset_results)
        df = attach_no_cl_deltas(df, dataset_name)
        write_reports(df, dataset_name)
        write_drift_summary(df, dataset_name)
        write_no_cl_comparison(df, dataset_name)
        write_per_category_reports(dataset_results, dataset_name)
        print_report(df, dataset_name, joint_baseline=load_joint_baseline(dataset_name))


if __name__ == "__main__":
    main()
