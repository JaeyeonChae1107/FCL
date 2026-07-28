"""리더보드 생성 — PRD Phase 4.

스모크 테스트를 통과한 조합의 results/*.json을 모아 F1 내림차순(동률 시
PR-AUC 내림차순)으로 정렬한 reports/leaderboard.csv를 만든다. Track A/B/전체
1위를 명시하고, 상위 5개는 BWT(+Track A는 best_f1_reference)도 함께 표시한다.
"""

import glob
import io
import json
import os
from typing import Any, Dict, List

import pandas as pd

from testbed.common.compatibility import enumerate_valid_combos
from testbed.common.result_schema import make_combo_id

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTBED_ROOT = os.path.dirname(_HERE)
RESULTS_DIR = os.path.join(_TESTBED_ROOT, "results")
REPORTS_DIR = os.path.join(_TESTBED_ROOT, "reports")

SUMMARY_COLS = [
    "combo_id", "exp_name", "dataset", "drift_detector", "sample_selector",
    "memory_manager", "anti_forgetting", "anomaly_scorer", "track",
    "f1", "precision", "recall", "pr_auc",
]


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


def build_leaderboard(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """F1 내림차순, 동률 시 PR-AUC 내림차순 (PRD 3.1절 leaderboard 정렬 기준)."""
    df = pd.DataFrame(results)
    df = df.sort_values(by=["f1", "pr_auc"], ascending=[False, False]).reset_index(drop=True)
    return df


CHART_COLS = [
    "combo_id", "dataset", "track", "drift_detector", "sample_selector",
    "memory_manager", "anti_forgetting", "anomaly_scorer",
    "f1", "precision", "recall", "pr_auc", "bwt",
]


def write_reports(df: pd.DataFrame) -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(REPORTS_DIR, "leaderboard.csv"), index=False, encoding="utf-8-sig")
    df[SUMMARY_COLS].to_csv(
        os.path.join(REPORTS_DIR, "summary.csv"), index=False, encoding="utf-8-sig")
    # HTML 아티팩트(리더보드 대시보드)가 그대로 읽어들이는 경량 JSON — DATA 배열.
    chart_records = df[CHART_COLS].round(4).to_dict(orient="records")
    with io.open(os.path.join(REPORTS_DIR, "leaderboard_for_chart.json"), "w",
                 encoding="utf-8") as f:
        json.dump(chart_records, f, ensure_ascii=False, indent=2)


def print_report(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("CL-NIDS-Bench v2.5 리더보드")
    print("=" * 70)
    print("주의: 이 결과는 SSF/SPIDER/CADE/CND-IDS 각 논문의 원 보고 수치를")
    print("재현한 것이 아니라, 공통 인프라 위에서 구성요소를 재조합해 비교한")
    print("것이다(PRD 0절). 각 논문의 원 수치와 달라도 그 자체는 결함이 아니다.")
    print()

    overall_top = df.iloc[0]
    print(f"[전체 1위] {overall_top['combo_id']} ({overall_top['dataset']}) "
          f"F1={overall_top['f1']:.4f} PR-AUC={overall_top['pr_auc']:.4f}")

    for track in ["A", "B"]:
        track_df = df[df["track"] == track]
        if len(track_df) == 0:
            continue
        top = track_df.iloc[0]
        print(f"[Track {track} 1위] {top['combo_id']} ({top['dataset']}) "
              f"F1={top['f1']:.4f} PR-AUC={top['pr_auc']:.4f}")

    print()
    print("두 Track은 서로 다른 문제를 푼다 - Track A(semi-supervised")
    print("discriminative)와 Track B(label-free novelty detection)의 순위 비교는")
    print("'우열'이 아니라 '트레이드오프'로 해석해야 한다(PRD 5절/7절 이슈2).")
    print()

    print("상위 5개 조합:")
    for _, row in df.head(5).iterrows():
        extra = (f" best_f1_reference={row['best_f1_reference']:.4f}"
                 if row["track"] == "A" else "")
        print(f"  {row['combo_id']} ({row['dataset']}): F1={row['f1']:.4f} "
              f"PR-AUC={row['pr_auc']:.4f} BWT={row['bwt']:.4f}{extra}")


def main() -> None:
    results = load_all_results()
    if not results:
        print("results/ 에 결과 파일이 없습니다. 먼저 grid_runner.py를 실행하세요.")
        return
    df = build_leaderboard(results)
    write_reports(df)
    print_report(df)


if __name__ == "__main__":
    main()
