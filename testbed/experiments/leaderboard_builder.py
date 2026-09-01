"""리더보드 생성 — PRD Phase 4.

스모크 테스트를 통과한 조합의 results/*.json을 모아 F1 내림차순(동률 시
PR-AUC 내림차순)으로 정렬한 reports/leaderboard_{dataset}.csv를 **데이터셋별로
따로** 만든다. Track A/B/전체 1위를 명시하고, 상위 5개는 BWT(+Track A는
best_f1_reference)도 함께 표시한다.

**데이터셋을 섞지 않는 이유**: NSL-KDD/UNSW-NB15/CICIDS2018은 규모·난이도가
서로 크게 달라(예: 같은 조합이라도 F1이 데이터셋마다 0.6대~0.9대까지 벌어짐,
docs/metric_justification.md 참고) 세 데이터셋 결과를 하나의 표에 섞어 정렬하면
"어느 조합이 최선인가"가 아니라 "어느 조합이 우연히 가장 쉬운 데이터셋에
배정됐는가"를 보게 된다. 그래서 데이터셋마다 독립적으로 리더보드를 만들고,
비교는 반드시 같은 데이터셋 내에서만 한다.
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


def write_reports(df: pd.DataFrame, dataset_name: str) -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(REPORTS_DIR, f"leaderboard_{dataset_name}.csv"),
              index=False, encoding="utf-8-sig")
    df[SUMMARY_COLS].to_csv(
        os.path.join(REPORTS_DIR, f"summary_{dataset_name}.csv"),
        index=False, encoding="utf-8-sig")
    # HTML 아티팩트(리더보드 대시보드)가 그대로 읽어들이는 경량 JSON — DATA 배열.
    chart_records = df[CHART_COLS].round(4).to_dict(orient="records")
    with io.open(os.path.join(REPORTS_DIR, f"leaderboard_for_chart_{dataset_name}.json"), "w",
                 encoding="utf-8") as f:
        json.dump(chart_records, f, ensure_ascii=False, indent=2)


def print_report(df: pd.DataFrame, dataset_name: str) -> None:
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

    print("상위 5개 조합:")
    for _, row in df.head(5).iterrows():
        extra = (f" best_f1_reference={row['best_f1_reference']:.4f}"
                 if row["track"] == "A" else "")
        print(f"  {row['combo_id']}: F1={row['f1']:.4f} "
              f"PR-AUC={row['pr_auc']:.4f} BWT={row['bwt']:.4f}{extra}")
    print()


def main() -> None:
    results = load_all_results()
    if not results:
        print("results/ 에 결과 파일이 없습니다. 먼저 grid_runner.py를 실행하세요.")
        return
    dataset_names = sorted({r["dataset"] for r in results})
    for dataset_name in dataset_names:
        dataset_results = [r for r in results if r["dataset"] == dataset_name]
        # 2026-08-26 추가(전수 재검토 중 발견) — grid_runner.py의 run_grid()는
        # 중간에 끊겨도 이어서 돌릴 수 있게 설계되어 있다(결과 파일이 있으면
        # code_version이 지금 코드와 일치할 때만 건너뜀). 그 말은 재실행이
        # 아직 다 끝나지 않은 상태에서 이 스크립트를 돌리면 results/ 안에
        # "새 코드로 갓 계산된 결과"와 "아직 재계산 못 한 옛 코드 결과"가
        # 섞여 있을 수 있다는 뜻이다 — 리더보드가 이 둘을 구분 없이 같은
        # 표에 섞어 정렬하면 조용히 오해를 부를 수 있어, 코드 버전이 섞여
        # 있으면 경고만 낸다(자동으로 막지는 않는다 — 부분 결과라도 봐야
        # 할 때가 있으므로).
        code_versions = {r.get("code_version") for r in dataset_results}
        if len(code_versions) > 1:
            print(f"경고: [{dataset_name}] 결과에 서로 다른 code_version이 섞여 "
                  f"있습니다({sorted(v for v in code_versions if v)}) — grid_runner.py "
                  f"재실행이 아직 끝나지 않았을 수 있습니다. 전체 재실행 완료 후 "
                  f"다시 생성하는 것을 권장합니다.")
        df = build_leaderboard(dataset_results)
        write_reports(df, dataset_name)
        print_report(df, dataset_name)


if __name__ == "__main__":
    main()
