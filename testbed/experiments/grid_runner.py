"""그리드 실행 — PRD Phase 3.

enumerate_valid_combos()가 직접 구성한 조합(현재 93개: Track A 90 + Track B 3 —
drift_detector가 실제로 소비되지 않는 (sample_selector, memory_manager) 조합은
'ssf'만 제외하고 'none'/'cade'는 남긴다, common/compatibility.py 참고) 중
스모크 테스트(Phase 2.5)를 통과한 조합만 NSL-KDD/UNSW-NB15/CICIDS2018 전체
데이터로 실행해 results/*.json(REQUIRED_RESULT_FIELDS 전체)을 생성한다.
완전 교차 조합을 그대로 도는 코드 경로는 두지 않는다(PRD 6절).
"""

import io
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score

from testbed.base import FCLAutoEncoder
from testbed.common.compatibility import enumerate_valid_combos
from testbed.common.metrics import (
    bwt,
    build_r_matrix,
    f1_score,
    pr_auc,
    precision_score,
    recall_score,
)
from testbed.common.result_schema import make_combo_id, validate_result
from testbed.data.dataset_loader import extract_normal_reference, load_dataset
from testbed.pipeline import CLClient
from testbed.experiments.smoke_test import _load_configs, _REPO_ROOT, _TESTBED_ROOT

RESULTS_DIR = os.path.join(_TESTBED_ROOT, "results")


def _best_f1_achievable(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """CND-IDS Best-F 방식(precision_recall_curve)으로 도달 가능한 최댓값 F1.

    Track A(cade_mad)의 `best_f1_reference`(14.1절) 계산에만 쓰는 참고용
    지표다 — leaderboard 정렬이나 조합 제외 기준으로 쓰지 않는다.
    """
    y = labels.numpy().astype(int)
    s = scores.numpy().astype(float)
    if len(np.unique(y)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(y, s)
    denom = precision + recall
    f1 = np.zeros_like(precision)
    nz = denom > 0
    f1[nz] = 2 * precision[nz] * recall[nz] / denom[nz]
    return float(f1.max())


def run_combo_full(combo: Dict[str, Any], dataset_name: str, dataset: Dict[str, Any],
                    global_hparams: Dict[str, Any],
                    component_hparams: Dict[str, Dict[str, Any]],
                    labeling_budget: Dict[str, Any],
                    device: str = "cpu") -> Dict[str, Any]:
    input_dim = dataset["input_dim"]
    hp = dict(global_hparams)
    hp["_input_dim"] = input_dim
    # Track B(CND-IDS)는 원 논문 에폭(20)을 쓴다 — Track A와 같은 200을
    # 그대로 적용하면 CND-IDS의 약한 망각방지 가중치(lambda_cl=0.1)가 그
    # 학습 강도를 못 버텨 catastrophic forgetting이 발생함을 실측으로
    # 확인했다(configs/global_hparams.yaml 주석 참고). Track B 3개 조합은
    # 전부 anti_forgetting=cndids로 고정이라 이 오버라이드로도 트랙 내부
    # 비교의 공정성은 그대로 유지된다.
    if combo["track"] == "B":
        hp["epochs_per_experience"] = global_hparams["epochs_per_experience_track_b"]
        hp["batch_size"] = global_hparams["batch_size_track_b"]
    seed = hp.get("seed", 42)

    torch.manual_seed(seed)
    model = FCLAutoEncoder(input_dim=input_dim, hidden_dim=hp["hidden_dim"],
                            latent_dim=hp["latent_dim"])
    client = CLClient(model, combo, hp, component_hparams, device=device)

    experiences = dataset["experiences"]
    all_test_splits = [(e["test_X"], e["test_y"]) for e in experiences]
    ref_size = min(hp.get("normal_reference_size", 500), len(experiences[0]["train_X"]))
    normal_ref = extract_normal_reference(experiences, size=ref_size, seed=seed)
    client.set_normal_reference(normal_ref)

    f1_rows: List[List[float]] = []
    total_selected = 0
    total_available = 0
    training_time_sec = 0.0
    last_out = None

    for exp_idx, e in enumerate(experiences):
        t0 = time.time()
        out = client.run_experience(
            exp_idx, e["train_X"], e["train_y"], all_test_splits, labeling_budget)
        training_time_sec += time.time() - t0

        threshold = out["threshold"]
        row = [
            f1_score(labels, (scores > threshold).long())
            for scores, labels in zip(out["eval_scores"], out["eval_labels"])
        ]
        f1_rows.append(row)
        total_selected += out["n_selected"]
        total_available += len(e["train_X"])
        last_out = out

    R = build_r_matrix(f1_rows)
    bwt_value = bwt(R)

    # 최종 지표: 마지막 experience까지 학습한 모델을 0..T-1 test split 전체에
    # 대해 pooled 평가한 것 (PRD 14.1절 f1/precision/recall/pr_auc).
    pooled_scores = torch.cat(last_out["eval_scores"])
    pooled_labels = torch.cat(last_out["eval_labels"])
    final_threshold = last_out["threshold"]
    pooled_preds = (pooled_scores > final_threshold).long()

    final_f1 = f1_score(pooled_labels, pooled_preds)
    final_precision = precision_score(pooled_labels, pooled_preds)
    final_recall = recall_score(pooled_labels, pooled_preds)
    final_pr_auc = pr_auc(pooled_labels, pooled_scores)
    try:
        final_roc_auc = float(roc_auc_score(pooled_labels.numpy(), pooled_scores.numpy()))
    except ValueError:
        final_roc_auc = 0.0

    best_f1_reference = (
        _best_f1_achievable(pooled_scores, pooled_labels) if combo["track"] == "A" else 0.0
    )

    # Inference latency: 마지막 라운드 모델로 pooled test 데이터를 다시
    # 인코딩+스코어링하는 데 걸리는 시간을 별도로 측정한다(학습/드리프트 감지
    # 등과 분리된 순수 추론 지연). CICIDS2018처럼 pooled 크기가 수백만 행이면
    # 통째로 한 번에 forward하다 GPU 메모리가 터지므로(실측: CUDA OOM),
    # client.forward_batched()로 배치 단위로 나눠 돌린다(pipeline/cl_client.py
    # 참고 — Step 2/3/7과 같은 이유).
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        pooled_test_x = torch.cat([tx for tx, _ in all_test_splits]).to(client.device)
        z_all, _, _ = client.forward_batched(pooled_test_x)
        client.anomaly_scorer.score(z_all)
    inference_time = time.time() - t0
    n_inference_samples = sum(len(tx) for tx, _ in all_test_splits)
    avg_inference_latency_ms = (inference_time / max(n_inference_samples, 1)) * 1000.0

    labeling_cost = total_selected / max(total_available, 1)
    memory_footprint = client.memory_manager.size()

    result = {
        "combo_id": make_combo_id(combo),
        "exp_name": f"{make_combo_id(combo)}__{dataset_name}",
        "dataset": dataset_name,
        "drift_detector": combo["drift_detector"],
        "sample_selector": combo["sample_selector"],
        "memory_manager": combo["memory_manager"],
        "anti_forgetting": combo["anti_forgetting"],
        "anomaly_scorer": combo["anomaly_scorer"],
        "track": combo["track"],
        "f1": final_f1,
        "precision": final_precision,
        "recall": final_recall,
        "pr_auc": final_pr_auc,
        "bwt": bwt_value,
        "perf_matrix": R.tolist(),
        "roc_auc": final_roc_auc,
        "labeling_cost": labeling_cost,
        "training_time_sec": training_time_sec,
        "memory_footprint": memory_footprint,
        "avg_inference_latency_ms": avg_inference_latency_ms,
        "best_f1_reference": best_f1_reference,
        "seed": seed,
    }
    validate_result(result)
    return result


def load_smoke_passed_combo_ids(smoke_results_path: str) -> Optional[Dict[str, set]]:
    """데이터셋별 스모크 테스트 통과 combo_id 집합을 반환한다.

    combo_id만으로 묶으면(데이터셋 구분 없이) 한 데이터셋에서 통과한 조합이
    다른 데이터셋에서 실제로 실패해도(예: CICIDS2018에서만 score 분포가
    퇴화하는 경우) 그 데이터셋 그리드에 그대로 포함되는 문제가 있었다 —
    데이터셋별로 분리해 반환한다.
    """
    if not os.path.exists(smoke_results_path):
        return None
    with io.open(smoke_results_path, encoding="utf-8") as f:
        smoke_results = json.load(f)
    passed_by_dataset: Dict[str, set] = {}
    for r in smoke_results:
        if r["passed"]:
            passed_by_dataset.setdefault(r["dataset"], set()).add(r["combo_id"])
    return passed_by_dataset


def run_grid(datasets: List[str] = ("nsl-kdd", "unsw-nb15"),
             smoke_results_path: Optional[str] = None,
             device: str = "cpu") -> List[Dict[str, Any]]:
    global_hparams, component_hparams = _load_configs()
    if smoke_results_path is None:
        smoke_results_path = os.path.join(_TESTBED_ROOT, "experiments", "smoke_test_results.json")
    passed_by_dataset = load_smoke_passed_combo_ids(smoke_results_path)

    all_combos = enumerate_valid_combos()
    total_combos = len(all_combos)

    labeling_budget = global_hparams["labeling_budget"]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    for dataset_name in datasets:
        if passed_by_dataset is not None:
            passed_ids = passed_by_dataset.get(dataset_name, set())
            combos = [c for c in all_combos if make_combo_id(c) in passed_ids]
            print(f"[{dataset_name}] 스모크 테스트를 통과한 "
                  f"{len(combos)}/{total_combos}개 조합만 실행합니다.")
        else:
            combos = all_combos
            print(f"경고: 스모크 테스트 결과 파일이 없어 [{dataset_name}] "
                  f"{total_combos}개 조합 전체를 실행합니다.")

        # 두 데이터셋 모두 동일한 프로토콜(원본 train/test 파일 분리 유지,
        # 각 파일 내부는 고정 seed로 섞은 뒤 분할)을 쓴다 — dataset_loader.py의
        # preserve_official_split 기본값(True)과 동일. UNSW-NB15는 원본 파일이
        # 라벨 기준으로 정렬되어 있었지만(실측 확인), 파일 내부를 섞으면서
        # 해결됐다(사용자 지시로 NSL-KDD와 동일 프로토콜로 통일).
        dataset = load_dataset(dataset_name, base_dir=_REPO_ROOT,
                                n_experiences=global_hparams["n_experiences"],
                                seed=global_hparams["seed"])
        for combo in combos:
            t0 = time.time()
            result = run_combo_full(
                combo, dataset_name, dataset, global_hparams, component_hparams,
                labeling_budget, device)
            elapsed = time.time() - t0
            print(f"[{dataset_name}] {result['combo_id']} f1={result['f1']:.3f} "
                  f"pr_auc={result['pr_auc']:.3f} bwt={result['bwt']:.3f} ({elapsed:.1f}s)")

            out_path = os.path.join(RESULTS_DIR, f"{result['combo_id']}__{dataset_name}.json")
            with io.open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            all_results.append(result)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CL-NIDS-Bench 그리드 실행")
    parser.add_argument(
        "--device", default="cpu",
        help="torch device 문자열 (예: 'cpu', 'cuda', 'cuda:0'). GPU가 있는 "
             "환경으로 옮긴 뒤 '--device cuda'로 실행하면 된다.")
    parser.add_argument(
        "--datasets", default="nsl-kdd,unsw-nb15",
        help="쉼표로 구분한 데이터셋 이름 목록 (nsl-kdd, unsw-nb15, cicids2018 중). "
             "예: --datasets nsl-kdd,unsw-nb15,cicids2018")
    args = parser.parse_args()
    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    run_grid(datasets=dataset_list, device=args.device)
