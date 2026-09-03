"""그리드 실행 — PRD Phase 3.

enumerate_valid_combos()가 직접 구성한 조합(현재 96개: Track A 90 + Track B 6 —
drift_detector가 실제로 소비되지 않는 (sample_selector, memory_manager) 조합은
'ssf'만 제외하고 'none'/'cade'는 남긴다, common/compatibility.py 참고) 중
스모크 테스트(Phase 2.5)를 통과한 조합만 NSL-KDD/UNSW-NB15/CICIDS2018 전체
데이터로 실행해 results/*.json(REQUIRED_RESULT_FIELDS 전체)을 생성한다.
완전 교차 조합을 그대로 도는 코드 경로는 두지 않는다(PRD 6절).
"""

import hashlib
import io
import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score

from testbed.base import FCLAutoEncoder, ssf_backbone_dims
from testbed.common.compatibility import enumerate_valid_combos
from testbed.common.metrics import (
    bwt,
    build_r_matrix,
    f1_score,
    per_category_counts,
    per_category_fpr,
    per_category_recall,
    pr_auc,
    precision_score,
    recall_score,
)
from testbed.common.result_schema import make_combo_id, validate_result
from testbed.data.dataset_loader import load_dataset
from testbed.pipeline import CLClient
from testbed.experiments.smoke_test import _load_configs, _REPO_ROOT, _TESTBED_ROOT

RESULTS_DIR = os.path.join(_TESTBED_ROOT, "results")
# 실패한 조합 기록을 results/*.json과 같은 디렉터리에 두면
# leaderboard_builder.py의 glob이 REQUIRED_RESULT_FIELDS 없는 깨진 행으로
# 주워갈 위험이 있다 — 별도 하위 디렉터리(glob은 재귀 탐색 안 함)로 분리.
FAILURES_DIR = os.path.join(RESULTS_DIR, "failures")


def _atomic_write_json(path: str, obj: Any) -> None:
    """json.dump()을 임시 파일에 쓴 뒤 os.replace()로 원자적으로 교체한다.

    GPU 서버 장시간 무인 실행 중 프로세스가 쓰기 도중 죽으면(OOM kill 등)
    파일이 반쯤 쓰인 채 남을 수 있고, 다음 실행이 resume 로직으로 이어서
    돌 때 `json.load()`가 예외를 던져 전체 그리드가 죽는다. `os.replace()`
    (POSIX/Windows 양쪽에서 원자적)는 임시 파일이 완전히 쓰인 뒤에만
    교체하므로 반쯤 쓰인 상태가 될 수 없다."""
    tmp_path = f"{path}.tmp{os.getpid()}"
    with io.open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _load_json_safely(path: str) -> Optional[Dict[str, Any]]:
    """캐시된 결과 파일을 읽는다. 손상된(잘려나간) JSON이면 예외 대신
    None을 반환해 "캐시 없음"으로 취급한다 — 장시간 무인 GPU 실행에서
    파일 하나 때문에 전체 그리드가 멈추는 것보다, 경고를 남기고 그 조합을
    다시 계산하는 편이 안전하다."""
    try:
        with io.open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"경고: {path} 을(를) 읽을 수 없어(손상된 파일로 추정: {exc}) "
              f"캐시 없음으로 간주하고 재계산합니다.")
        return None

# 결과가 어떤 코드 상태에서 계산됐는지 해시로 남겨, 캐시를 재사용하기 전에
# 지금 코드와 일치하는지 확인한다(compute_code_version 참고). configs/
# (하이퍼파라미터 YAML)와 common/(F1/BWT 공식, result_schema.py)도 포함 —
# 이 파일들이 안 바뀌면 code_version이 그대로라 하이퍼파라미터만 바뀐
# 재실행에서 낡은 결과를 조용히 재사용할 위험이 있다.
# experiments/grid_runner.py 자기 자신도 포함(결과-구성 로직 변경 감지용).
_VERSIONED_PATHS = [
    os.path.join(_TESTBED_ROOT, "components"),
    os.path.join(_TESTBED_ROOT, "base"),
    os.path.join(_TESTBED_ROOT, "pipeline"),
    os.path.join(_TESTBED_ROOT, "common"),
    os.path.join(_TESTBED_ROOT, "configs"),
    os.path.join(_TESTBED_ROOT, "data", "dataset_loader.py"),
    os.path.join(_TESTBED_ROOT, "experiments", "grid_runner.py"),
]


def compute_code_version() -> str:
    """_VERSIONED_PATHS 아래 모든 .py/.yaml 파일 내용을 해시해 12자리로
    줄인다. 이 해시가 바뀌면 결과에 영향을 줄 수 있는 코드나 하이퍼파라미터가
    바뀐 것으로 간주한다."""
    h = hashlib.sha256()
    files = []
    for p in _VERSIONED_PATHS:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                for name in sorted(names):
                    if name.endswith(".py") or name.endswith(".yaml"):
                        files.append(os.path.join(root, name))
        elif os.path.isfile(p):
            files.append(p)
    for path in sorted(files):
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


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
                    device: str = "cpu",
                    code_version: Optional[str] = None) -> Dict[str, Any]:
    input_dim = dataset["input_dim"]
    hp = dict(global_hparams)
    hp["_input_dim"] = input_dim
    # hidden_dim/latent_dim을 SSF 원 논문 공식으로 데이터셋별 계산
    # (configs/global_hparams.yaml 참고).
    hp["hidden_dim"], hp["latent_dim"] = ssf_backbone_dims(input_dim)
    # Track B(CND-IDS)는 원 논문 에폭(20)을 쓴다 — Track A와 같은 200을
    # 적용하면 CND-IDS의 약한 망각방지 가중치(lambda_cl=0.1)가 그 강도를
    # 못 버텨 catastrophic forgetting이 발생함을 실측으로 확인했다.
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
    # 공격 category별 recall 리포팅용. dataset_loader.py가 test_category를
    # 보존한다(없으면 None → category 분석 생략). 학습 경로(CLClient)에는
    # 넘기지 않는다.
    all_test_categories = [e.get("test_category") for e in experiences]
    has_category = all(c is not None for c in all_test_categories)
    pooled_test_category = (
        np.concatenate([np.asarray(c) for c in all_test_categories]) if has_category else None)

    f1_rows: List[List[float]] = []
    # 부가 분석 필드(학습 경로 무변경). CLClient.run_experience()는 이미
    # drift_detected/drift_score를 라운드마다 반환한다 — 여기서 기록만
    # 한다. 해석 주의: 라운드 0은 비교할 버퍼가 없어 구조적으로 False,
    # memory_manager=none이면 buf_ref가 항상 None이라 매 라운드 False
    # (base/drift_detector.py 계약), dd=none은 항상 False. 의미 있는 값은
    # Track A의 dd∈{ssf,cade} × mm∈{spider,ssf} 조합에서만 나온다
    # (common/compatibility.py 참고).
    drift_detected_per_round: List[bool] = []
    drift_score_per_round: List[float] = []
    # category별 recall의 라운드 이력 — 라운드 t의 모델·threshold로 pooled
    # test(0..T-1 전부)를 채점한 뒤 category별로 나눈 것. 이미 계산된
    # out["eval_scores"]를 재사용하므로 추가 forward도, RNG 소비도 없다.
    per_category_recall_history: Dict[str, List[float]] = {}
    normal_fpr_history: List[float] = []
    total_selected = 0
    total_available = 0
    training_time_sec = 0.0
    last_out = None
    # memory_footprint를 마지막 라운드 스냅샷 하나만 보면, SPIDER처럼 매
    # 라운드 버퍼를 그 라운드 selected_data 크기로 통째로 교체하는
    # memory_manager는 class-incremental 분할의 라운드별 데이터량 변동
    # 때문에 오해를 살 수 있다 — 라운드별 크기를 전부 기록해 peak/avg도 남긴다.
    memory_footprint_history: List[int] = []

    for exp_idx, e in enumerate(experiences):
        t0 = time.time()
        out = client.run_experience(
            exp_idx, e["train_X"], e["train_y"], all_test_splits, labeling_budget,
            train_category=e.get("train_category"))
        training_time_sec += time.time() - t0

        threshold = out["threshold"]
        row = [
            f1_score(labels, (scores > threshold).long())
            for scores, labels in zip(out["eval_scores"], out["eval_labels"])
        ]
        f1_rows.append(row)
        drift_detected_per_round.append(bool(out["drift_detected"]))
        drift_score_per_round.append(float(out["drift_score"]))
        if has_category:
            round_preds = (torch.cat(out["eval_scores"]) > threshold).long().numpy()
            round_labels = torch.cat(out["eval_labels"]).numpy()
            counts = per_category_counts(round_labels, round_preds, pooled_test_category)
            for cat, rec in per_category_recall(counts).items():
                per_category_recall_history.setdefault(cat, []).append(rec)
            # 정상 행은 y==0 기준으로 category와 무관하게 하나로 합쳐 FPR을 본다
            # (정상 category 표기가 데이터셋마다 다르고 하나뿐이라 합쳐도 같다).
            n_normal = int((round_labels == 0).sum())
            n_fp = int(((round_labels == 0) & (round_preds == 1)).sum())
            normal_fpr_history.append(float(n_fp / n_normal) if n_normal > 0 else 0.0)
        total_selected += out["n_selected"]
        total_available += len(e["train_X"])
        last_out = out
        memory_footprint_history.append(client.memory_manager.size())

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

    # Track B는 이 참고 지표를 계산하지 않는다 — "계산 안 함"을 NaN으로
    # 명시(0.0이면 "도달 가능한 최선의 F1이 0"으로 오독될 위험이 있음).
    best_f1_reference = (
        _best_f1_achievable(pooled_scores, pooled_labels)
        if combo["track"] == "A" else float("nan")
    )

    # Inference latency: 마지막 라운드 모델로 pooled test 데이터를 다시
    # 인코딩+스코어링하는 데 걸리는 시간(학습/드리프트 감지와 분리된 순수
    # 추론 지연). CICIDS2018처럼 pooled 크기가 수백만 행이면 통째로
    # forward하다 CUDA OOM이 나므로 forward_batched()로 나눠 돌린다.
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        pooled_test_x = torch.cat([tx for tx, _ in all_test_splits]).to(client.device)
        # CADEMADScorer가 사설 인코더에 연결된 경우 공유 backbone을 거치지
        # 않고 원본을 그대로 넘긴다(cl_client.py Step 6/7과 동일한 이유).
        if client.anomaly_scorer.uses_shared_representation:
            z_all, _, _ = client.forward_batched(pooled_test_x)
            client.anomaly_scorer.score(z_all)
        else:
            client.anomaly_scorer.score(pooled_test_x)
    inference_time = time.time() - t0
    n_inference_samples = sum(len(tx) for tx, _ in all_test_splits)
    avg_inference_latency_ms = (inference_time / max(n_inference_samples, 1)) * 1000.0

    # Track B(CND-IDS)는 compute_loss()가 라벨을 전혀 쓰지 않는 라벨-프리
    # 설계다(cndids_anti_forgetting.py) — experience 전체를 학습에 쓰더라도
    # 실제 소비 라벨 수는 항상 0.
    labeling_cost = 0.0 if combo["track"] == "B" else total_selected / max(total_available, 1)
    memory_footprint = client.memory_manager.size()
    memory_footprint_peak = max(memory_footprint_history) if memory_footprint_history else 0
    memory_footprint_avg = (
        sum(memory_footprint_history) / len(memory_footprint_history)
        if memory_footprint_history else 0.0
    )

    # category별 최종 요약. first_seen_round는 그 category가 test에 처음
    # 등장하는 experience(class-incremental 분할이라 공격 category당 정확히
    # 하나의 experience에만 배정 — `_class_incremental_split`). forgetting =
    # 처음 등장한 라운드의 recall - 마지막 라운드의 recall.
    per_category_final: Dict[str, Dict[str, Any]] = {}
    if has_category:
        n_rounds = len(experiences)
        for cat, history in per_category_recall_history.items():
            first_seen = next(
                (j for j, c in enumerate(all_test_categories)
                 if bool(np.any(np.asarray(c) == cat))), 0)
            n_test = int(np.sum(pooled_test_category == cat))
            per_category_final[cat] = {
                "n_test": n_test,
                "first_seen_round": int(first_seen),
                "recall_at_first_seen": float(history[first_seen]),
                "recall_final": float(history[n_rounds - 1]),
                "forgetting": float(history[first_seen] - history[n_rounds - 1]),
            }
    normal_fpr = float(normal_fpr_history[-1]) if normal_fpr_history else float("nan")

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
        "memory_footprint_peak": memory_footprint_peak,
        "memory_footprint_avg": memory_footprint_avg,
        "avg_inference_latency_ms": avg_inference_latency_ms,
        "best_f1_reference": best_f1_reference,
        "seed": seed,
        "code_version": code_version if code_version is not None else compute_code_version(),
        "drift_detected_per_round": drift_detected_per_round,
        "drift_score_per_round": drift_score_per_round,
        "n_drift_detected": int(sum(drift_detected_per_round)),
        "per_category_final": per_category_final,
        "per_category_recall_history": per_category_recall_history,
        "normal_fpr": normal_fpr,
        "normal_fpr_history": normal_fpr_history,
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
             device: str = "cpu",
             shard: Optional[tuple] = None,
             track: Optional[str] = None) -> List[Dict[str, Any]]:
    global_hparams, component_hparams = _load_configs()
    code_version = compute_code_version()
    print(f"code_version={code_version} (components/base/pipeline/dataset_loader 해시 — "
          f"이 값이 캐시된 결과 파일과 다르면 재계산합니다)")
    if smoke_results_path is None:
        smoke_results_path = os.path.join(_TESTBED_ROOT, "experiments", "smoke_test_results.json")
    passed_by_dataset = load_smoke_passed_combo_ids(smoke_results_path)

    all_combos = enumerate_valid_combos()
    total_combos = len(all_combos)

    labeling_budget = global_hparams["labeling_budget"]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    failed_combos: List[Dict[str, Any]] = []
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

        if track is not None:
            # 특정 Track만 재계산이 필요할 때(다른 Track의 결과 파일 존재
            # 여부와 무관하게) 명시적으로 필터링.
            combos = [c for c in combos if c["track"] == track]
            print(f"[{dataset_name}] Track {track}만 실행 대상 {len(combos)}개")

        if shard is not None:
            shard_idx, n_shards = shard
            # all_combos 순서가 보존되므로, 동일한 --datasets/smoke_results_path
            # 로 여러 프로세스를 띄우면 인덱스가 항상 동일 — 별도 동기화 없이
            # 인덱스 기반 분할만으로 겹치지 않게 나뉜다.
            combos = [c for i, c in enumerate(combos) if i % n_shards == shard_idx]
            print(f"[{dataset_name}] shard {shard_idx}/{n_shards} 담당 조합 {len(combos)}개")

        # 원본 train/test 파일 분리 유지, 각 파일 내부는 고정 seed로 섞은 뒤
        # 분할(dataset_loader.py의 preserve_official_split 기본값과 동일).
        # n_experiences는 명시하지 않는다 — dataset_loader.py가 데이터셋 자신의
        # 실제 공격 유형 수로 자동 결정한다(2026-09-03, configs/global_hparams.yaml
        # 참고. 데이터셋마다 라운드 수가 달라도 되는 이유는 리더보드가 데이터셋별로
        # 완전히 분리돼 있어서다).
        dataset = load_dataset(dataset_name, base_dir=_REPO_ROOT,
                                seed=global_hparams["seed"])
        for combo in combos:
            combo_id = make_combo_id(combo)
            out_path = os.path.join(RESULTS_DIR, f"{combo_id}__{dataset_name}.json")
            if os.path.exists(out_path):
                # 이전 실행이 중간에 끊긴 뒤 이어서 돌릴 때(CICIDS2018처럼
                # 조합당 25~50분+ 걸리는 경우 재계산 낭비 방지) — 결과
                # 파일이 있으면 다시 계산하지 않고 건너뛴다. 단, 그 파일이
                # 지금 코드로 계산된 게 맞는지 code_version으로 확인한 뒤에만
                # 건너뛴다(컴포넌트를 고친 뒤에도 낡은 결과가 재사용되는
                # 사고를 막기 위함). code_version 필드가 없으면(도입 이전
                # 파일) 무조건 재계산한다. `_load_json_safely()`가 손상된
                # 캐시 파일이면 None을 반환하므로 그 경우도 재계산으로 넘어간다.
                cached = _load_json_safely(out_path)
                if cached is not None and cached.get("code_version") == code_version:
                    print(f"[{dataset_name}] {combo_id} 이미 결과 있음(코드 버전 일치), 건너뜀")
                    continue
                if cached is not None:
                    print(f"[{dataset_name}] {combo_id} 결과는 있지만 코드 버전이 달라 재계산합니다 "
                          f"(cached={cached.get('code_version')!r}, current={code_version!r})")

            t0 = time.time()
            try:
                result = run_combo_full(
                    combo, dataset_name, dataset, global_hparams, component_hparams,
                    labeling_budget, device, code_version=code_version)
            except Exception as exc:
                # 조합 하나가 예외로 죽어도(수치 불안정, GPU OOM 등) 전체
                # 그리드가 죽지 않고 이 조합만 "실패"로 기록한 뒤 계속
                # 진행한다. 상세 기록은 results/failures/ 아래 별도 파일에
                # 남긴다(FAILURES_DIR 정의 참고).
                elapsed = time.time() - t0
                tb = traceback.format_exc()
                print(f"[{dataset_name}] {combo_id} 실패({elapsed:.1f}s 경과 후) — "
                      f"{type(exc).__name__}: {exc}")
                print(tb)
                os.makedirs(FAILURES_DIR, exist_ok=True)
                failure_record = {
                    "combo_id": combo_id,
                    "exp_name": f"{combo_id}__{dataset_name}",
                    "dataset": dataset_name,
                    "track": combo["track"],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": tb,
                    "elapsed_sec": elapsed,
                    "code_version": code_version,
                }
                failure_path = os.path.join(FAILURES_DIR, f"{combo_id}__{dataset_name}.json")
                _atomic_write_json(failure_path, failure_record)
                failed_combos.append(failure_record)
                continue

            elapsed = time.time() - t0
            print(f"[{dataset_name}] {result['combo_id']} f1={result['f1']:.3f} "
                  f"pr_auc={result['pr_auc']:.3f} bwt={result['bwt']:.3f} ({elapsed:.1f}s)")

            _atomic_write_json(out_path, result)
            # 이전 시도에서 실패해 failures/에 남아있던 기록이 있다면 지운다
            # (안 지우면 성공한 조합인데도 실패 목록에 계속 남는다).
            stale_failure_path = os.path.join(FAILURES_DIR, f"{combo_id}__{dataset_name}.json")
            if os.path.exists(stale_failure_path):
                os.remove(stale_failure_path)
            all_results.append(result)

    if failed_combos:
        print(f"\n경고: {len(failed_combos)}개 조합이 실패했습니다(상세 내역은 "
              f"{FAILURES_DIR} 참고). 성공한 {len(all_results)}개 조합의 결과는 "
              f"그대로 저장되어 있습니다:")
        for f in failed_combos:
            print(f"  - [{f['dataset']}] {f['combo_id']}: "
                  f"{f['error_type']}: {f['error_message']}")

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
    parser.add_argument(
        "--shard", default=None,
        help="'i/n' 형식으로 조합을 n등분해 i번째 몫만 실행한다(0-indexed). "
             "여러 GPU/터미널에서 동시에 실행할 때 조합이 겹치지 않게 나누는 "
             "용도. 예: 3개 프로세스를 동시에 돌리려면 각각 "
             "--shard 0/3, --shard 1/3, --shard 2/3 로 실행. 모든 프로세스가 "
             "동일한 --datasets/--smoke-results 를 써야 동일하게 나뉜다.")
    parser.add_argument(
        "--track", default=None, choices=["A", "B"],
        help="'A' 또는 'B'를 주면 그 Track에 속한 조합만 실행한다. 특정 "
             "Track만 코드가 바뀌어 재계산이 필요할 때(예: Track B), 다른 "
             "Track의 결과 파일이 우연히 없어도 실행되지 않도록 명시적으로 "
             "막아준다. 생략하면 두 Track 다 대상이 된다(기존 동작).")
    args = parser.parse_args()
    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    shard_arg = None
    if args.shard is not None:
        shard_idx_str, n_shards_str = args.shard.split("/")
        shard_idx, n_shards = int(shard_idx_str), int(n_shards_str)
        assert 0 <= shard_idx < n_shards, "--shard 는 0 <= i < n 이어야 함"
        shard_arg = (shard_idx, n_shards)
    run_grid(datasets=dataset_list, device=args.device, shard=shard_arg, track=args.track)
