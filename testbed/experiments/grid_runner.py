"""그리드 실행 — PRD Phase 3.

enumerate_valid_combos()가 직접 구성한 조합(현재 78개: Track A 72 + Track B 6,
2026-09-14 재검토로 90→72 — drift_detector가 실제로 소비되지 않는
(sample_selector, memory_manager, anomaly_scorer) 조합은 'ssf'와, as=none일
때의 'cade'까지 제외하고 'none'만 남긴다, common/compatibility.py 참고) 중
스모크 테스트(Phase 2.5)를 통과한 조합만 NSL-KDD/UNSW-NB15 전체
데이터로 실행해 results/*.json(REQUIRED_RESULT_FIELDS 전체)을 생성한다.
완전 교차 조합을 그대로 도는 코드 경로는 두지 않는다(PRD 6절).
"""

import hashlib
import io
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

# 2026-09-14 추가 — Windows 콘솔(cp949 등 비-UTF-8 코드페이지)에서 print()가
# em-dash(—, U+2014) 같은 문자를 만나면 UnicodeEncodeError로 프로세스 전체가
# 죽는다(experiments/smoke_test.py 모듈 docstring 참고 — 이 크래시가 실제로
# X-IIoTID 스모크에서 재현돼 "특정 콤보가 원인 불명으로 죽는다"로 오인했던
# 사례가 있다). 정식 그리드는 GPU 서버에서 수 시간 걸리므로 이 사소한 이유로
# 도중에 죽는 걸 원천 차단한다.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, roc_auc_score

from testbed.base import FCLAutoEncoder, ssf_backbone_dims
from testbed.common.compatibility import enumerate_valid_combos
from testbed.common.metrics import (
    bwt,
    build_r_matrix,
    f1_score,
    per_category_counts,
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
# experiments/smoke_test.py도 포함(2026-09-14 추가) — smoke_test.py는 이
# 함수(compute_code_version)를 그대로 import해서 자신의 resume 판단에 쓰는데
# (smoke_test.py의 15.1~15.5 게이트 임계값, SMOKE_BEHAVIORAL_GATES_AS_WARNINGS
# 정책 등을 이 파일 자신이 정의) 정작 이 목록에 자기 자신이 없어서, 게이트
# 로직만 고치고 실제 학습 코드는 안 바뀐 경우 code_version이 그대로라
# resume=True가 낡은(옛 게이트 기준으로 판정된) 스모크 결과를 조용히 재사용하는
# 구멍이 있었다 — "코드가 바뀌면 캐시를 못 믿는다"는 이 파일의 원칙 자체가
# smoke_test.py 변경에 대해서는 지켜지지 않고 있었다.
_VERSIONED_PATHS = [
    os.path.join(_TESTBED_ROOT, "components"),
    os.path.join(_TESTBED_ROOT, "base"),
    os.path.join(_TESTBED_ROOT, "pipeline"),
    os.path.join(_TESTBED_ROOT, "common"),
    os.path.join(_TESTBED_ROOT, "configs"),
    os.path.join(_TESTBED_ROOT, "data", "dataset_loader.py"),
    os.path.join(_TESTBED_ROOT, "experiments", "grid_runner.py"),
    os.path.join(_TESTBED_ROOT, "experiments", "smoke_test.py"),
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
    client = CLClient(model, combo, hp, component_hparams, device=device,
                       held_out_normal_reference=dataset.get("held_out_normal_reference"))

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
        # pooled 평가 세트가 단일 클래스뿐이라 정의되지 않는 경우(2026-09-14
        # 재검토로 일관성 수정) — best_f1_reference와 같은 원칙: 0.0으로
        # 채우면 "무작위보다 나쁨"으로 오독될 위험이 있어 "계산 불가"를
        # NaN으로 명시한다. 매우 드문 경계 케이스(pooled=0..T-1 test 전체가
        # 한 클래스뿐이어야 함)라 지금까지의 실제 결과에 영향은 없었다.
        final_roc_auc = float("nan")

    # Track B는 이 참고 지표를 계산하지 않는다 — "계산 안 함"을 NaN으로
    # 명시(0.0이면 "도달 가능한 최선의 F1이 0"으로 오독될 위험이 있음).
    best_f1_reference = (
        _best_f1_achievable(pooled_scores, pooled_labels)
        if combo["track"] == "A" else float("nan")
    )

    # Inference latency: 마지막 라운드 모델로 pooled test 데이터를 다시
    # 인코딩+스코어링하는 데 걸리는 시간(학습/드리프트 감지와 분리된 순수
    # 추론 지연). pooled 크기가 크면 통째로 forward하다 CUDA OOM이 나므로
    # forward_batched()로 나눠 돌린다.
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
    # 하나의 experience에만 배정 — `_class_incremental_split`). bwt = 마지막
    # 라운드의 recall - 처음 등장한 라운드의 recall(2026-09-15부로
    # common/metrics.py의 bwt()와 부호를 통일 — 음수가 망각, 양수가 개선.
    # 예전엔 이 필드 이름이 "forgetting"이고 부호가 반대(첫 라운드-마지막
    # 라운드, 양수가 망각)였다 — design_decisions.md 7절이 "BWT 부호 규약이
    # 파일마다 다르다"고 지적했던 바로 그 불일치를 없앤 것. 프로젝트 전체에서
    # "음수=망각"이라는 규칙이 예외 없이 성립하도록 통일했다).
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
                "bwt": float(history[n_rounds - 1] - history[first_seen]),
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


def run_joint_baseline(dataset_name: str, dataset: Dict[str, Any],
                        global_hparams: Dict[str, Any], device: str = "cpu",
                        code_version: Optional[str] = None) -> Dict[str, Any]:
    """Joint/Offline 상한선 baseline (2026-09-13 도입).

    9.2절 근거: "이 CL 기법들이 얼마나 망각하는가"를 주장하려면 라운드
    분할·라벨 예산 둘 다 없는 이상적 상한선이 있어야 한다 — 이게 없으면
    forgetting 수치가 나빠도 "원래 이 정도가 한계인지 메커니즘이 부족한
    건지" 구분할 근거가 없다. `grid_runner.py`에 이런 실행 모드가 전혀
    없었음을 확인(2026-09-12, 9.2절) — 신규 구현.

    **설계**: 5-슬롯 컴포넌트 구조(drift_detector/sample_selector/
    memory_manager/anti_forgetting/anomaly_scorer)를 전혀 쓰지 않는다 —
    이 슬롯들은 전부 "스트리밍/라벨 희소성 문제를 어떻게 다룰까"에 대한
    답인데, Joint는 그 문제 자체를 없앤 것(전체 데이터를 한 번에, 전체
    라벨로 학습)이라 5-슬롯 어디에도 속하지 않는다. 그래서 `CLClient`의
    8단계 루프를 전혀 쓰지 않고, 공유 backbone(`FCLAutoEncoder`)만 갖고
    별도로 구현한다.

    - 모든 experience의 train_X/train_y를 라운드 구분 없이 통째로 합쳐
      `epochs_per_experience`(Track A 관례값, 200)만큼 plain BCE로 학습한다
      — Track A의 72개 콤보와 같은 backbone·epoch 관례를 공유해야
      "상한선"이라는 비교가 의미 있다(Track B의 라벨-프리·다른 epoch
      관례는 이 baseline이 다루는 범위 밖 — 9.1절에서 이미 Track A/B
      학습 조건 자체가 다름을 인정하고 있으므로 Joint도 Track A 기준
      하나만 계산한다).
    - 라벨 예산(10%)도 적용하지 않는다 — "라벨을 전부 쓸 수 있다면"까지
      포함한 절대 상한이 목적이라, 10% 라벨 예산판 상한(예산은 그대로 두고
      순서만 없앤 버전)은 별도로 필요하면 추후 추가할 수 있는 여지로 남긴다.
    - threshold는 0.5 고정(as=none과 같은 관례) 대신 실제로는 두 값을
      함께 낸다: `f1`(0.5 고정)과 `best_f1_reference`(precision_recall_
      curve로 찾은 도달 가능 최댓값, 기존 `_best_f1_achievable()` 재사용) —
      0.5가 막 학습한 모델에 우연히 안 맞을 수 있어(재보정 메커니즘이
      아예 없음) 상한을 과소평가할 위험을 줄인다.
    - `bwt`는 정의되지 않는다(순차 학습 자체가 없음) — 0.0으로 채운다.
      `perf_matrix`도 T×T가 아니라 1×T(유일한 학습 결과를 각 experience의
      test에서 채점한 것) — 다른 콤보와 다른 모양임을 명시적으로 기록.
      `per_category_final`은 스키마는 맞추되(`first_seen_round`/`bwt`
      필드 존재) bwt=0.0으로 고정(망각이 발생할 수 없으므로).

    Returns:
        `REQUIRED_RESULT_FIELDS`를 전부 채운 결과 dict(`validate_result()`
        통과) — `drift_detector`등 5-슬롯 필드는 전부 `"joint"`라는 특수
        마커(`"none"`이 아님 — dd=none/ss=random/mm=none/af=none/as=none인
        진짜 78개 중 하나(naive fine-tuning 기준선)와 혼동되면 안 되므로).
        `combo_id="JOINT_BASELINE"`, `track="JOINT"`.
    """
    input_dim = dataset["input_dim"]
    hidden_dim, latent_dim = ssf_backbone_dims(input_dim)
    seed = global_hparams.get("seed", 42)
    torch.manual_seed(seed)

    experiences = dataset["experiences"]
    all_test_splits = [(e["test_X"], e["test_y"]) for e in experiences]
    all_test_categories = [e.get("test_category") for e in experiences]
    has_category = all(c is not None for c in all_test_categories)

    device_t = torch.device(device)
    model = FCLAutoEncoder(input_dim=input_dim, hidden_dim=hidden_dim,
                            latent_dim=latent_dim).to(device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=global_hparams["learning_rate"])
    batch_size = global_hparams["batch_size"]
    epochs = global_hparams["epochs_per_experience"]

    all_train_X = torch.cat([e["train_X"] for e in experiences], dim=0).to(device_t)
    all_train_y = torch.cat([e["train_y"] for e in experiences], dim=0).to(device_t)
    n = len(all_train_X)

    model.train()
    t0 = time.time()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device_t)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch_x, batch_y = all_train_X[idx], all_train_y[idx]
            optimizer.zero_grad()
            _, _, logit = model(batch_x)
            loss = F.binary_cross_entropy_with_logits(logit.squeeze(-1), batch_y.float())
            loss.backward()
            optimizer.step()
    training_time_sec = time.time() - t0

    model.eval()
    per_exp_f1 = []
    per_exp_scores, per_exp_labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for test_x, test_y in all_test_splits:
            _, _, logit = model(test_x.to(device_t))
            scores = torch.sigmoid(logit.squeeze(-1)).cpu()
            per_exp_scores.append(scores)
            per_exp_labels.append(test_y)
            preds = (scores > 0.5).long()
            per_exp_f1.append(f1_score(test_y, preds))
    inference_time = time.time() - t0
    n_inference_samples = sum(len(tx) for tx, _ in all_test_splits)
    avg_inference_latency_ms = (inference_time / max(n_inference_samples, 1)) * 1000.0

    pooled_scores = torch.cat(per_exp_scores)
    pooled_labels = torch.cat(per_exp_labels)
    pooled_preds = (pooled_scores > 0.5).long()
    final_f1 = f1_score(pooled_labels, pooled_preds)
    final_precision = precision_score(pooled_labels, pooled_preds)
    final_recall = recall_score(pooled_labels, pooled_preds)
    final_pr_auc = pr_auc(pooled_labels, pooled_scores)
    try:
        final_roc_auc = float(roc_auc_score(pooled_labels.numpy(), pooled_scores.numpy()))
    except ValueError:
        # run_combo_full()과 동일한 이유로 NaN(2026-09-14 일관성 수정) — 위 참고.
        final_roc_auc = float("nan")
    best_f1_reference = _best_f1_achievable(pooled_scores, pooled_labels)

    per_category_final: Dict[str, Any] = {}
    if has_category:
        pooled_test_category = np.concatenate(
            [np.asarray(c) for c in all_test_categories])
        counts = per_category_counts(pooled_labels.numpy(), pooled_preds.numpy(), pooled_test_category)
        for cat, rec in per_category_recall(counts).items():
            n_test = int(np.sum(pooled_test_category == cat))
            per_category_final[cat] = {
                "n_test": n_test, "first_seen_round": 0,
                "recall_at_first_seen": float(rec), "recall_final": float(rec),
                "bwt": 0.0,
            }
        n_normal = int((pooled_labels.numpy() == 0).sum())
        n_fp = int(((pooled_labels.numpy() == 0) & (pooled_preds.numpy() == 1)).sum())
        normal_fpr = float(n_fp / n_normal) if n_normal > 0 else 0.0
    else:
        normal_fpr = float("nan")

    result = {
        "combo_id": "JOINT_BASELINE",
        "exp_name": f"JOINT_BASELINE__{dataset_name}",
        "dataset": dataset_name,
        "drift_detector": "joint", "sample_selector": "joint",
        "memory_manager": "joint", "anti_forgetting": "joint", "anomaly_scorer": "joint",
        "track": "JOINT",
        "f1": final_f1, "precision": final_precision, "recall": final_recall,
        "pr_auc": final_pr_auc,
        "bwt": 0.0,  # 순차 학습이 없어 정의되지 않음 — 함수 docstring 참고.
        "perf_matrix": [per_exp_f1],  # 1×T — 다른 콤보(T×T)와 모양이 다름.
        "roc_auc": final_roc_auc,
        "labeling_cost": 1.0,  # 라벨 예산 없음 — 전체 라벨 사용.
        "training_time_sec": training_time_sec,
        "memory_footprint": 0, "memory_footprint_peak": 0, "memory_footprint_avg": 0.0,
        "avg_inference_latency_ms": avg_inference_latency_ms,
        "best_f1_reference": best_f1_reference,
        "seed": seed,
        "code_version": code_version if code_version is not None else compute_code_version(),
        "drift_detected_per_round": [], "drift_score_per_round": [], "n_drift_detected": 0,
        "per_category_final": per_category_final,
        "per_category_recall_history": {cat: [v["recall_final"]] for cat, v in per_category_final.items()},
        "normal_fpr": normal_fpr, "normal_fpr_history": [normal_fpr],
    }
    validate_result(result)
    return result


def load_smoke_passed_combo_ids(smoke_results_path: str,
                                 code_version: Optional[str] = None) -> Optional[Dict[str, set]]:
    """데이터셋별 스모크 테스트 통과 combo_id 집합을 반환한다.

    combo_id만으로 묶으면(데이터셋 구분 없이) 한 데이터셋에서 통과한 조합이
    다른 데이터셋에서 실제로 실패해도(예: 특정 데이터셋에서만 score 분포가
    퇴화하는 경우) 그 데이터셋 그리드에 그대로 포함되는 문제가 있었다 —
    데이터셋별로 분리해 반환한다.

    `code_version`(2026-09-14 재검토로 발견·수정): 이전엔 `r["passed"]`만
    보고 그 판정이 어느 코드 버전에서 나왔는지 전혀 확인하지 않았다 —
    `run_combo_full()`/`run_joint_baseline()`의 개별 결과 캐시는
    code_version이 다르면 재계산하도록 이미 지켜지는데, "어느 조합을 아예
    실행 대상에 넣을지"를 정하는 이 스모크 필터링 단계에는 같은 안전장치가
    없었다 — 스모크 이후 코드가 바뀌어도(이 세션의 RNG 격리처럼) 낡은
    판정으로 78개 중 무엇을 실행할지가 결정되는 구멍이었다. `code_version`을
    넘기면 그것과 일치하는 결과만 "통과"로 인정하고, 낡은 항목이 있으면
    경고를 출력한다(자동으로 재실행을 트리거하지는 않음 — smoke_test.py를
    먼저 재실행하라는 신호만 준다)."""
    if not os.path.exists(smoke_results_path):
        return None
    with io.open(smoke_results_path, encoding="utf-8") as f:
        smoke_results = json.load(f)
    passed_by_dataset: Dict[str, set] = {}
    n_stale = 0
    for r in smoke_results:
        if code_version is not None and r.get("code_version") != code_version:
            if r["passed"]:
                n_stale += 1
            continue
        if r["passed"]:
            passed_by_dataset.setdefault(r["dataset"], set()).add(r["combo_id"])
    if code_version is not None and n_stale > 0:
        print(f"경고: smoke_test_results.json의 {n_stale}개 통과 판정이 현재 "
              f"code_version({code_version})과 다른 버전에서 나왔습니다 — "
              "이 조합들은 실행 대상에서 제외됩니다. `python -m "
              "testbed.experiments.smoke_test`로 먼저 재검증하세요("
              "또는 낡은 판정을 감수하려면 --ignore-smoke).")
    return passed_by_dataset


def run_grid(datasets: List[str] = ("nsl-kdd", "unsw-nb15"),
             smoke_results_path: Optional[str] = None,
             device: str = "cpu",
             shard: Optional[tuple] = None,
             track: Optional[str] = None,
             ignore_smoke: bool = False,
             blurry_ratio: float = 0.0,
             min_per_round_share: int = 30,
             skip_joint_baseline: bool = False) -> List[Dict[str, Any]]:
    global_hparams, component_hparams = _load_configs()
    code_version = compute_code_version()
    print(f"code_version={code_version} (components/base/pipeline/dataset_loader 해시 — "
          f"이 값이 캐시된 결과 파일과 다르면 재계산합니다)")
    # 2026-09-04 추가 — 스모크 테스트가 일부 조합을 실패로 걸러도(예:
    # 데이터 규모/난이도가 데이터셋마다 달라 특정 데이터셋에서만 유독 몇
    # 개가 실패하는 경우) 사용자가 78개 유효 조합 전체를 강제로 돌려보고 싶을
    # 때를 위한 탈출구다 — 스모크 게이트 자체(15절)를 없애는 게 아니라,
    # 이번 실행 한 번만 그 필터링을 건너뛴다. `passed_by_dataset=None`으로
    # 두면 아래 루프가 "스모크 결과 파일이 없을 때"와 완전히 같은 경로
    # (전체 조합 실행)를 타므로 별도 분기를 새로 만들 필요가 없다.
    if ignore_smoke:
        passed_by_dataset = None
        print("경고: --ignore-smoke로 스모크 테스트 필터링을 건너뜁니다 — "
              "유효 조합 전체(78개)를 실행합니다.")
    else:
        if smoke_results_path is None:
            smoke_results_path = os.path.join(_TESTBED_ROOT, "experiments", "smoke_test_results.json")
        passed_by_dataset = load_smoke_passed_combo_ids(smoke_results_path, code_version=code_version)

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
                                seed=global_hparams["seed"],
                                blurry_ratio=blurry_ratio,
                                min_per_round_share=min_per_round_share)
        # blurry_ratio=0(기본값)이면 라벨을 안 건드려 결과 파일명·캐시가
        # 기존과 완전히 동일하다 — i-Blurry(data/dataset_loader.py의
        # `_class_incremental_split` docstring 참고)를 켰을 때만 별도
        # 라벨을 붙여 disjoint 결과와 겹쳐 쓰지 않는다.
        result_label = (dataset_name if blurry_ratio <= 0
                         else f"{dataset_name}-blurry{int(round(blurry_ratio * 100))}")

        # 6.2 Joint/Offline 상한선(2026-09-13) — 라운드/블러링 개념 자체가
        # 없는 baseline이라 blurry_ratio와 무관하게 데이터셋당 한 번만
        # 계산한다(run_joint_baseline() docstring 참고). 콤보 그리드와
        # 같은 code_version 캐시 규칙 — 코드가 안 바뀌었으면 재계산하지 않는다.
        #
        # **반드시 disjoint(blurry_ratio=0) 데이터로만 계산한다(2026-09-14
        # 재검토로 발견·수정)** — 위에서 로드한 `dataset`은 이 실행의
        # `blurry_ratio` 인자를 그대로 반영한 것이라, 여기서 그걸 그대로
        # 넘기면 "blurry_ratio와 무관"이라는 주석과 달리 실제로는
        # blurry_ratio=0.3으로 돌릴 때 블러링된 행 순서로 학습된 모델이
        # 저장된다. `JOINT_BASELINE__{dataset_name}.json`이라는 파일명
        # 자체가 blurry_ratio를 구분하지 않으므로(의도적으로 공유),
        # `--blurry-ratio 0`과 `--blurry-ratio 0.3`을 문서(--blurry-ratio
        # 도움말)가 권장하는 대로 순서 바꿔가며 두 번 실행하면 먼저 실행한
        # 쪽의 blurry_ratio가 이후 실행의 캐시에 조용히 고정되는 순서
        # 의존적 버그가 있었다(i-Blurry가 표본을 라운드 간에 "이동"만 하고
        # 중복·삭제는 안 하므로 최종 f1 자체가 크게 틀리지는 않지만, 같은
        # 파일명이 실행 순서에 따라 서로 다른 데이터로 계산된 결과를 담게
        # 되는 건 이 프로젝트의 핵심 원칙("코드가 바뀌면 캐시를 못 믿는다",
        # smoke_test.py의 code_version 검사와 같은 정신)과 정면으로
        # 어긋난다). 항상 별도로 disjoint 데이터셋을 로드해 계산한다 —
        # dataset_loader.py의 디스크 캐시가 blurry_ratio를 캐시 키에
        # 포함하므로(`_dataset_cache_key`) blurry_ratio=0 데이터셋이 이미
        # 있으면 재계산 없이 바로 재사용된다.
        if not skip_joint_baseline:
            joint_path = os.path.join(RESULTS_DIR, f"JOINT_BASELINE__{dataset_name}.json")
            cached_joint = _load_json_safely(joint_path) if os.path.exists(joint_path) else None
            if cached_joint is not None and cached_joint.get("code_version") == code_version:
                print(f"[{dataset_name}] JOINT_BASELINE 이미 결과 있음(코드 버전 일치), 건너뜀")
            else:
                print(f"[{dataset_name}] JOINT_BASELINE 계산 중...")
                t0 = time.time()
                disjoint_dataset = (
                    dataset if blurry_ratio <= 0 else
                    load_dataset(dataset_name, base_dir=_REPO_ROOT,
                                 seed=global_hparams["seed"], blurry_ratio=0.0,
                                 min_per_round_share=min_per_round_share))
                # min_per_round_share=0.0일 때 실제 분할에 전혀 쓰이지 않지만
                # (`_class_incremental_split`의 blurry_ratio<=0 "빠른 경로"),
                # `_dataset_cache_key()`가 이 값도 캐시 키에 포함시키므로(다른
                # --min-per-round-share로 별도 돌린 disjoint 그리드와 디스크
                # 캐시를 공유하려면) 이 실행의 실제 인자를 그대로 넘긴다 —
                # 안 넘기면 내용은 동일한데 캐시 파일만 중복 생성된다(2026-09-14
                # Round-6 감사로 발견, 정확성 문제는 아니었음).
                joint_result = run_joint_baseline(dataset_name, disjoint_dataset, global_hparams,
                                                   device, code_version=code_version)
                print(f"[{dataset_name}] JOINT_BASELINE f1={joint_result['f1']:.3f} "
                      f"best_f1_reference={joint_result['best_f1_reference']:.3f} "
                      f"pr_auc={joint_result['pr_auc']:.3f} ({time.time() - t0:.1f}s)")
                _atomic_write_json(joint_path, joint_result)

        for combo in combos:
            combo_id = make_combo_id(combo)
            out_path = os.path.join(RESULTS_DIR, f"{combo_id}__{result_label}.json")
            if os.path.exists(out_path):
                # 이전 실행이 중간에 끊긴 뒤 이어서 돌릴 때(조합당 시간이
                # 오래 걸리는 경우 재계산 낭비 방지) — 결과
                # 파일이 있으면 다시 계산하지 않고 건너뛴다. 단, 그 파일이
                # 지금 코드로 계산된 게 맞는지 code_version으로 확인한 뒤에만
                # 건너뛴다(컴포넌트를 고친 뒤에도 낡은 결과가 재사용되는
                # 사고를 막기 위함). code_version 필드가 없으면(도입 이전
                # 파일) 무조건 재계산한다. `_load_json_safely()`가 손상된
                # 캐시 파일이면 None을 반환하므로 그 경우도 재계산으로 넘어간다.
                cached = _load_json_safely(out_path)
                if cached is not None and cached.get("code_version") == code_version:
                    print(f"[{result_label}] {combo_id} 이미 결과 있음(코드 버전 일치), 건너뜀")
                    continue
                if cached is not None:
                    print(f"[{result_label}] {combo_id} 결과는 있지만 코드 버전이 달라 재계산합니다 "
                          f"(cached={cached.get('code_version')!r}, current={code_version!r})")

            t0 = time.time()
            try:
                result = run_combo_full(
                    combo, result_label, dataset, global_hparams, component_hparams,
                    labeling_budget, device, code_version=code_version)
            except Exception as exc:
                # 조합 하나가 예외로 죽어도(수치 불안정, GPU OOM 등) 전체
                # 그리드가 죽지 않고 이 조합만 "실패"로 기록한 뒤 계속
                # 진행한다. 상세 기록은 results/failures/ 아래 별도 파일에
                # 남긴다(FAILURES_DIR 정의 참고).
                elapsed = time.time() - t0
                tb = traceback.format_exc()
                print(f"[{result_label}] {combo_id} 실패({elapsed:.1f}s 경과 후) — "
                      f"{type(exc).__name__}: {exc}")
                print(tb)
                os.makedirs(FAILURES_DIR, exist_ok=True)
                failure_record = {
                    "combo_id": combo_id,
                    "exp_name": f"{combo_id}__{result_label}",
                    "dataset": result_label,
                    "track": combo["track"],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": tb,
                    "elapsed_sec": elapsed,
                    "code_version": code_version,
                }
                failure_path = os.path.join(FAILURES_DIR, f"{combo_id}__{result_label}.json")
                _atomic_write_json(failure_path, failure_record)
                failed_combos.append(failure_record)
                continue

            elapsed = time.time() - t0
            print(f"[{result_label}] {result['combo_id']} f1={result['f1']:.3f} "
                  f"pr_auc={result['pr_auc']:.3f} bwt={result['bwt']:.3f} ({elapsed:.1f}s)")

            _atomic_write_json(out_path, result)
            # 이전 시도에서 실패해 failures/에 남아있던 기록이 있다면 지운다
            # (안 지우면 성공한 조합인데도 실패 목록에 계속 남는다).
            stale_failure_path = os.path.join(FAILURES_DIR, f"{combo_id}__{result_label}.json")
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
        help="쉼표로 구분한 데이터셋 이름 목록 (nsl-kdd, unsw-nb15, x-iiotid 중). "
             "기본값은 X-IIoTID를 포함하지 않는다 — 필요하면 명시적으로 "
             "추가할 것. 예: --datasets nsl-kdd,unsw-nb15,x-iiotid")
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
    parser.add_argument(
        "--ignore-smoke", action="store_true",
        help="스모크 테스트 결과와 무관하게 enumerate_valid_combos()의 78개 "
             "조합 전체를 실행한다(2026-09-04 추가). 예: 특정 데이터셋에서만 "
             "78개 중 일부가 스모크 게이트에 걸려도 전체를 강행하고 싶을 때. "
             "생략하면 기존처럼 smoke_test_results.json으로 필터링한다.")
    parser.add_argument(
        "--blurry-ratio", type=float, default=0.0,
        help="i-Blurry 재등장 비율(2026-09-12 도입). 0(기본값)이면 "
             "기존 disjoint 분할과 완전히 동일 — 결과 파일명도 그대로 "
             "{combo_id}__{dataset}.json. 0보다 크면(예: 0.3) 블러링 대상 "
             "카테고리 표본의 그 비율만큼을 다른 라운드에도 재배정하고, "
             "결과 파일명이 {combo_id}__{dataset}-blurry{N}.json으로 바뀌어 "
             "disjoint 결과와 겹쳐 쓰지 않는다. Disjoint와 Blurry를 둘 다 "
             "보려면 이 스크립트를 두 번(--blurry-ratio 0, --blurry-ratio "
             "0.3) 실행한다.")
    parser.add_argument(
        "--min-per-round-share", type=int, default=30,
        help="i-Blurry 블러링 대상 카테고리 판정 임계값(기본 30) — "
             "`dataset_loader._class_incremental_split` docstring 참고. "
             "--blurry-ratio가 0이면 무시된다.")
    parser.add_argument(
        "--skip-joint-baseline", action="store_true",
        help="6.2 Joint/Offline 상한선(results/JOINT_BASELINE__{dataset}.json) "
             "계산을 건너뛴다. 기본은 데이터셋마다 한 번(코드 버전 캐시 적용) "
             "계산한다 — `run_joint_baseline()` 참고.")
    args = parser.parse_args()
    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    shard_arg = None
    if args.shard is not None:
        shard_idx_str, n_shards_str = args.shard.split("/")
        shard_idx, n_shards = int(shard_idx_str), int(n_shards_str)
        assert 0 <= shard_idx < n_shards, "--shard 는 0 <= i < n 이어야 함"
        shard_arg = (shard_idx, n_shards)
    run_grid(datasets=dataset_list, device=args.device, shard=shard_arg, track=args.track,
             ignore_smoke=args.ignore_smoke, blurry_ratio=args.blurry_ratio,
             min_per_round_share=args.min_per_round_share,
             skip_joint_baseline=args.skip_joint_baseline)
