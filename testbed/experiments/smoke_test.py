"""스모크 테스트 — PRD 15절. 그리드 실행(Phase 3) 전 필수 게이트.

valid combo 각각에 대해 전체 데이터의 일부(첫 2개 experience)만으로 15.1~15.5의
정량 기준을 그대로 assert 조건으로 적용한다. 하나라도 통과하지 못하면 그
combo는 실패(fail) 처리하고 본 그리드(Phase 3)에서 실행하지 않는다.
"""

import io
import json
import math
import os
from typing import Any, Dict, List

import torch
import yaml

from testbed.base import FCLAutoEncoder, ssf_backbone_dims
from testbed.common.compatibility import enumerate_valid_combos
from testbed.common.result_schema import make_combo_id
from testbed.components.cndids.cndids_anti_forgetting import CNDIDSAntiForgetting
from testbed.data.dataset_loader import load_dataset
from testbed.pipeline import CLClient

SMOKE_N_EXPERIENCES = 2
_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTBED_ROOT = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_TESTBED_ROOT)


def _load_yaml(path: str) -> Dict[str, Any]:
    with io.open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_configs():
    global_hparams = _load_yaml(os.path.join(_TESTBED_ROOT, "configs", "global_hparams.yaml"))
    component_dir = os.path.join(_TESTBED_ROOT, "configs", "component_hparams")
    component_hparams = {
        name[:-5]: _load_yaml(os.path.join(component_dir, name))
        for name in os.listdir(component_dir) if name.endswith(".yaml")
    }
    return global_hparams, component_hparams


def _flatten_params(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def run_smoke_test_for_combo(combo: Dict[str, Any], dataset: Dict[str, Any],
                              global_hparams: Dict[str, Any],
                              component_hparams: Dict[str, Dict[str, Any]],
                              labeling_budget: Dict[str, Any],
                              device: str = "cpu") -> Dict[str, Any]:
    """단일 combo에 대해 15.1~15.5 게이트를 적용한다.

    Returns:
        {'combo_id', 'combo', 'passed', 'failures', 'warnings'}
    """
    failures: List[str] = []
    warnings: List[str] = []

    input_dim = dataset["input_dim"]
    hp = dict(global_hparams)
    hp["_input_dim"] = input_dim
    # 2026-08-11: hidden_dim/latent_dim을 SSF 원 논문 공식으로 이 데이터셋의
    # input_dim에 맞춰 그때그때 계산한다(configs/global_hparams.yaml 주석
    # 참고) — grid_runner.py의 run_combo_full과 동일한 근거.
    hp["hidden_dim"], hp["latent_dim"] = ssf_backbone_dims(input_dim)
    # Track B(CND-IDS) 에폭/배치크기 오버라이드 — grid_runner.py의
    # run_combo_full과 동일한 근거(configs/global_hparams.yaml 주석 참고).
    if combo["track"] == "B":
        hp["epochs_per_experience"] = global_hparams["epochs_per_experience_track_b"]
        hp["batch_size"] = global_hparams["batch_size_track_b"]

    torch.manual_seed(hp.get("seed", 42))
    model = FCLAutoEncoder(input_dim=input_dim, hidden_dim=hp["hidden_dim"],
                            latent_dim=hp["latent_dim"])
    client = CLClient(model, combo, hp, component_hparams, device=device)

    experiences = dataset["experiences"][:SMOKE_N_EXPERIENCES]
    all_test_splits = [(e["test_X"], e["test_y"]) for e in experiences]

    batch_size = hp["batch_size"]
    epochs = hp["epochs_per_experience"]

    for exp_idx, e in enumerate(experiences):
        theta_before = _flatten_params(model)

        out = client.run_experience(
            exp_idx, e["train_X"], e["train_y"], all_test_splits, labeling_budget)

        theta_after = _flatten_params(model)

        # ---- 15.1a: 학습이 실제로 일어났는가 ----
        delta_norm = torch.norm(theta_after - theta_before).item()
        if delta_norm <= 1e-6:
            failures.append(
                f"exp{exp_idx}: 15.1a 학습이 사실상 일어나지 않음 (delta_norm={delta_norm:.2e})")

        # ---- 15.1b: optimizer.step 호출 횟수 ----
        n_selected = out["n_selected"]
        expected_steps = epochs * math.ceil(n_selected / batch_size) if n_selected > 0 else 0
        if out["n_optimizer_steps"] != expected_steps:
            failures.append(
                f"exp{exp_idx}: 15.1b optimizer.step 호출 횟수 불일치 "
                f"(actual={out['n_optimizer_steps']}, expected={expected_steps})")

        # ---- 15.1c: 라벨 예산 5% 이내 일치 ----
        label_budget_int = out["label_budget_int"]
        if label_budget_int > 0:
            deviation = abs(n_selected - label_budget_int) / label_budget_int
            if deviation > 0.05:
                failures.append(
                    f"exp{exp_idx}: 15.1c label_budget 5% 초과 이탈 "
                    f"(n_selected={n_selected}, label_budget={label_budget_int}, "
                    f"deviation={deviation:.2%})")

        # ---- 15.2/15.3: 예측 퇴화 여부 + threshold 범위 ----
        all_scores = torch.cat(out["eval_scores"])
        all_labels = torch.cat(out["eval_labels"])
        threshold = out["threshold"]
        preds = (all_scores > threshold).long()

        ratio0 = (preds == 0).float().mean().item()
        ratio1 = (preds == 1).float().mean().item()
        majority_ratio = max(ratio0, ratio1)
        if majority_ratio >= 1.0:
            failures.append(f"exp{exp_idx}: 15.2 예측이 완전히 한 클래스로 퇴화")
        elif majority_ratio >= 0.99:
            warnings.append(
                f"exp{exp_idx}: 15.2 경고 - 다수 클래스 비율 {majority_ratio:.4f} >= 0.99")

        # min-max range는 극단치 하나에도 확 벌어진다 — CADE-MAD처럼
        # median/MAD 정규화 거리 기반 스코어는 원래 설계상 대다수가 낮은
        # 값에 뭉치고 진짜 이상치 몇 개만 매우 큰 값을 내는 게 정상 동작
        # (2026-07-30 CICIDS2018 전체 데이터 실행에서 실측 확인: 정상
        # 조합인데도 min-max range 기준으로 오탐이 남 — 근거는
        # docs/metric_justification.md 참고). 1~99 percentile 기반 범위로
        # 바꿔 소수 극단치에 흔들리지 않게 한다.
        q01, q99 = torch.quantile(all_scores, torch.tensor([0.01, 0.99])).tolist()
        robust_range = q99 - q01
        score_std = all_scores.std().item()
        if robust_range > 0 and score_std < robust_range * 0.01:
            failures.append(
                f"exp{exp_idx}: 15.2 score 분포가 사실상 상수 출력 "
                f"(std={score_std:.4e}, robust_range(1~99%)={robust_range:.4e})")

        s_min, s_max = all_scores.min().item(), all_scores.max().item()
        if not (s_min <= threshold <= s_max):
            failures.append(
                f"exp{exp_idx}: 15.3 threshold가 score 범위를 벗어남 "
                f"(threshold={threshold:.4f}, range=[{s_min:.4f},{s_max:.4f}])")

        # ---- 15.5: predict() 출력 shape 검증 ----
        test_x0, test_y0 = all_test_splits[0]
        model.eval()
        with torch.no_grad():
            z0, _, _ = model(test_x0.to(client.device))
        pred0 = client.anomaly_scorer.predict(z0, threshold).cpu()
        if pred0.shape != test_y0.shape:
            failures.append(
                f"exp{exp_idx}: 15.5 predict() shape 불일치 "
                f"(pred={tuple(pred0.shape)}, label={tuple(test_y0.shape)})")

        # ---- 15.4: Track B pseudo-label 균형 (경고 전용, CNDIDS 한정) ----
        if isinstance(client.anti_forgetting, CNDIDSAntiForgetting):
            ratio = client.anti_forgetting.last_pseudo_label_ratio
            if ratio is not None and ratio >= 0.95:
                warnings.append(
                    f"exp{exp_idx}: 15.4 경고 - pseudo-label 쏠림 {ratio:.4f} >= 0.95")

    combo_id = make_combo_id(combo)
    return {
        "combo_id": combo_id,
        "combo": combo,
        "passed": len(failures) == 0,
        "failures": failures,
        "warnings": warnings,
    }


def run_all(dataset_name: str = "nsl-kdd", device: str = "cpu") -> List[Dict[str, Any]]:
    global_hparams, component_hparams = _load_configs()
    # 두 데이터셋 모두 동일한 프로토콜(dataset_loader.py 기본값) — 원본
    # train/test 파일 분리 유지, 각 파일 내부는 고정 seed로 섞은 뒤 분할.
    dataset = load_dataset(dataset_name, base_dir=_REPO_ROOT,
                            n_experiences=global_hparams["n_experiences"],
                            seed=global_hparams["seed"])
    labeling_budget = global_hparams["labeling_budget"]

    results = []
    for combo in enumerate_valid_combos():
        result = run_smoke_test_for_combo(
            combo, dataset, global_hparams, component_hparams, labeling_budget, device)
        result["dataset"] = dataset_name
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['combo_id']} ({dataset_name})")
        for f in result["failures"]:
            print(f"    FAIL: {f}")
        for w in result["warnings"]:
            print(f"    WARN: {w}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CL-NIDS-Bench 스모크 테스트")
    parser.add_argument(
        "--device", default="cpu",
        help="torch device 문자열 (예: 'cpu', 'cuda'). GPU 환경에서는 "
             "'--device cuda'로 실행한다.")
    parser.add_argument(
        "--datasets", default="nsl-kdd,unsw-nb15",
        help="쉼표로 구분한 데이터셋 이름 목록 (nsl-kdd, unsw-nb15, cicids2018 중).")
    args = parser.parse_args()
    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]

    all_results = []
    for ds_name in dataset_list:
        all_results.extend(run_all(ds_name, device=args.device))

    n_pass = sum(1 for r in all_results if r["passed"])
    print(f"\n{n_pass}/{len(all_results)} combo-dataset 조합이 스모크 테스트 통과")

    # 데이터셋을 나눠서(예: NSL/UNSW 먼저, CICIDS2018 나중) 여러 번 실행해도
    # 이전 실행 결과가 사라지지 않도록, 이번에 실행한 데이터셋에 해당하는
    # 항목만 교체하고 나머지는 그대로 보존한다(무조건 덮어쓰지 않음).
    out_path = os.path.join(_TESTBED_ROOT, "experiments", "smoke_test_results.json")
    existing_results = []
    if os.path.exists(out_path):
        with io.open(out_path, encoding="utf-8") as f:
            existing_results = json.load(f)
    kept = [r for r in existing_results if r.get("dataset") not in dataset_list]
    merged_results = kept + all_results

    with io.open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {out_path} (이번 실행 {len(all_results)}개 + 기존 보존 {len(kept)}개 "
          f"= 총 {len(merged_results)}개)")
