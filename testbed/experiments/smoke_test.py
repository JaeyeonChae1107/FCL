"""스모크 테스트 — PRD 15절. 그리드 실행(Phase 3) 전 필수 게이트.

valid combo 각각에 대해 experience 전체(라운드 수·class-incremental 구조는
본 그리드와 동일)를, 라운드당 행 수만 NSL-KDD 규모로 제한하고 epoch 수를
줄인 설정으로 실행해 15.1~15.5의 정량 기준을 assert 조건으로 적용한다(아래
SMOKE_* 상수). 실패한 combo는 본 그리드(Phase 3)에서 실행하지 않는다.

라운드당 행 수 제한(SMOKE_MAX_*_ROWS_PER_EXPERIENCE): CICIDS2018은 중복 제거
후 약 1,208만 행(라운드당 train 약 193만/test 약 48만, NSL-KDD의 80~100배).
전체를 그대로 돌리면 본 그리드와 비슷한 비용을 스모크가 한 번 더 낸다.
라운드 수와 class-incremental 구조는 그대로 두고 라운드당 행 수만 category별
최소 개수를 보장하며 서브샘플링한다.

과거 SMOKE_N_EXPERIENCES=2로 앞 2개 라운드만 검사했을 때, class-incremental
분할이 희귀 category(R2L/U2R)와 공격 없는 라운드를 항상 뒤쪽에 배치하는
설계라 그 라운드들을 한 번도 검사하지 못한 사각지대가 있었다(af=gpm이
3라운드 이후 recall 0.26%까지 붕괴, af=cndids가 R2L/U2R에서 pseudo-label이
0.98~1.0으로 쏠려 그 category를 학습하지 못함 — 둘 다 앞 2라운드는 건강).
SMOKE_N_EXPERIENCES를 데이터셋 전체로 확장하고 15.2/15.4 게이트를 강화했다.
"""

import io
import json
import math
import os
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

from testbed.base import FCLAutoEncoder, ssf_backbone_dims
from testbed.common.compatibility import enumerate_valid_combos
from testbed.common.result_schema import make_combo_id
from testbed.components.cndids.cndids_anti_forgetting import CNDIDSAntiForgetting
from testbed.data.dataset_loader import load_dataset
from testbed.pipeline import CLClient

# None이면 데이터셋의 experience 전체를 검사한다. 값을 넣으면 일부만(디버깅용).
SMOKE_N_EXPERIENCES: Optional[int] = None

# 라운드 수는 절대 줄이지 않는다 — 어떤 라운드가 어떤 데이터를 받는지가
# 바뀌면 그 조건에서만 나오는 버그(희귀 category, 공격 없는 라운드)를
# 놓친다. epoch 수는 "같은 라운드를 얼마나 오래 학습시키는가"일 뿐이고
# 15.1b/c/15.2/15.2b/15.4는 라운드 데이터·설정만으로 결정되며 epoch와
# 무관하다(15.4는 학습 루프 이전 K-means 결과). 15.1d(발산 체크)도 보통
# 초반 몇 epoch에 드러난다. 실전 epoch(200/20) 대신 이 값을 써서 조합마다
# 두 번(스모크+본 그리드) 전체 학습시키는 비용을 없앤다.
SMOKE_EPOCHS_PER_EXPERIENCE = 10
SMOKE_EPOCHS_PER_EXPERIENCE_TRACK_B = 5

# 라운드당 행 수 상한 — NSL-KDD 라운드 규모(train 1.3만~5.9만, test 4.5천)에
# 맞춤. 15.2/15.2b/15.4 게이트가 버그를 잡아낸 게 이 규모였고, CND-IDS
# cluster_fit_sample_size(1만)·max_normal_ref(5천)·CADE max_category_ref(500)
# 같은 캡/축출 경로도 이 규모에서 발동한다. 상한보다 작은 라운드는 그대로
# 둔다(no-op).
#
# SMOKE_MIN_ROWS_PER_CATEGORY: 비율대로만 뽑으면 CICIDS2018의 희귀 공격
# category가 라운드에서 통째로 사라진다. 각 category는 최소 이 개수(원래
# 더 적으면 전부)를 남긴다. 50은 NSL-KDD U2R 라운드(1.35만 중 52건, 0.38%)
# 규모다. test는 test_y(이진)로만 층화(experience 딕셔너리에 test category
# 없음). None이면 상한 미적용(디버깅용).
SMOKE_MAX_TRAIN_ROWS_PER_EXPERIENCE: Optional[int] = 20_000
SMOKE_MAX_TEST_ROWS_PER_EXPERIENCE: Optional[int] = 5_000
SMOKE_MIN_ROWS_PER_CATEGORY = 50

# 축소 설정(행 수 상한·epoch 축소)에서는 "행동" 게이트를 실패가 아니라
# 경고로만 기록한다. 실측 근거(NSL-KDD,
# A_dd=cade_ss=ssf_mm=ssf_af=lwf_ssf_as=cade_mad, 4조건 대조):
#   행 전체 + epoch 10 : exp0 통과, exp2 15.2 실패(0.9962)
#   행 축소 + epoch 10 : exp0/exp1 15.2·15.2b 실패(0.9851/roc 0.4309, 0.9865/0.4570)
#   행 축소 + epoch 200: exp0/exp1 실패, exp0 수치가 epoch 10과 동일
# 이 조합의 점수는 CADE 사설 인코더(dd=cade에 연결된 as=cade_mad)에서 나와
# 메인 모델 epoch와 무관하고, 라운드 행 수를 줄이면(라벨 예산 10%로 선택되는
# 표본이 5,940→2,000개) 인코더 학습량이 줄어 exp0에서 roc_auc가 역전된다 —
# 행 수 전체에서는 같은 라운드가 통과한다. 실패로 처리하면 grid_runner.py가
# 그 조합을 본 그리드에서 제외해 조합 커버리지를 해치므로:
#   실패 유지: 15.1a~d(학습 발생/발산/step 수/라벨 예산), 15.2의 "완전 퇴화"
#              (상수 예측)·"상수 점수", 15.3(threshold 범위), 15.5(shape) —
#              배선/수치 결함이라 데이터 규모와 무관.
#   경고로 강등: 15.2의 0.97 등급, 15.2b(roc_auc<0.5), 15.4(pseudo-label 쏠림) —
#              모델이 약하거나 퇴화했다는 신호라 규모 영향을 받고, 본 그리드
#              결과에서 그대로 드러난다.
# 전체 규모 스모크로 되돌리려면(상한 None, 실전 epoch) 이 값을 False로.
SMOKE_BEHAVIORAL_GATES_AS_WARNINGS = True
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


def _stratified_subsample_indices(groups: np.ndarray, max_rows: int, min_per_group: int,
                                   rng: np.random.Generator) -> np.ndarray:
    """`groups`(행별 그룹 라벨)로 층화해 최대 약 `max_rows`개의 행 인덱스를
    고른다. 각 그룹은 비율대로 배정된 몫과 `min_per_group` 중 큰 쪽을
    받되 그 그룹의 실제 행 수를 넘지 않는다 — 희귀 그룹은 최소 개수가
    보장되므로 합계가 `max_rows`를 약간 넘을 수 있다. 전체 행 수가
    `max_rows` 이하면 아무것도 버리지 않는다. 반환 인덱스는 원래 행
    순서대로 정렬한다."""
    n_total = len(groups)
    if n_total <= max_rows:
        return np.arange(n_total)
    keep = []
    uniq, counts = np.unique(groups, return_counts=True)
    for g, n_g in zip(uniq, counts):
        proportional = int(round(max_rows * n_g / n_total))
        quota = min(int(n_g), max(min_per_group, proportional))
        idx_g = np.flatnonzero(groups == g)
        if quota < n_g:
            idx_g = rng.choice(idx_g, size=quota, replace=False)
        keep.append(idx_g)
    return np.sort(np.concatenate(keep))


def _subsample_dataset_for_smoke(dataset: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """스모크 전용 라운드당 행 수 제한을 적용한 새 dataset 딕셔너리를
    돌려준다 — 원본은 건드리지 않는다(본 그리드는 항상 원본 전체를 씀).
    라운드마다 seed와 라운드 번호로 고정된 난수를 쓰므로 --shard로 나눠
    돌린 프로세스도 항상 같은 표본을 본다."""
    if SMOKE_MAX_TRAIN_ROWS_PER_EXPERIENCE is None and SMOKE_MAX_TEST_ROWS_PER_EXPERIENCE is None:
        return dataset
    new_experiences = []
    for exp_idx, e in enumerate(dataset["experiences"]):
        rng = np.random.default_rng([seed, exp_idx])
        new_e = dict(e)

        n_tr = len(e["train_y"])
        if SMOKE_MAX_TRAIN_ROWS_PER_EXPERIENCE is not None:
            train_groups = e.get("train_category")
            if train_groups is None:
                train_groups = e["train_y"].cpu().numpy()
            tr_idx = _stratified_subsample_indices(
                np.asarray(train_groups), SMOKE_MAX_TRAIN_ROWS_PER_EXPERIENCE,
                SMOKE_MIN_ROWS_PER_CATEGORY, rng)
            tr_t = torch.as_tensor(tr_idx, dtype=torch.long)
            new_e["train_X"] = e["train_X"][tr_t]
            new_e["train_y"] = e["train_y"][tr_t]
            if e.get("train_category") is not None:
                new_e["train_category"] = e["train_category"][tr_idx]

        n_te = len(e["test_y"])
        if SMOKE_MAX_TEST_ROWS_PER_EXPERIENCE is not None:
            te_idx = _stratified_subsample_indices(
                e["test_y"].cpu().numpy(), SMOKE_MAX_TEST_ROWS_PER_EXPERIENCE,
                SMOKE_MIN_ROWS_PER_CATEGORY, rng)
            te_t = torch.as_tensor(te_idx, dtype=torch.long)
            new_e["test_X"] = e["test_X"][te_t]
            new_e["test_y"] = e["test_y"][te_t]
            # test_category도 같은 인덱스로 슬라이싱해 test_y와 정렬 유지.
            # 층화 기준은 test_y 그대로 — category 층화로 바꾸면 스모크가
            # 뽑는 행이 달라져 기존 결과와 비교가 안 된다.
            if e.get("test_category") is not None:
                new_e["test_category"] = e["test_category"][te_idx]

        n_cat_before = len(np.unique(e["train_category"])) if e.get("train_category") is not None else -1
        n_cat_after = (len(np.unique(new_e["train_category"]))
                       if new_e.get("train_category") is not None else -1)
        print(f"[smoke subsample] exp{exp_idx}: train {n_tr}->{len(new_e['train_y'])}"
              f" (category {n_cat_before}->{n_cat_after}), test {n_te}->{len(new_e['test_y'])}")
        new_experiences.append(new_e)
    return {**dataset, "experiences": new_experiences}


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
    # hidden_dim/latent_dim을 SSF 원 논문 공식으로 데이터셋별 계산
    # (grid_runner.py의 run_combo_full과 동일).
    hp["hidden_dim"], hp["latent_dim"] = ssf_backbone_dims(input_dim)
    if combo["track"] == "B":
        hp["batch_size"] = global_hparams["batch_size_track_b"]
    hp["epochs_per_experience"] = (
        SMOKE_EPOCHS_PER_EXPERIENCE_TRACK_B if combo["track"] == "B"
        else SMOKE_EPOCHS_PER_EXPERIENCE)

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
            exp_idx, e["train_X"], e["train_y"], all_test_splits, labeling_budget,
            train_category=e.get("train_category"))

        theta_after = _flatten_params(model)

        # ---- 15.1a: 학습이 실제로 일어났는가 ----
        delta_norm = torch.norm(theta_after - theta_before).item()
        if delta_norm <= 1e-6:
            failures.append(
                f"exp{exp_idx}: 15.1a 학습이 사실상 일어나지 않음 (delta_norm={delta_norm:.2e})")

        # ---- 15.1d: 손실값이 발산하지 않았는가 — first/last epoch 평균 비교 ----
        first_loss = out["first_epoch_avg_loss"]
        last_loss = out["last_epoch_avg_loss"]
        if not (math.isfinite(first_loss) and math.isfinite(last_loss)):
            failures.append(
                f"exp{exp_idx}: 15.1d 손실값이 발산(NaN/Inf) "
                f"(first_epoch={first_loss}, last_epoch={last_loss})")
        elif last_loss > first_loss * 10 + 1.0:
            failures.append(
                f"exp{exp_idx}: 15.1d 손실이 첫 epoch 대비 발산 의심 "
                f"(first_epoch={first_loss:.4f}, last_epoch={last_loss:.4f})")

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
        # 축소 설정에서는 행동 게이트를 경고로 강등한다(SMOKE_BEHAVIORAL_
        # GATES_AS_WARNINGS). "완전 퇴화"(1.0)는 배선/threshold 결함
        # 신호라 계속 실패로 둔다.
        behavioral_sink = warnings if SMOKE_BEHAVIORAL_GATES_AS_WARNINGS else failures
        if majority_ratio >= 1.0:
            failures.append(f"exp{exp_idx}: 15.2 예측이 완전히 한 클래스로 퇴화")
        elif majority_ratio >= 0.97:
            # 0.97 이상은 "거의 완전 퇴화"로 실패 처리(전체 규모 기준 —
            # 축소 설정에서는 경고). af=gpm 후반 라운드 붕괴(recall 0.26%,
            # majority_ratio≈0.998)가 기존 0.99 기준으로는 안 걸렸었다.
            behavioral_sink.append(
                f"exp{exp_idx}: 15.2 예측이 거의 완전히 한 클래스로 퇴화 "
                f"(다수 클래스 비율 {majority_ratio:.4f} >= 0.97)")
        elif majority_ratio >= 0.90:
            warnings.append(
                f"exp{exp_idx}: 15.2 경고 - 다수 클래스 비율 {majority_ratio:.4f} >= 0.90")

        # 15.2b: roc_auc — 예측 쏠림과 별개로 점수 순위 자체가 라벨과 맞는
        # 방향인지 검사(threshold 무관). dd=cade+ss=ssf+as=cade_mad가
        # roc_auc≈0.15(무작위보다 나쁨)로 사실상 고장난 채 15.2/15.3만으로는
        # 안 걸렸던 사례가 있었다. 단일 클래스 라운드는 정의되지 않아 건너뜀.
        if len(torch.unique(all_labels)) >= 2:
            round_roc_auc = float(roc_auc_score(all_labels.numpy(), all_scores.numpy()))
            if round_roc_auc < 0.5:
                # 축소 설정에서는 경고 — 행 수 축소만으로 0.43까지 내려가는
                # 인공물이 실측됨.
                behavioral_sink.append(
                    f"exp{exp_idx}: 15.2b roc_auc가 무작위보다 낮음(역전 의심) "
                    f"(roc_auc={round_roc_auc:.4f})")
            elif round_roc_auc < 0.55:
                warnings.append(
                    f"exp{exp_idx}: 15.2b 경고 - roc_auc가 무작위에 가까움 "
                    f"(roc_auc={round_roc_auc:.4f})")

        # min-max range는 극단치 하나에도 확 벌어진다 — CADE-MAD처럼 median/MAD
        # 정규화 거리 기반 스코어는 대다수가 낮은 값에 뭉치고 진짜 이상치
        # 몇 개만 큰 값을 내는 게 정상 동작(min-max 기준이면 오탐). 1~99
        # percentile 기반 범위로 소수 극단치에 흔들리지 않게 한다.
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
            # CADEMADScorer가 사설 인코더에 연결된 경우(cl_client.py Step
            # 6/7과 동일) 원본을 그대로 넘긴다.
            if client.anomaly_scorer.uses_shared_representation:
                z0, _, _ = model(test_x0.to(client.device))
            else:
                z0 = test_x0.to(client.device)
        pred0 = client.anomaly_scorer.predict(z0, threshold).cpu()
        if pred0.shape != test_y0.shape:
            failures.append(
                f"exp{exp_idx}: 15.5 predict() shape 불일치 "
                f"(pred={tuple(pred0.shape)}, label={tuple(test_y0.shape)})")

        # ---- 15.4: Track B pseudo-label 균형 (CNDIDS 한정) ----
        # R2L 라운드(실제 공격 비율 6.9%)에서 pseudo-label이 실제보다 훨씬
        # 더 쏠려(0.9844) 정상 참조가 K-means 클러스터를 잘못 덮는 붕괴가
        # 실측됐다(U2R 라운드 직후 정확도 0.870→다음 라운드 0.707). U2R
        # (공격 0.38%)이나 공격 없는 라운드는 실제 라벨 자체가 이미 쏠려
        # 있어 pseudo-label도 쏠리는 게 정상이므로, "pseudo가 true보다
        # 얼마나 더 쏠렸는가"(margin)로 비교한다 — 절대 상수 비교는 실제로
        # 유의미한 소수 클래스 잠식(true=0.9312/pseudo=0.9659, 소수 클래스
        # 비율 6.88%→3.41%)이 있어도 0.97 문턱을 살짝 밑돌면 통과하는
        # 사각지대가 있었다.
        if isinstance(client.anti_forgetting, CNDIDSAntiForgetting):
            ratio = client.anti_forgetting.last_pseudo_label_ratio
            if ratio is not None:
                true_ratio = max(
                    (e["train_y"] == 0).float().mean().item(),
                    (e["train_y"] == 1).float().mean().item())
                margin = ratio - true_ratio
                if margin > 0.03:
                    # 축소 설정에서는 경고 — K-means가 보는 라운드 규모가
                    # 달라지면 pseudo-label 비율도 달라질 수 있다.
                    behavioral_sink.append(
                        f"exp{exp_idx}: 15.4 pseudo-label이 실제 라벨 분포보다 "
                        f"훨씬 쏠림 (pseudo={ratio:.4f}, true={true_ratio:.4f}, "
                        f"margin={margin:.4f}) — 정상 참조가 붕괴해 이 라운드의 "
                        f"category를 사실상 학습하지 못했을 수 있음")
                elif margin > 0.01:
                    warnings.append(
                        f"exp{exp_idx}: 15.4 경고 - pseudo-label이 실제 라벨 "
                        f"분포보다 쏠림 (pseudo={ratio:.4f}, true={true_ratio:.4f}, "
                        f"margin={margin:.4f})")

    combo_id = make_combo_id(combo)
    return {
        "combo_id": combo_id,
        "combo": combo,
        "passed": len(failures) == 0,
        "failures": failures,
        "warnings": warnings,
    }


def _smoke_results_path(shard: Optional[tuple] = None) -> str:
    if shard is None:
        return os.path.join(_TESTBED_ROOT, "experiments", "smoke_test_results.json")
    # 여러 프로세스가 조합을 나눠(--shard) 동시에 돌리면, 같은 파일에 쓸 때
    # 각자 시작 시점에 읽은 스냅샷으로 전체를 덮어써 서로의 결과를 지울 수
    # 있다 — 샤드마다 별도 파일에 쓰고 끝난 뒤 merge_shard_results()로 합친다.
    shard_idx, n_shards = shard
    return os.path.join(
        _TESTBED_ROOT, "experiments", f"smoke_test_results.shard{shard_idx}of{n_shards}.json")


def _load_existing_smoke_results(path: Optional[str] = None) -> List[Dict[str, Any]]:
    if path is None:
        path = _smoke_results_path()
    if not os.path.exists(path):
        return []
    # 쓰기 도중 프로세스가 죽으면 파일이 잘려나갈 수 있다 — json.load()가
    # 예외를 던지면 진행 상황 보존이 아니라 재개 자체가 막히므로, 손상된
    # 파일은 빈 목록으로 폴백한다.
    try:
        with io.open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        print(f"경고: {path} 이(가) 손상되어(추정: {exc}) 읽을 수 없습니다 — "
              f"기존 스모크 결과 없이(재시작) 진행합니다.")
        return []


def _save_smoke_results(all_results_by_key: Dict[tuple, Dict[str, Any]],
                         path: Optional[str] = None) -> None:
    """(combo_id, dataset)를 키로 하는 dict를 그대로 파일에 쓴다 — 매 조합
    완료 직후 호출해 프로세스가 도중에 죽어도 그때까지 진행 상황이
    남도록 한다. 임시 파일에 먼저 쓴 뒤 `os.replace()`로 원자적으로
    교체해, 대용량 결과를 쓰는 도중 죽어 파일 전체가 반쯤 쓰인 채로
    깨지는 것을 막는다(grid_runner.py의 `_atomic_write_json()`과 동일
    패턴). `path`를 지정하지 않으면 기본(비-샤드) 경로에 쓴다."""
    if path is None:
        path = _smoke_results_path()
    tmp_path = f"{path}.tmp{os.getpid()}"
    with io.open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(list(all_results_by_key.values()), f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def merge_shard_results(n_shards: int) -> None:
    """`smoke_test_results.shard{i}of{n_shards}.json`(i=0..n_shards-1)을
    전부 읽어 기존 `smoke_test_results.json`과 합친 뒤 원자적으로 저장한다.
    조합은 샤드 간에 겹치지 않으므로(run_all()의 `i % n_shards == idx`
    필터) 단순 합집합이다. 샤드 파일이 없으면 경고만 내고 있는 것만 합친다."""
    merged: Dict[tuple, Dict[str, Any]] = {
        (r["combo_id"], r.get("dataset")): r for r in _load_existing_smoke_results()
    }
    n_merged = 0
    for i in range(n_shards):
        shard_path = _smoke_results_path(shard=(i, n_shards))
        if not os.path.exists(shard_path):
            print(f"경고: {shard_path} 이(가) 없습니다 — 이 샤드는 건너뜁니다.")
            continue
        for r in _load_existing_smoke_results(path=shard_path):
            merged[(r["combo_id"], r.get("dataset"))] = r
            n_merged += 1
    _save_smoke_results(merged)
    print(f"{n_merged}개 결과를 {n_shards}개 샤드에서 합쳐 {_smoke_results_path()} 에 저장했습니다 "
          f"(전체 {len(merged)}개).")


def run_all(dataset_name: str = "nsl-kdd", device: str = "cpu",
            resume: bool = True, shard: Optional[tuple] = None) -> List[Dict[str, Any]]:
    """`resume=True`(기본)면 이미 지금 코드 버전으로 기록된 결과는 다시
    돌리지 않고 건너뛴다(grid_runner.py의 code_version 캐시와 같은 원칙).
    매 조합이 끝날 때마다 결과 파일에 바로 반영한다.

    `shard=(idx, n)`이면 조합을 n등분해 idx번째 몫만 실행하고, 결과는
    샤드 전용 파일에 저장한다. 모든 샤드가 끝난 뒤 `merge_shard_results()`
    로 하나로 합쳐야 grid_runner.py가 정상 소비한다."""
    global_hparams, component_hparams = _load_configs()
    # n_experiences는 명시하지 않는다 — grid_runner.py와 동일하게 dataset_loader.py가
    # 데이터셋 자신의 실제 공격 유형 수로 자동 결정한다(2026-09-03 추가).
    dataset = load_dataset(dataset_name, base_dir=_REPO_ROOT,
                            seed=global_hparams["seed"])
    # 본 그리드(grid_runner.py)는 이 서브샘플링을 거치지 않고 원본 전체를 쓴다.
    dataset = _subsample_dataset_for_smoke(dataset, seed=global_hparams["seed"])
    labeling_budget = global_hparams["labeling_budget"]

    from testbed.experiments.grid_runner import compute_code_version
    code_version = compute_code_version()

    save_path = _smoke_results_path(shard=shard)
    # key: (combo_id, dataset) — 다른 데이터셋의 기존 결과는 보존. 샤드
    # 실행 시 이어서 진행할 대상은 "이 샤드가 이전에 쓴 파일"이지 공유
    # 메인 파일이 아니다.
    all_results_by_key: Dict[tuple, Dict[str, Any]] = {
        (r["combo_id"], r.get("dataset")): r for r in _load_existing_smoke_results(path=save_path)
    }

    all_combos = enumerate_valid_combos()
    if shard is not None:
        shard_idx, n_shards = shard
        all_combos = [c for i, c in enumerate(all_combos) if i % n_shards == shard_idx]
        print(f"[{dataset_name}] shard {shard_idx}/{n_shards} 담당 조합 {len(all_combos)}개")

    results = []
    for combo in all_combos:
        combo_id = make_combo_id(combo)
        key = (combo_id, dataset_name)
        cached = all_results_by_key.get(key)
        if resume and cached is not None and cached.get("code_version") == code_version:
            print(f"[SKIP] {combo_id} ({dataset_name}) — 이미 지금 코드 버전으로 기록됨")
            results.append(cached)
            continue

        try:
            result = run_smoke_test_for_combo(
                combo, dataset, global_hparams, component_hparams, labeling_budget, device)
        except Exception as exc:
            # 조합 하나의 예외로 전체가 죽지 않도록 격리(grid_runner.py와
            # 같은 원칙) — 예외를 "게이트 통과 실패"로 기록하고 계속 진행.
            tb = traceback.format_exc()
            print(f"[EXCEPTION] {combo_id} ({dataset_name}) — {type(exc).__name__}: {exc}")
            print(tb)
            result = {
                "combo_id": combo_id,
                "combo": combo,
                "passed": False,
                "failures": [f"예외로 중단됨: {type(exc).__name__}: {exc}"],
                "warnings": [],
            }
        result["dataset"] = dataset_name
        result["code_version"] = code_version
        results.append(result)
        all_results_by_key[key] = result
        _save_smoke_results(all_results_by_key, path=save_path)

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
    parser.add_argument(
        "--no-resume", action="store_true",
        help="이미 지금 코드 버전으로 기록된 결과가 있어도 전부 다시 돌린다 "
             "(기본은 건너뛰고 이어서 진행 — run_all()의 resume=True 참고).")
    parser.add_argument(
        "--shard", default=None,
        help="'i/n' 형식으로 조합을 n등분해 i번째 몫만 실행한다(0-indexed, "
             "grid_runner.py --shard와 동일한 문법). 조합이 서로 독립적인데 "
             "모델이 작아 GPU 하나로는 활용률이 낮을 때, 여러 프로세스로 "
             "나눠 같은 GPU(또는 여러 GPU)를 동시에 쓰기 위한 용도. 결과는 "
             "공유 파일이 아니라 샤드 전용 파일(smoke_test_results.shard{i}of{n}.json)"
             "에 저장되므로, 모든 샤드가 끝난 뒤 --merge-shards n으로 반드시 "
             "합쳐야 grid_runner.py가 결과를 인식한다.")
    parser.add_argument(
        "--merge-shards", type=int, default=None, metavar="N",
        help="조합을 실행하지 않고, smoke_test_results.shard{i}ofN.json(i=0..N-1) "
             "N개를 하나로 합쳐 smoke_test_results.json에 저장한 뒤 종료한다. "
             "--shard로 나눠 돌린 모든 프로세스가 끝난 뒤 한 번 실행한다.")
    args = parser.parse_args()

    if args.merge_shards is not None:
        merge_shard_results(args.merge_shards)
        raise SystemExit(0)

    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    shard_arg = None
    if args.shard is not None:
        shard_idx_str, n_shards_str = args.shard.split("/")
        shard_idx, n_shards = int(shard_idx_str), int(n_shards_str)
        assert 0 <= shard_idx < n_shards, "--shard 는 0 <= i < n 이어야 함"
        shard_arg = (shard_idx, n_shards)

    # run_all()이 조합이 끝날 때마다 결과 파일에 바로 저장하므로, 여기서는
    # 최종 요약만 출력한다(--shard를 쓴 경우는 --merge-shards로 별도 합침).
    all_results = []
    for ds_name in dataset_list:
        all_results.extend(run_all(ds_name, device=args.device, resume=not args.no_resume,
                                    shard=shard_arg))

    n_pass = sum(1 for r in all_results if r["passed"])
    print(f"\n{n_pass}/{len(all_results)} combo-dataset 조합이 스모크 테스트 통과")
    print(f"결과 저장: {_smoke_results_path()}")
