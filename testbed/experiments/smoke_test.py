"""스모크 테스트 — PRD 15절. 그리드 실행(Phase 3) 전 필수 게이트.

valid combo 각각에 대해 experience 전체(라운드 수·class-incremental 구조는
본 그리드와 동일)를, 라운드당 행 수만 NSL-KDD 규모로 제한하고 epoch 수를
줄인 설정으로 실행해 15.1~15.5의 정량 기준을 그대로 assert 조건으로
적용한다(아래 SMOKE_* 상수 참고). 하나라도 통과하지 못하면 그 combo는
실패(fail) 처리하고 본 그리드(Phase 3)에서 실행하지 않는다.

**2026-09-02 — 라운드당 행 수 제한(SMOKE_MAX_*_ROWS_PER_EXPERIENCE)**:
CICIDS2018은 중복 제거 후 약 1,208만 행이라 라운드당 train 약 193만/test
약 48만 행 — NSL-KDD(train 2.5만/test 4.5천)의 80~100배다. 스모크가 이
전체를 그대로 돌리면 본 그리드의 복제나 다름없어(통과할 대다수 조합에
대해서도 본 그리드와 비슷한 비용을 먼저 한 번 더 냄) 총 계산량으로는
손해다. 스모크의 진짜 가치는 "본 그리드 전에 버그를 고쳐 전체 재실행을
피하는 것"인데, 아래 2026-08-26 사례의 버그들은 "몇 번째 라운드에 어떤
category가 오는가"에서 나온 것이지 데이터 양에서 나온 게 아니었다 —
그래서 라운드 수와 class-incremental 구조는 그대로 두고 **라운드당 행
수만** category별 최소 개수를 보장하며 서브샘플링한다(사용자 결정).
남는 사각지대는 "CICIDS2018 전체 규모에서만 나타나는 버그"(예전 K-means
클러스터 수 스케일링 문제 같은 종류)인데, 이건 본 그리드 결과에서
드러나고 grid_runner.py의 조합별 실패 격리 덕에 그리드 전체가 죽지는
않는다.

**2026-08-26 발견·수정 — SMOKE_N_EXPERIENCES=2가 만든 구조적 사각지대**:
4개 논문 컴포넌트 전수 재감사(4개 병렬 에이전트)에서, class-incremental
분할이 설계상 희귀·어려운 공격 category(R2L/U2R)와 공격이 아예 없는
라운드를 **항상 뒤쪽 experience**(3~5번째)에 배치하는데, 스모크 테스트는
앞의 2개 experience만 실행해 정확히 그 뒤쪽 라운드들을 한 번도 검사한 적이
없었다는 사실을 확인했다. 그 결과 이미 통과 처리된 조합 중 최소 2개에서
심각한 후반 라운드 붕괴가 방치되어 있었다:
  - `af=gpm`: 3라운드 이후 recall이 0.26%까지 붕괴(전체 5라운드 실행 실측,
    `docs/metric_justification.md` 참고) — 앞 2라운드에서는 건강했다.
  - `af=cndids`: R2L/U2R 라운드에서 pseudo-label 참조가 거의 전부 "정상"으로
    쏠려(0.98~1.0) 그 category를 사실상 학습하지 못함 — 앞 2라운드(DoS/Probe)
    에서는 건강했다.
`SMOKE_N_EXPERIENCES`를 데이터셋의 실제 experience 수 전체로 확장하고,
아래 15.2/15.4 게이트를 이 두 사례를 실제로 잡아낼 수 있도록 강화했다
(15.2의 "거의 완전 퇴화" 등급 신설 + roc_auc 역전 감지, 15.4의 실제 라벨
분포 대비 조건부 실패 처리). 데이터셋별 소요 시간이 기존 대비 늘어난다
(experience 2개→전체, 대략 비례 증가) — 정확성이 속도보다 우선이라는 판단
(사용자 지시).
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

# 2026-08-26 수정 — 전에는 2로 고정해 뒤쪽(희귀 category/공격 없음) 라운드를
# 한 번도 검사하지 못했다(위 모듈 docstring 참고). None이면 데이터셋의
# experience 전체를 검사한다. 값을 넣으면(디버깅 등) 예전처럼 일부만 검사.
SMOKE_N_EXPERIENCES: Optional[int] = None

# 2026-09-01 추가 — epochs_per_experience는 SMOKE_N_EXPERIENCES와 성격이
# 다르다: 라운드 수를 줄이면 "어떤 라운드가 어떤 데이터로 실행되는가"가
# 바뀌어 그 조건에서만 나오는 버그(희귀 category 존재 여부, 공격 없는
# 라운드)를 놓친다 — 위 SMOKE_N_EXPERIENCES=2였을 때 실제로 겪은 문제
# (모듈 docstring 참고)라 라운드 수는 절대 줄이지 않는다(데이터 양은
# 2026-09-02부터 category 최소 개수를 보장하는 방식으로만 제한한다 —
# 아래 SMOKE_MAX_*_ROWS_PER_EXPERIENCE 참고). 반면 epoch 수는 "같은
# 라운드를 얼마나 오래 학습시키는가"일 뿐 — 15.1b/c(optimizer step 횟수/
# label budget), 15.2/15.2b(예측 퇴화/roc_auc), 15.4(CND-IDS pseudo-label)
# 는 전부 그 라운드의 데이터·설정만으로 결정되고 epoch 수와 무관하다
# (15.4는 라운드 시작 시 한 번 도는 K-means 결과라 학습 루프보다도 먼저
# 정해짐). 유일하게 epoch 수에 걸리는 15.1d(loss 발산 체크, 첫/마지막
# epoch 비교)도 발산은 보통 초반 몇 epoch 안에 이미 드러나는 문제라 실전
# epoch 수(200/20)를 다 안 돌려도 잡힌다. 그래서 라운드는 그대로 유지한
# 채 epoch 수만 줄여 스모크 테스트가 사실상 본 그리드(grid_runner.py)와
# 동일한 비용으로 조합마다 두 번 학습시키던 낭비를 없앤다(사용자 지시).
SMOKE_EPOCHS_PER_EXPERIENCE = 10
SMOKE_EPOCHS_PER_EXPERIENCE_TRACK_B = 5

# 2026-09-02 추가 — 라운드당 행 수 상한(모듈 docstring "2026-09-02" 절
# 참고). 값은 NSL-KDD의 라운드 규모(train 1.3만~5.9만, test 4.5천)에
# 맞췄다 — 15.2/15.2b/15.4 게이트가 실제로 버그를 잡아낸 것이 바로 이
# 규모에서였고, CND-IDS `cluster_fit_sample_size`(1만)·`max_normal_ref`
# (5천)·CADE `max_category_ref`(500) 같은 캡/축출 경로도 이 규모에서
# 전부 발동하므로, 그 코드 경로들이 스모크에서 계속 실행된다. 상한보다
# 작은 라운드(NSL-KDD 뒤쪽 라운드 등)는 그대로 둔다(no-op).
#
# category별 최소 개수(SMOKE_MIN_ROWS_PER_CATEGORY): 비율대로만 뽑으면
# CICIDS2018의 희귀 공격(수십~수백 행짜리 category)이 라운드에서 통째로
# 사라져 "그 라운드에 그 category가 있다"는 조건 자체가 없어진다 —
# 2026-08-26 사례(`af=cndids`가 R2L/U2R에서 붕괴)가 정확히 희귀 category
# 라운드에서 나온 문제라, 각 category는 최소 이 개수(원래 그보다 적으면
# 전부)를 반드시 남긴다. 50은 NSL-KDD U2R 라운드(1.35만 중 52건, 0.38%)와
# 같은 "이미 검증된 희귀 regime"에 해당하는 값이다. test는 `test_y`
# (이진)로만 층화한다(experience 딕셔너리에 test category가 없음).
# None이면 해당 상한을 적용하지 않는다(디버깅용).
SMOKE_MAX_TRAIN_ROWS_PER_EXPERIENCE: Optional[int] = 20_000
SMOKE_MAX_TEST_ROWS_PER_EXPERIENCE: Optional[int] = 5_000
SMOKE_MIN_ROWS_PER_CATEGORY = 50

# 2026-09-02 추가 — 축소 설정(행 수 상한·epoch 축소)에서는 "행동" 게이트를
# 실패가 아니라 경고로만 기록한다. 실측 근거(NSL-KDD,
# `A_dd=cade_ss=ssf_mm=ssf_af=lwf_ssf_as=cade_mad`, 4조건 대조):
#   - 행 전체 + epoch 10 : exp0 통과, exp2 15.2 실패(0.9962)
#   - 행 축소 + epoch 10 : exp0/exp1 15.2·15.2b 실패(0.9851/roc 0.4309, 0.9865/0.4570)
#   - 행 축소 + epoch 200: exp0/exp1 실패, exp0 수치가 epoch 10과 완전히 동일
# 즉 이 조합의 점수는 CADE 사설 인코더(as=cade_mad가 dd=cade에 연결됨)에서
# 나와 메인 모델 epoch와 무관하고, 라운드 행 수를 줄이면(라벨 예산 10%로
# 선택되는 표본이 5,940→2,000개) 인코더 학습량이 그만큼 줄어 exp0에서
# roc_auc가 역전된다 — 행 수 전체에서는 같은 라운드가 통과한다. 축소
# 설정의 행동 게이트 실패는 이렇게 "축소 자체가 만든 인공물"일 수 있고,
# 실패로 처리하면 grid_runner.py가 그 조합을 본 그리드에서 **제외**해
# 버려 조합 커버리지(이 테스트베드의 핵심 목적)를 해친다. 스모크의 역할은
# "코드가 제대로 연결돼 실행되는가"이고 조합의 실제 성능/퇴화 여부는 본
# 그리드 결과가 판정한다(사용자 결정) — 그래서 축소 설정에서는
#   실패 유지: 15.1a~d(학습 발생/발산/step 수/라벨 예산), 15.2의
#              "완전 퇴화"(예측이 상수 함수) 및 "상수 점수", 15.3(threshold
#              범위), 15.5(shape) — 배선/수치 결함이라 데이터 규모와 무관.
#   경고로 강등: 15.2의 0.97 등급, 15.2b(roc_auc<0.5), 15.4(pseudo-label
#              쏠림) — 모델이 약하거나 퇴화했다는 신호라 축소 규모의
#              영향을 받으며, 본 그리드 결과에서 그대로 드러난다.
# 행동 게이트를 실패로 되돌리려면(예: 상한을 전부 None으로 두고 실전
# epoch로 돌리는 전체 규모 스모크) 이 값을 False로 바꾼다.
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
    보장되므로 합계가 `max_rows`를 약간 넘을 수 있다(그룹 수 × 최소
    개수만큼이 상한이라 무시할 수준). 전체 행 수가 `max_rows` 이하면
    아무것도 버리지 않는다. 반환 인덱스는 원래 행 순서대로 정렬한다."""
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
    """스모크 전용 라운드당 행 수 제한(SMOKE_MAX_*_ROWS_PER_EXPERIENCE)을
    적용한 **새** dataset 딕셔너리를 돌려준다 — 원본은 건드리지 않는다
    (원본은 `load_dataset()` 캐시에서 온 객체일 수 있고, 본 그리드는 항상
    원본 전체를 쓴다). 라운드마다 `seed`와 라운드 번호로 고정된 난수를
    쓰므로 실행/샤드 프로세스가 달라도 항상 같은 표본이 뽑힌다(--shard로
    나눠 돌린 프로세스들이 서로 다른 데이터를 보면 안 되므로 중요)."""
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
    # 2026-08-11: hidden_dim/latent_dim을 SSF 원 논문 공식으로 이 데이터셋의
    # input_dim에 맞춰 그때그때 계산한다(configs/global_hparams.yaml 주석
    # 참고) — grid_runner.py의 run_combo_full과 동일한 근거.
    hp["hidden_dim"], hp["latent_dim"] = ssf_backbone_dims(input_dim)
    # Track B(CND-IDS) 배치크기 오버라이드 — grid_runner.py의 run_combo_full과
    # 동일한 근거(configs/global_hparams.yaml 주석 참고). epochs_per_experience는
    # 아래에서 스모크 전용 값으로 다시 덮어쓴다.
    if combo["track"] == "B":
        hp["batch_size"] = global_hparams["batch_size_track_b"]
    # 2026-09-01 추가 — 위 SMOKE_EPOCHS_PER_EXPERIENCE(_TRACK_B) 절 참고.
    # global_hparams의 실전 epoch 수(200/20) 대신 스모크 전용 축소 값을 쓴다.
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

        # ---- 15.1d: 손실값이 발산하지 않았는가 (2026-08-14 추가) ----
        # 15.1a/15.1b는 "학습 루프가 설계대로 실행됐는가"만 검증하고 "손실이
        # 실제로 줄어드는 방향으로 갔는가"는 검증하지 않았다(구조 전수 감사에서
        # 발견) — 15.2/15.3이 스코어 퇴화/NaN을 간접적으로 잡아내긴 하지만
        # 진단 메시지가 근본 원인(손실 발산)을 가리킨다는 보장이 없었다.
        # first/last epoch 평균 손실을 직접 비교해 명시적으로 검증한다.
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
        # 2026-09-02 — 축소 설정에서는 행동 게이트를 경고로 강등한다
        # (SMOKE_BEHAVIORAL_GATES_AS_WARNINGS 절 참고). "완전 퇴화"(1.0)는
        # 예측이 상수 함수라는 배선/threshold 결함 신호라 계속 실패로 둔다.
        behavioral_sink = warnings if SMOKE_BEHAVIORAL_GATES_AS_WARNINGS else failures
        if majority_ratio >= 1.0:
            failures.append(f"exp{exp_idx}: 15.2 예측이 완전히 한 클래스로 퇴화")
        elif majority_ratio >= 0.97:
            # 2026-08-26 추가 — af=gpm 후반 라운드 붕괴(recall 0.26%,
            # majority_ratio≈0.998)가 기존 0.99 미만은 경고에 그치는 느슨한
            # 기준 때문에 실패로 걸러지지 못했다(SMOKE_N_EXPERIENCES가 2라서
            # 애초에 그 라운드 자체를 검사하지 않은 것과는 별개의 문제 —
            # 이제 검사는 하되 기준도 강화한다). 0.97 이상은 "거의 완전
            # 퇴화"로 보고 실패 처리한다(전체 규모 스모크 기준 — 축소
            # 설정에서는 경고, 위 2026-09-02 절 참고).
            behavioral_sink.append(
                f"exp{exp_idx}: 15.2 예측이 거의 완전히 한 클래스로 퇴화 "
                f"(다수 클래스 비율 {majority_ratio:.4f} >= 0.97)")
        elif majority_ratio >= 0.90:
            warnings.append(
                f"exp{exp_idx}: 15.2 경고 - 다수 클래스 비율 {majority_ratio:.4f} >= 0.90")

        # 2026-08-26 추가 — dd=cade+ss=ssf+as=cade_mad 콤보가 roc_auc≈0.15
        # (무작위보다 나쁨, 점수와 라벨의 상관관계가 뒤집힘)로 사실상 고장난
        # 채 스모크 테스트를 통과했던 사례(4개 병렬 에이전트 재감사에서 발견)
        # — 위 15.2/15.3 게이트는 "예측이 한쪽으로 쏠렸는가"만 보고 "점수
        # 순위 자체가 라벨과 맞는 방향인가"는 전혀 검사하지 않아 이런 경우를
        # 못 잡는다. roc_auc는 threshold와 무관하게 순위 품질만 재므로
        # 별도로 검사한다. 두 클래스가 다 있어야 정의되므로(단일 클래스
        # 라운드, 예: exp4 공격 0건은 건너뛴다).
        if len(torch.unique(all_labels)) >= 2:
            round_roc_auc = float(roc_auc_score(all_labels.numpy(), all_scores.numpy()))
            if round_roc_auc < 0.5:
                # 축소 설정에서는 경고(2026-09-02, SMOKE_BEHAVIORAL_GATES_AS_
                # WARNINGS 절 참고 — 행 수 축소만으로 0.43까지 내려가는
                # 인공물이 실측됐다).
                behavioral_sink.append(
                    f"exp{exp_idx}: 15.2b roc_auc가 무작위보다 낮음(역전 의심) "
                    f"(roc_auc={round_roc_auc:.4f})")
            elif round_roc_auc < 0.55:
                warnings.append(
                    f"exp{exp_idx}: 15.2b 경고 - roc_auc가 무작위에 가까움 "
                    f"(roc_auc={round_roc_auc:.4f})")

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
            # 2026-08-14: CADEMADScorer가 사설 인코더에 연결된 경우
            # (cl_client.py Step 6/7과 동일한 이유) 원본을 그대로 넘긴다.
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
        # 2026-08-26 수정 — 기존엔 pseudo-label 쏠림 자체만 보고 항상 경고로만
        # 처리했는데, 실측(4개 병렬 에이전트 재감사)으로 R2L 라운드(실제
        # 공격 비율 6.9%)에서 pseudo-label이 실제 라벨 분포보다 훨씬 더
        # 쏠려(0.9844) "정상" 참조가 거의 모든 K-means 클러스터를 잘못
        # 덮어버리는 실제 붕괴를 확인했다(U2R 라운드 학습 직후 정확도
        # 0.870→다음 라운드 0.707로 하락, 원인 추적 완료). 반면 U2R
        # 라운드(실제 공격 비율 0.38%)나 공격이 아예 없는 라운드는 실제
        # 라벨 자체가 이미 극단적으로 쏠려 있어 pseudo-label도 쏠리는 게
        # 당연하다 — 이 경우까지 실패 처리하면 "원래 그런 라운드"를 오탐
        # 하게 된다. 그래서 실제 라벨 쏠림 정도와 비교해, pseudo-label이
        # 실제보다 뚜렷이 더 쏠렸을 때만(실제 라벨 자체는 0.97 미만으로
        # 아직 유의미하게 섞여 있는데 pseudo-label은 0.97 이상으로 쏠림)
        # 실패로 승격한다 — 실제 라벨 자체가 이미 0.97 이상으로 쏠린
        # 라운드(U2R, 공격 0건)는 기존처럼 경고에 그친다.
        if isinstance(client.anti_forgetting, CNDIDSAntiForgetting):
            ratio = client.anti_forgetting.last_pseudo_label_ratio
            if ratio is not None:
                true_ratio = max(
                    (e["train_y"] == 0).float().mean().item(),
                    (e["train_y"] == 1).float().mean().item())
                # 2026-08-26 재수정(재감사에서 발견) — 절대 상수 두 개(0.97/0.95)를
                # 각각 독립적으로 비교하는 기존 방식은, 실제로 유의미한 소수
                # 클래스 잠식(예: true=0.9312/pseudo=0.9659 — 소수 클래스 비율이
                # 6.88%→3.41%로 절반 가까이 줄어듦)이 있어도 절대값이 0.97
                # 문턱을 살짝 밑돌면 그냥 통과해버리는 사각지대가 있었다.
                # "pseudo가 true보다 얼마나 더 쏠렸는가"(margin)로 직접 비교한다.
                margin = ratio - true_ratio
                if margin > 0.03:
                    # 축소 설정에서는 경고(2026-09-02, SMOKE_BEHAVIORAL_GATES_
                    # AS_WARNINGS 절 참고 — K-means가 보는 라운드 규모가
                    # 달라지면 pseudo-label 비율도 달라질 수 있다).
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
    # 2026-09-01 추가 — 여러 프로세스가 조합을 나눠(--shard) 동시에 돌릴 때,
    # 전부 같은 smoke_test_results.json에 쓰면 경쟁 조건이 생긴다: 각 프로세스가
    # `run_all()` 시작 시점에 그 파일을 한 번 읽어 메모리에 들고 있다가 매 조합마다
    # "메모리 전체"를 다시 써서 교체하므로(_save_smoke_results), 프로세스 A가 조합을
    # 다 끝내고 저장한 뒤 프로세스 B가 (A 시작 이전에 읽어둔) 낡은 스냅샷으로
    # 저장하면 A가 이미 써둔 결과가 그대로 덮어써져 사라진다 — grid_runner.py의
    # 원자적 쓰기(2026-09-01 추가)가 막는 것은 "한 프로세스가 쓰다가 죽는 것"뿐,
    # "여러 프로세스가 같은 파일을 동시에 쓰는 것"은 별개 문제라 안 막아준다.
    # 샤드마다 별도 파일에 쓰게 해 애초에 같은 파일을 동시에 쓸 일이 없게 한다 —
    # 다 끝난 뒤 merge_shard_results()로 하나로 합친다.
    shard_idx, n_shards = shard
    return os.path.join(
        _TESTBED_ROOT, "experiments", f"smoke_test_results.shard{shard_idx}of{n_shards}.json")


def _load_existing_smoke_results(path: Optional[str] = None) -> List[Dict[str, Any]]:
    if path is None:
        path = _smoke_results_path()
    if not os.path.exists(path):
        return []
    # 2026-09-01 추가 — 저장이 원자적이지 않던 시절(아래 `_save_smoke_results`
    # "2026-09-01" 절 참고)에 프로세스가 쓰기 도중 죽었다면 이 파일이 잘려나간
    # 채로 남아있을 수 있다. 그러면 `json.load()`가 예외를 던지는데, 이 함수를
    # 부르는 `run_all()`은 매 조합마다 이 결과를 이어서 쓰므로, 여기서 예외가
    # 잡히지 않으면 "진행 상황을 보존"하려던 원래 목적과 반대로 재개 자체가
    # 막혀버린다 — 손상된 파일이면 빈 목록(처음부터 다시)으로 안전하게 폴백한다.
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
    완료 직후 호출해, 프로세스가 도중에 죽어도(2026-08-26 실제로 겪음 —
    93개 조합×5라운드 전체 스모크가 세션 중단으로 한 줄도 못 쓰고 날아간
    적이 있다) 그때까지의 진행 상황이 남도록 한다.

    2026-09-01 수정 — 기존엔 목적지 파일에 `json.dump()`를 직접 실행해서,
    "프로세스가 도중에 죽어도 안전"이 절반만 맞았다: 죽는 시점이 매 조합
    "사이"라면 안전하지만, 이 `json.dump()` 자체가 실행되는 도중(대용량
    결과가 쌓인 뒤반이라 파일 쓰기에 시간이 걸리는 경우)에 죽으면 오히려
    이 파일 전체가 반쯤 쓰인 채로 깨진다 — 그러면 다음 실행이 이어서 돌 때
    `_load_existing_smoke_results()`가 예외를 던져(위 참고) 그동안 쌓아둔
    모든 진행 상황을 통째로 잃는, 이 함수가 막으려던 것과 정반대의 결과가
    난다. 임시 파일에 먼저 다 쓴 뒤 `os.replace()`로 원자적으로 교체해
    이 위험을 없앤다(grid_runner.py의 `_atomic_write_json()`과 동일한 패턴).
    `path`를 지정하지 않으면 기본(비-샤드) 경로에 쓴다 — 샤드 실행 시에는
    `run_all()`이 샤드 전용 경로를 명시적으로 넘긴다."""
    if path is None:
        path = _smoke_results_path()
    tmp_path = f"{path}.tmp{os.getpid()}"
    with io.open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(list(all_results_by_key.values()), f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def merge_shard_results(n_shards: int) -> None:
    """`smoke_test_results.shard{i}of{n_shards}.json`(i=0..n_shards-1)을
    전부 읽어 기존 `smoke_test_results.json`과 합친 뒤 원자적으로 저장한다.

    조합은 샤드 간에 겹치지 않게 나뉘므로(run_all()의 `i % n_shards == idx`
    필터, grid_runner.py의 --shard와 동일한 방식) 같은 키가 서로 다른 값으로
    충돌할 일은 없다 — 단순 합집합이다. 샤드 파일이 하나라도 없으면(해당
    샤드를 아직 안 돌렸거나 실패) 경고만 내고 있는 것만 합친다."""
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
    돌리지 않고 건너뛴다 — grid_runner.py의 code_version 캐시와 같은
    원칙(2026-08-26 추가, 아래 참고). 매 조합이 끝날 때마다 결과 파일에
    바로 반영한다(중간에 죽어도 그때까지 진행 상황을 보존).

    `shard=(idx, n)`이면 조합을 n등분해 idx번째 몫만 실행하고, 결과는
    공유 파일이 아니라 샤드 전용 파일에 저장한다(_smoke_results_path()의
    "2026-09-01" 절 참고 — 여러 프로세스가 같은 파일에 동시에 쓰면 경쟁
    조건으로 서로의 결과를 덮어쓸 수 있다). 모든 샤드가 끝난 뒤
    `merge_shard_results()`로 하나로 합쳐야 grid_runner.py가 정상 소비한다."""
    global_hparams, component_hparams = _load_configs()
    # 두 데이터셋 모두 동일한 프로토콜(dataset_loader.py 기본값) — 원본
    # train/test 파일 분리 유지, 각 파일 내부는 고정 seed로 섞은 뒤 분할.
    dataset = load_dataset(dataset_name, base_dir=_REPO_ROOT,
                            n_experiences=global_hparams["n_experiences"],
                            seed=global_hparams["seed"])
    # 2026-09-02 추가 — 스모크 전용 라운드당 행 수 제한(모듈 docstring
    # "2026-09-02" 절 참고). 본 그리드(grid_runner.py)는 이 함수를 거치지
    # 않고 원본 전체를 쓴다.
    dataset = _subsample_dataset_for_smoke(dataset, seed=global_hparams["seed"])
    labeling_budget = global_hparams["labeling_budget"]

    from testbed.experiments.grid_runner import compute_code_version
    code_version = compute_code_version()

    save_path = _smoke_results_path(shard=shard)
    # 다른 데이터셋의 기존 결과는 그대로 보존하고, 이번 dataset_name에
    # 대해서만 이어서 진행한다 — key: (combo_id, dataset). 샤드 실행 시
    # 이어서 진행할 대상은 "이 샤드가 이전에 쓴 파일"이지 공유 메인
    # 파일이 아니다(공유 파일을 읽어서 그대로 다시 쓰면 다른 샤드의
    # 결과까지 이 샤드 전용 파일로 복제되어 merge 시 중복/혼선이 생긴다).
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
            # 2026-09-01 추가 — grid_runner.py의 조합별 예외 격리와 같은 이유:
            # 93개 조합 x 5라운드 전체를 도는 이 함수가 조합 하나의 예외로
            # 전부 죽으면(예: 특정 조합에서만 나오는 shape 불일치, NaN 등)
            # 나머지 조합의 스모크 결과를 하나도 못 얻는다 — 예외를 "그 조합은
            # 게이트를 통과하지 못한 것"으로 기록하고 계속 진행한다. 이렇게
            # 하면 grid_runner.py 쪽에서도 이 조합이 자동으로 제외되므로(passed
            # =False), 크래시하는 조합이 전체 그리드까지 막는 상황도 막는다.
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

    # 2026-08-26 수정 — run_all()이 이제 조합이 끝날 때마다 결과 파일에
    # 바로 저장한다(중간에 프로세스가 죽어도 그때까지 진행 상황이 남도록 —
    # 93개 조합×5라운드 전체 스모크가 세션 중단으로 한 줄도 못 쓰고 날아간
    # 적이 실제로 있었다). 여기서는 최종 요약만 출력한다 — 별도로 다시
    # 파일을 합쳐 쓰지 않는다(이미 매 조합마다 저장되어 있음. --shard를 쓴
    # 경우는 예외 — 위 --merge-shards 설명 참고).
    all_results = []
    for ds_name in dataset_list:
        all_results.extend(run_all(ds_name, device=args.device, resume=not args.no_resume,
                                    shard=shard_arg))

    n_pass = sum(1 for r in all_results if r["passed"])
    print(f"\n{n_pass}/{len(all_results)} combo-dataset 조합이 스모크 테스트 통과")
    print(f"결과 저장: {_smoke_results_path()}")
