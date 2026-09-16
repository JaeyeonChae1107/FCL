"""이 벤치마크의 핵심 전제("슬롯 값만 바꿔 통제된 비교를 한다", "코드가
바뀌면 캐시를 못 믿는다")를 지키는 교차-슬롯(cross-cutting) 불변조건을
코드로 고정한 회귀 테스트 — 2026-09-14 감사에서 발견된 문제들이 사람이
다시 읽지 않아도 재발 즉시 잡히도록 하기 위함.

각 테스트가 지키려는 구체적 사건:
- `cl_client.py:118`이 `**merged_component_kwargs`를 memory_manager에만
  안 넘겨 SPIDERMemoryManager/SSFMemoryManager/CNDIDSMemoryManager의
  `seed`가 조용히 버려지던 버그.
- `torch.random.fork_rng()` 격리가 있어야 할 컴포넌트에서 빠지거나 깨져
  `dd=cade`처럼 "신호는 안 쓰이지만 전역 RNG를 오염시켜 결과가 달라지는"
  현상이 재발하는 것.
- `compatibility.py`의 active/inert 분류가 컴포넌트 재작성 이후에도
  실제 코드 동작과 계속 일치하는지.
- `load_smoke_passed_combo_ids()`가 code_version 불일치를 무시하고 낡은
  판정으로 그리드를 실행하는 것.
- CADE/SPIDER/GPM처럼 "원 논문에 라벨예산 개념이 없는" 컴포넌트가
  `consumes_full_round_data` 면제를 빠짐없이 받는지.
- `dd=cade`+`as=cade_mad`일 때 af/ss/mm이 정확도 지표에 아무 영향을 못
  주는 알려진 성질(`design_decisions.md` 3절)이 향후 변경으로 조용히
  깨지거나(반대로 계속 유지되는지) 추적.
"""

import numpy as np
import torch

from testbed.base import FCLAutoEncoder
from testbed.common.compatibility import (
    NO_CL_BASELINE_COMBO,
    TRACK_A_DD_ACTIVE_SS_MM,
    enumerate_valid_combos,
    validate_combo,
)
from testbed.pipeline import CLClient
from testbed.pipeline.component_registry import REGISTRY


def _make_client(combo, seed=42, global_hparams_overrides=None):
    global_hparams = {
        "optimizer": "adam", "learning_rate": 1e-3, "batch_size": 16,
        "epochs_per_experience": 1, "hidden_dim": 16, "latent_dim": 4,
        "_input_dim": 8, "seed": seed,
    }
    if global_hparams_overrides:
        global_hparams.update(global_hparams_overrides)
    model = FCLAutoEncoder(input_dim=8, hidden_dim=16, latent_dim=4)
    return CLClient(model, combo, global_hparams, component_hparams={})


def _synthetic_experiences(seed, n=2, batch_n=40, dim=8):
    """실제 데이터셋 없이 재현 가능한 합성 라운드를 만든다."""
    gen = torch.Generator().manual_seed(seed)
    all_test_splits = []
    exps = []
    for _ in range(n):
        X = torch.randn(batch_n, dim, generator=gen)
        y = (torch.rand(batch_n, generator=gen) < 0.4).long()
        train_n = int(batch_n * 0.8)
        exps.append((X[:train_n], y[:train_n]))
        all_test_splits.append((X[train_n:], y[train_n:]))
    return exps, all_test_splits


# ---------------------------------------------------------------------------
# 1. 5개 슬롯 전부 컴포넌트 kwargs(특히 seed)를 실제로 받는가
#    (cl_client.py:118의 memory_manager 누락 재발 방지)
# ---------------------------------------------------------------------------

_KWARGS_PROBE_COMBOS = [
    # (Track A) CADEDriftDetector/RandomSelector/SPIDERMemoryManager/GPMAntiForgetting
    {"track": "A", "drift_detector": "cade", "sample_selector": "random",
     "memory_manager": "spider", "anti_forgetting": "gpm", "anomaly_scorer": "none"},
    # (Track A) SSFSampleSelector/SSFMemoryManager
    {"track": "A", "drift_detector": "none", "sample_selector": "ssf",
     "memory_manager": "ssf", "anti_forgetting": "none", "anomaly_scorer": "none"},
    # (Track B) CNDIDSMemoryManager/CNDIDSAntiForgetting
    {"track": "B", "drift_detector": "none", "sample_selector": "random",
     "memory_manager": "cndids", "anti_forgetting": "cndids", "anomaly_scorer": "pca"},
]


_SEED_PROBE_SLOTS = (
    "drift_detector", "sample_selector", "memory_manager", "anti_forgetting",
    "anomaly_scorer",
)


def test_every_seed_aware_component_receives_configured_seed():
    """`build()`로 생성된 각 슬롯 인스턴스 중 `_seed` 속성을 선언한 것들이
    전부 CLClient에 넘긴 seed를 실제로 받았는지 확인한다 — 슬롯 하나라도
    kwargs 전달 경로가 빠지면(과거 memory_manager처럼) 여기서 즉시 실패한다.

    `anomaly_scorer`도 포함한다(2026-09-14 Round-2 감사로 발견된 커버리지
    공백 수정) — 지금은 PCAScorer/CADEMADScorer 둘 다 `seed`를 안 받아
    `touched_any`에 기여하지 않지만, 향후 seed를 쓰는 anomaly_scorer가
    추가되면 이 5번째 슬롯도 다른 4개와 똑같이 kwargs 누락을 잡아야 한다."""
    seed = 913427
    touched_any = False
    for combo in _KWARGS_PROBE_COMBOS:
        client = _make_client(combo, seed=seed)
        for slot_name in _SEED_PROBE_SLOTS:
            component = getattr(client, slot_name)
            if hasattr(component, "_seed"):
                touched_any = True
                assert component._seed == seed, (
                    f"{combo['track']}/{slot_name}={combo[slot_name]!r} 컴포넌트가 "
                    f"seed={seed}를 못 받음(실제: {component._seed!r}) — "
                    "component_registry.build() 호출부에 **merged_component_kwargs가 "
                    "빠졌을 수 있다(cl_client.py 각 build() 호출 확인).")
    assert touched_any, "seed-aware 컴포넌트를 하나도 못 건드림 — 테스트 콤보를 점검하라"


def test_different_seeds_produce_different_effective_seed():
    """seed 값 자체가 바뀌면 컴포넌트의 `_seed`도 실제로 달라지는지(같은 값에
    묶여있지 않은지) 확인 — kwargs가 "전달은 되지만 무시된다"는 종류의
    회귀까지 잡는다."""
    for combo in _KWARGS_PROBE_COMBOS:
        client_a = _make_client(combo, seed=111)
        client_b = _make_client(combo, seed=222)
        for slot_name in _SEED_PROBE_SLOTS:
            comp_a = getattr(client_a, slot_name)
            comp_b = getattr(client_b, slot_name)
            if hasattr(comp_a, "_seed"):
                assert comp_a._seed != comp_b._seed


# ---------------------------------------------------------------------------
# 2. 전역 RNG 오염 격리 — 신호를 안 쓰는 슬롯 값은 dd=none과 완전히 같아야
#    하고, 실제로 값을 소비하는 조합은 여전히 달라야 한다.
# ---------------------------------------------------------------------------

def _run_and_collect(combo, seed):
    # 모델 가중치 초기화도 재현 가능해야 두 실행(dd=none/dd=cade)의 차이가
    # 오직 drift_detector 탓인지 검증할 수 있다 — 실제 grid_runner.py도
    # torch.manual_seed(seed)를 model 생성 "이전"에 호출한다(같은 순서).
    torch.manual_seed(seed)
    client = _make_client(combo, seed=seed)
    exps, all_test_splits = _synthetic_experiences(seed=seed, n=2)
    labeling_budget = {"mode": "fixed_ratio", "value": 0.5}
    torch.manual_seed(seed)
    outs = []
    for i, (train_X, train_y) in enumerate(exps):
        out = client.run_experience(i, train_X, train_y, all_test_splits, labeling_budget)
        outs.append((out["avg_train_loss"], out["threshold"],
                      [s.tolist() for s in out["eval_scores"]]))
    return outs


def test_dd_cade_identical_to_dd_none_when_signal_unused():
    """`compatibility.py`가 "비활성"으로 분류한 (ss,mm)+as=none 조합에서는
    `dd=cade`가 신호(get_drift_score/detect)를 아무도 안 읽으므로 `dd=none`과
    완전히 동일한 결과를 내야 한다(RNG 오염이 fork_rng로 격리돼 있다는
    전제) — 이 항등성이 깨지면 RNG 격리가 어딘가 새고 있다는 뜻이다."""
    base = {"track": "A", "sample_selector": "random", "memory_manager": "none",
            "anti_forgetting": "none", "anomaly_scorer": "none"}
    assert ("random", "none") not in TRACK_A_DD_ACTIVE_SS_MM, (
        "이 테스트가 가정하는 비활성 쌍이 compatibility.py에서 활성으로 "
        "바뀌었다 — 테스트 콤보를 다시 골라야 한다.")
    out_none = _run_and_collect({**base, "drift_detector": "none"}, seed=42)
    out_cade = _run_and_collect({**base, "drift_detector": "cade"}, seed=42)
    assert out_none == out_cade, (
        "dd=cade가 dd=none과 달라짐 — CADEDriftDetector의 전역 RNG 격리"
        "(torch.random.fork_rng())가 깨졌을 가능성이 높다.")


def test_dd_cade_differs_from_dd_none_when_wired_to_cade_mad():
    """반대로 `as=cade_mad`와 짝지어지면 `dd=cade`는 CADEMADScorer의 사설
    encoder 연결(uses_shared_representation=False)로 인해 `dd=none`과 여전히
    달라야 한다 — 위 항등성 테스트를 "일부러 결과를 어디서나 강제로 같게
    만드는" 방식으로 잘못 고쳐버리면 이 테스트가 잡는다."""
    base = {"track": "A", "sample_selector": "random", "memory_manager": "none",
            "anti_forgetting": "none", "anomaly_scorer": "cade_mad"}
    out_none = _run_and_collect({**base, "drift_detector": "none"}, seed=42)
    out_cade = _run_and_collect({**base, "drift_detector": "cade"}, seed=42)
    assert out_none != out_cade


# ---------------------------------------------------------------------------
# 3. enumerate_valid_combos() 자체 정합성
# ---------------------------------------------------------------------------

def test_enumerate_valid_combos_shape_and_membership():
    combos = enumerate_valid_combos()
    track_a = [c for c in combos if c["track"] == "A"]
    track_b = [c for c in combos if c["track"] == "B"]
    assert len(track_a) == 72
    assert len(track_b) == 6
    assert len(combos) == 78
    assert NO_CL_BASELINE_COMBO in combos
    for c in combos:
        validate_combo(c)  # 예외 없이 통과해야 함


def test_registry_has_every_slot_value_enumerate_uses():
    """enumerate_valid_combos()가 생성하는 값들이 실제로 component_registry에
    전부 등록돼 있는지 — 그리드 값과 레지스트리가 따로 노는 걸 방지."""
    combos = enumerate_valid_combos()
    for slot in ("drift_detector", "sample_selector", "memory_manager",
                 "anti_forgetting", "anomaly_scorer"):
        used_values = {c[slot] for c in combos}
        assert used_values <= set(REGISTRY[slot]), (
            f"{slot} 슬롯에 레지스트리에 없는 값이 쓰임: "
            f"{used_values - set(REGISTRY[slot])}")


# ---------------------------------------------------------------------------
# 4. 스모크 캐시가 code_version 불일치를 실제로 거부하는가
# ---------------------------------------------------------------------------

def test_load_smoke_passed_combo_ids_rejects_stale_code_version(tmp_path):
    from testbed.experiments.grid_runner import load_smoke_passed_combo_ids

    path = tmp_path / "smoke_test_results.json"
    import json
    path.write_text(json.dumps([
        {"combo_id": "A_x", "dataset": "nsl-kdd", "passed": True, "code_version": "old"},
        {"combo_id": "A_y", "dataset": "nsl-kdd", "passed": True, "code_version": "current"},
    ]), encoding="utf-8")

    result = load_smoke_passed_combo_ids(str(path), code_version="current")
    assert result["nsl-kdd"] == {"A_y"}, (
        "code_version이 다른 통과 판정이 걸러지지 않음 — "
        "load_smoke_passed_combo_ids()가 낡은 그리드 실행을 허용하고 있다.")

    # code_version을 안 넘기면(기존 호출부 호환) 필터링을 안 하는 게 맞다 —
    # 이 인자는 opt-in이지 강제가 아니다.
    result_no_filter = load_smoke_passed_combo_ids(str(path))
    assert result_no_filter["nsl-kdd"] == {"A_x", "A_y"}


# ---------------------------------------------------------------------------
# 5. "원 논문에 라벨예산 개념이 없는" 컴포넌트는 consumes_full_round_data를
#    빠짐없이 선언해야 한다(GPM이 빠졌던 사건 재발 방지) — 이 목록 자체는
#    사람이 새 컴포넌트를 추가할 때 수동으로 검토해 갱신해야 한다.
# ---------------------------------------------------------------------------

def test_no_label_budget_components_declare_full_round_data_exemption():
    from testbed.components.cade.cade_drift_detector import CADEDriftDetector
    from testbed.components.cade.cade_anomaly_scorer import CADEMADScorer
    from testbed.components.spider_gpm.spider_memory_manager import SPIDERMemoryManager
    from testbed.components.spider_gpm.gpm_anti_forgetting import GPMAntiForgetting

    # 이 4개는 전부 "원 논문에 라벨예산 개념이 없다"는 같은 근거로
    # consumes_full_round_data=True를 선언해야 한다(각 파일 docstring
    # "라벨예산 면제" 절 참고). 새 컴포넌트를 추가할 때 같은 근거가
    # 적용된다면 이 목록에도 추가할 것.
    for cls in (CADEDriftDetector, CADEMADScorer, SPIDERMemoryManager, GPMAntiForgetting):
        assert getattr(cls, "consumes_full_round_data", False) is True, (
            f"{cls.__name__}이 라벨예산 면제(consumes_full_round_data)를 "
            "선언하지 않음 — 원 논문에 라벨예산 개념이 없다면 다른 3개 "
            "컴포넌트와 같은 근거로 면제해야 한다(2026-09-14 GPM 누락 사건 참고).")


def test_gpm_receives_full_round_data_hook():
    """`consumes_full_round_data=True`를 선언한 컴포넌트 중 실제로 라운드당
    1회 별도 데이터를 받는 GPM이 그 훅(`set_full_round_data`)을 실제로
    구현하는지, `cl_client.py`가 실제로 호출하는지 통합 확인."""
    combo = {"track": "A", "drift_detector": "none", "sample_selector": "random",
             "memory_manager": "none", "anti_forgetting": "gpm", "anomaly_scorer": "none"}
    client = _make_client(combo, seed=42)
    assert hasattr(client.anti_forgetting, "set_full_round_data")

    calls = []
    original = client.anti_forgetting.set_full_round_data

    def spy(new_data, new_labels):
        calls.append((len(new_data), len(new_labels)))
        return original(new_data, new_labels)

    client.anti_forgetting.set_full_round_data = spy
    exps, all_test_splits = _synthetic_experiences(seed=42, n=2, batch_n=40)
    labeling_budget = {"mode": "fixed_ratio", "value": 0.1}  # 라벨예산 10% << 라운드 전체
    for i, (train_X, train_y) in enumerate(exps):
        client.run_experience(i, train_X, train_y, all_test_splits, labeling_budget)

    assert len(calls) == 2, "set_full_round_data가 라운드마다 호출되지 않음"
    for n_data, n_labels in calls:
        # 라벨예산 10%인 32행짜리 라운드에서 전체(32)를 받아야 한다 —
        # 선택된 서브셋(3~4개)만 받으면 라벨예산 면제가 다시 깨진 것.
        assert n_data == 32 and n_labels == 32, (
            f"GPM이 라운드 전체(32) 대신 라벨예산 서브셋({n_data})만 받음 — "
            "consumes_full_round_data 배선이 다시 끊겼을 수 있다.")


def test_gpm_class_balance_branch_actually_exercised_for_mm_spider():
    """2026-09-14 재검토로 발견된 커버리지 공백 수정 — 기존
    `test_gpm_receives_full_round_data_hook`은 `mm=none`만 써서
    `set_full_round_data()`의 `mm=spider` 전용 클래스 균형 분기
    (`_pending_count_by_class`)를 한 번도 실행하지 않았다. `mm=spider`로
    실제 클래스 균형 누적이 일어나는지 직접 확인한다."""
    combo = {"track": "A", "drift_detector": "none", "sample_selector": "random",
             "memory_manager": "spider", "anti_forgetting": "gpm", "anomaly_scorer": "none"}
    client = _make_client(combo, seed=42)
    gpm = client.anti_forgetting

    # 극단적으로 불균형한 라운드 전체(label=1이 90%)를 직접 넣어, 균형
    # 분기가 실제로 두 클래스를 목표치까지 채우려 시도하는지 확인한다.
    new_data = torch.randn(100, 8)
    new_labels = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(90, dtype=torch.long)])
    gpm._self_training_pool = torch.randn(5, 8)  # mm=spider가 채운 것처럼 설정
    gpm.activation_sample_size = 20  # 클래스당 목표 10개

    gpm.set_full_round_data(new_data, new_labels)

    assert gpm._pending_count_by_class, "클래스 균형 분기가 전혀 실행되지 않음"
    # label=0은 라운드에 10개뿐이라 목표(10)를 딱 채우고, label=1은 90개
    # 중 목표(10)만큼만 채워야 한다 — "그냥 다 넣는" 게 아니라 실제로
    # 클래스별 캡이 걸리는지 확인.
    assert gpm._pending_count_by_class.get(0) == 10
    assert gpm._pending_count_by_class.get(1) == 10


def test_mm_spider_actually_populates_gpm_self_training_pool_via_real_wiring():
    """2026-09-14 Round-6 감사(테스트 스위트 자체의 공허함 점검)로 발견된
    커버리지 공백 수정 — 위 `test_gpm_class_balance_branch_actually_
    exercised_for_mm_spider`는 `gpm._self_training_pool`을 직접 대입해
    "mm=spider가 채운 것처럼" 흉내만 냈을 뿐, `cl_client.py` Step 3의 실제
    배선(`combo["memory_manager"] == "spider"`일 때만
    `anti_forgetting.set_self_training_pool()`을 호출하는 `if` 분기)을 전혀
    거치지 않았다 — 그 분기 자체가 망가지는 회귀(예: 문자열 오타, 조건
    반전)는 어떤 기존 테스트도 못 잡았다. `run_experience()`를 실제로 호출해
    이 배선이 진짜로 작동하는지 직접 확인한다."""
    combo = {"track": "A", "drift_detector": "none", "sample_selector": "random",
             "memory_manager": "spider", "anti_forgetting": "gpm", "anomaly_scorer": "none"}
    client = _make_client(combo, seed=42)
    gpm = client.anti_forgetting
    assert gpm._self_training_pool is None, "라운드 실행 전인데 이미 채워져 있음 — 테스트 전제가 깨짐"

    calls = []
    original = gpm.set_self_training_pool

    def spy(pool):
        calls.append(len(pool))
        return original(pool)

    gpm.set_self_training_pool = spy

    exps, all_test_splits = _synthetic_experiences(seed=42, n=1, batch_n=40)
    labeling_budget = {"mode": "fixed_ratio", "value": 0.5}  # 라벨예산 50% -> 나머지 50%가 self-training pool
    train_X, train_y = exps[0]
    client.run_experience(0, train_X, train_y, all_test_splits, labeling_budget)

    assert len(calls) == 1, (
        f"mm=spider인데 set_self_training_pool()이 라운드당 정확히 1회 호출되지 "
        f"않음(실제 {len(calls)}회) — cl_client.py Step 3의 "
        "combo['memory_manager']=='spider' 배선이 깨졌을 수 있다.")
    # 라벨예산 50%(32행 라운드 중 절반 근처가 선택)라 잔여(self-training pool)도
    # 0은 아니어야 한다 — 정확한 개수보다 "실제로 무언가 채워졌는가"가 핵심.
    assert calls[0] > 0, "set_self_training_pool()에 빈 풀이 전달됨"
    assert gpm._self_training_pool is not None and len(gpm._self_training_pool) > 0, (
        "run_experience() 이후에도 gpm._self_training_pool이 비어있음 — "
        "실제 배선을 거쳐도 풀이 채워지지 않는다.")


# ---------------------------------------------------------------------------
# 6. dd=cade + as=cade_mad일 때 af/ss/mm이 채점 결과에 영향을 못 주는
#    성질(design_decisions.md 3절) — CADEMADScorer가 사설 encoder로 연결돼
#    (uses_shared_representation=False) 공유 backbone의 z를 완전히 우회하기
#    때문. 2026-09-14 실측: 4개 콤보(ssf/ssf/gpm, ssf/spider/none,
#    random/none/gpm, random/none/lwf_ssf, [ss/mm/af] 순)가 f1/bwt/pr_auc
#    소수점 10자리까지 완전히 일치했다. 이 테스트는 그 채점 입력 자체
#    (threshold, eval_scores)가 af/ss/mm과 무관하게 항등인지 고정한다 —
#    f1/bwt/pr_auc는 이 값들의 순수 함수라 여기서 항등이면 지표도 항등이다.
# ---------------------------------------------------------------------------

def _run_and_collect_with_category(combo, seed):
    """`_run_and_collect`와 동일하지만 `train_category`를 함께 넘긴다 —
    CADE의 `consumes_full_round_data` 면제 경로(`cl_client.py:347-348`)는
    `fit_with_category`가 실제로 호출될 때만(`train_category is not None`)
    타므로, 이걸 안 넘기면 CADE가 조용히 `selected_data`(라벨예산 서브셋,
    sample_selector에 따라 달라짐) 경로로 폴백해 이 불변조건 자체가
    성립하지 않는 다른 코드 경로를 테스트하게 된다(2026-09-14 이 테스트
    작성 중 실제로 이 폴백에 걸려 최초 버전이 실패했다 — 그 발견 자체가
    이 헬퍼가 존재하는 이유)."""
    torch.manual_seed(seed)
    client = _make_client(combo, seed=seed)
    exps, all_test_splits = _synthetic_experiences(seed=seed, n=2)
    labeling_budget = {"mode": "fixed_ratio", "value": 0.5}
    torch.manual_seed(seed)
    outs = []
    for i, (train_X, train_y) in enumerate(exps):
        # 이진 라벨만 있는 합성 데이터라 category는 "normal"/"attack1"
        # 둘뿐이지만, fit_with_category가 실제로 호출되는 경로를 타는 게
        # 목적이라 다중클래스 다양성 자체는 이 테스트에 필요 없다.
        train_category = np.array(
            ["normal" if int(lbl) == 0 else "attack1" for lbl in train_y])
        out = client.run_experience(
            i, train_X, train_y, all_test_splits, labeling_budget,
            train_category=train_category)
        outs.append((out["avg_train_loss"], out["threshold"],
                      [s.tolist() for s in out["eval_scores"]]))
    return outs


_CADE_MAD_INVARIANCE_COMBOS = [
    {"track": "A", "drift_detector": "cade", "sample_selector": "ssf",
     "memory_manager": "ssf", "anti_forgetting": "gpm", "anomaly_scorer": "cade_mad"},
    {"track": "A", "drift_detector": "cade", "sample_selector": "ssf",
     "memory_manager": "spider", "anti_forgetting": "none", "anomaly_scorer": "cade_mad"},
    {"track": "A", "drift_detector": "cade", "sample_selector": "random",
     "memory_manager": "none", "anti_forgetting": "gpm", "anomaly_scorer": "cade_mad"},
    {"track": "A", "drift_detector": "cade", "sample_selector": "random",
     "memory_manager": "none", "anti_forgetting": "lwf_ssf", "anomaly_scorer": "cade_mad"},
]


def test_dd_cade_as_cade_mad_scoring_independent_of_af_ss_mm():
    """`dd=cade`+`as=cade_mad`가 고정된 채 af/ss/mm만 바뀌는 4개 콤보에서
    threshold와 eval_scores(즉 f1/bwt/pr_auc를 결정하는 전부)가 완전히
    동일해야 한다. 이 항등성이 깨지면 CADEMADScorer의 사설 encoder 연결
    (set_private_encoder → uses_shared_representation=False)이 더 이상
    공유 backbone을 완전히 우회하지 않게 됐다는 뜻 — 향후 슬롯 기여도
    분석 코드가 이 18/72 콤보(전체 Track A의 25%)를 여전히 특별 취급해야
    하는지 재검토해야 한다(design_decisions.md 3절)."""
    seed = 42
    reference = None
    for combo in _CADE_MAD_INVARIANCE_COMBOS:
        out = _run_and_collect_with_category(combo, seed=seed)
        thresholds_and_scores = [(threshold, scores) for (_loss, threshold, scores) in out]
        if reference is None:
            reference = thresholds_and_scores
        else:
            assert thresholds_and_scores == reference, (
                f"dd=cade+as=cade_mad인데 af={combo['anti_forgetting']!r}/"
                f"ss={combo['sample_selector']!r}/mm={combo['memory_manager']!r} 콤보의 "
                "채점 결과(threshold/eval_scores)가 기준 콤보와 달라짐 — "
                "CADEMADScorer의 사설 encoder 연결이 깨졌을 수 있다.")


def test_dd_cade_as_cade_mad_training_loss_still_varies_by_af():
    """반대로 avg_train_loss(공유 backbone 학습 자체)는 af가 다르면 여전히
    달라야 한다 — 위 테스트를 "아예 실행을 스킵하게" 잘못 고쳐 항등성을
    만드는 회귀를 잡는다. 공유 모델은 여전히 학습되고(자원 지표에 반영),
    다만 그 출력이 채점에 안 쓰일 뿐이다(design_decisions.md 3절).

    af=gpm이 아니라 af=lwf_ssf를 골랐다 — GPM(Saha et al.)은 손실 값 자체를
    안 바꾸고 오직 `project_gradients()`로 역전파 이후 그래디언트만
    투영한다(그래디언트 투영이지 손실항 추가가 아님), 그래서 af=gpm과
    af=none은 avg_train_loss가 항등인 게 정상이라 이 테스트의 반례가
    못 된다(작성 중 실제로 이걸로 시도했다가 거짓양성으로 실패해 발견).
    반면 af=lwf_ssf는 `base/models.py`에 문서화된 대로 x_hat에 InfoNCE
    재구성-대조 손실을 명시적으로 더하므로 loss 값 자체가 달라야 한다."""
    seed = 42
    combo_lwf = _CADE_MAD_INVARIANCE_COMBOS[3]  # af=lwf_ssf
    combo_none_af = {**combo_lwf, "anti_forgetting": "none"}
    out_lwf = _run_and_collect_with_category(combo_lwf, seed=seed)
    out_none = _run_and_collect_with_category(combo_none_af, seed=seed)
    losses_lwf = [loss for (loss, _t, _s) in out_lwf]
    losses_none = [loss for (loss, _t, _s) in out_none]
    assert losses_lwf != losses_none, (
        "af=lwf_ssf/af=none 간 avg_train_loss가 안 바뀜 — InfoNCE 손실항이 "
        "적용 안 되거나 공유 backbone 학습 자체가 생략됐을 수 있다(단순히 "
        "채점 항등성과는 별개 문제).")


# ---------------------------------------------------------------------------
# 7. Step 4(학습, replay_batch는 "이전" 버퍼)가 Step 5(메모리 갱신)보다
#    먼저 일어나야 한다 — 순서가 바뀌면 이번 라운드 자기 자신을 리플레이하게
#    된다(cl_client.py 모듈 docstring "Step 4→5 순서" 절). 2026-09-14
#    Round-2 감사에서 "이 순서는 함수 본문의 문(statement) 순서로만
#    강제되고 회귀 테스트가 없다"는 지적을 받아 추가.
# ---------------------------------------------------------------------------

def test_memory_update_happens_after_training_not_before():
    """`memory_manager.update()`(Step 5) 호출이 그 라운드의 `get_replay_batch()`
    (Step 4, 미니배치마다 호출) 호출들보다 항상 나중에 일어나는지 직접
    감시한다. 순서가 뒤집히면(Step 5를 Step 4보다 먼저 하면) 이번 라운드
    자기 자신의 selected_data를 이번 라운드 학습에 리플레이하게 되는
    치명적 버그가 SPIDER/CNDIDSMemoryManager에서 발생한다(모듈 docstring
    참고) — 지금까지는 코드 안의 문 순서로만 지켜지고 있어 이 테스트가
    없으면 실수로 두 블록을 바꿔도 조용히 통과한다."""
    combo = {"track": "A", "drift_detector": "none", "sample_selector": "ssf",
             "memory_manager": "ssf", "anti_forgetting": "none", "anomaly_scorer": "none"}
    client = _make_client(combo, seed=42)

    call_log = []
    original_replay = client.memory_manager.get_replay_batch
    original_update = client.memory_manager.update

    def replay_spy(*args, **kwargs):
        call_log.append("replay")
        return original_replay(*args, **kwargs)

    def update_spy(*args, **kwargs):
        call_log.append("update")
        return original_update(*args, **kwargs)

    client.memory_manager.get_replay_batch = replay_spy
    client.memory_manager.update = update_spy

    exps, all_test_splits = _synthetic_experiences(seed=42, n=1, batch_n=40)
    labeling_budget = {"mode": "fixed_ratio", "value": 0.5}
    train_X, train_y = exps[0]
    client.run_experience(0, train_X, train_y, all_test_splits, labeling_budget)

    assert "update" in call_log, "memory_manager.update()가 한 번도 호출되지 않음"
    update_idx = call_log.index("update")
    assert call_log.count("update") == 1, (
        f"update()가 라운드당 1회가 아니라 {call_log.count('update')}회 호출됨")
    assert all(c == "replay" for c in call_log[:update_idx]), (
        f"update() 이전에 replay가 아닌 호출이 섞여 있음: {call_log[:update_idx]}")
    assert "replay" not in call_log[update_idx + 1:], (
        f"update() 다음에 다시 replay가 호출됨 — 순서 가정이 이 테스트가 "
        f"기대한 것과 다르다: {call_log}")


# ---------------------------------------------------------------------------
# 8. CNDIDSAntiForgetting의 정상 참조 저수지 캡(reservoir cap) — 2026-09-14
#    Round-2 감사(두 개의 독립된 서브에이전트)가 공통으로 발견한, 이
#    파일에서 유일하게 fork_rng() 격리가 빠져 있던 torch.randperm을 고친
#    직후 추가한 회귀 테스트. 실제 grid_runner.py/smoke_test.py 실행에서는
#    held_out_normal_reference가 항상 채워져 있어 이 분기 자체가 지금까지
#    한 번도 실행된 적이 없지만(design_decisions.md 참고), 그렇다고 격리가
#    깨져도 되는 건 아니다 — 이 분기가 실행되는 경로(단위 테스트, held_out
#    참조가 없는 호출) 자체를 여기서 직접 만들어 검증한다.
# ---------------------------------------------------------------------------

def test_cndids_reservoir_cap_isolated_from_global_rng():
    """`on_experience_start()`가 저수지 캡(무작위 서브샘플)을 실행하는지
    여부(`max_normal_ref`로 강제)와 무관하게, 그 호출 이후의 전역 torch RNG
    상태가 완전히 동일해야 한다 — `torch.random.fork_rng()`로 캡 로직의
    randperm이 실제로 격리돼 있다는 뜻이다. 격리가 깨지면(과거처럼 맨
    torch.randperm을 쓰면) 캡이 발동하는 라운드부터 이후 호출되는
    torch.rand()의 값이 달라진다."""
    from testbed.components.cndids.cndids_anti_forgetting import CNDIDSAntiForgetting

    normal_subset = torch.randn(30, 4)

    def _run_and_draw(max_normal_ref):
        afo = CNDIDSAntiForgetting(seed=42, max_normal_ref=max_normal_ref)
        torch.manual_seed(123)
        afo.on_experience_start(normal_subset, normal_subset)
        return torch.rand(5)

    # max_normal_ref=5 < 30이라 캡(randperm) 분기를 강제로 밟는다.
    # max_normal_ref=1000 >= 30이라 캡 분기를 아예 건너뛴다.
    out_capped = _run_and_draw(max_normal_ref=5)
    out_uncapped = _run_and_draw(max_normal_ref=1000)
    assert torch.equal(out_capped, out_uncapped), (
        "저수지 캡 randperm이 격리 안 된 전역 RNG를 소비함 — "
        "cndids_anti_forgetting.py의 fork_rng() 블록이 깨졌을 수 있다.")


def test_cndids_reservoir_cap_deterministic_and_seed_sensitive():
    """캡 결과 자체는(전역 RNG와 무관하게) seed로 재현 가능해야 하고,
    seed가 다르면 실제로 달라져야 한다 — derived_seed 배선이 죽어 항상
    같은 파생 시드만 쓰는 회귀를 잡는다."""
    from testbed.components.cndids.cndids_anti_forgetting import CNDIDSAntiForgetting

    normal_subset = torch.randn(30, 4)

    def _capped_pool(seed):
        afo = CNDIDSAntiForgetting(seed=seed, max_normal_ref=5)
        afo.on_experience_start(normal_subset, normal_subset)
        return afo._normal_ref_pool

    pool_a1 = _capped_pool(seed=42)
    pool_a2 = _capped_pool(seed=42)
    pool_b = _capped_pool(seed=999)
    assert torch.equal(pool_a1, pool_a2), "같은 seed인데 캡 결과가 재현되지 않음"
    assert not torch.equal(pool_a1, pool_b), "seed를 바꿨는데 캡 결과가 그대로임"
