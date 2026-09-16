"""CLClient — PRD 13절의 8단계 실행 흐름.

  1. 새 데이터 도착
  2. Drift 감지 (buf_ref = 이전 experience까지의 버퍼)
  3. 샘플 선택과 라벨 예산 확정 (select → slice → drift_detector.fit)
  4. 모델 학습 (epochs_per_experience, replay_batch는 "이전" 버퍼에서,
     selected_data만 사용)
  5. 메모리 갱신 (학습 이후, selected_data)
  6. Anomaly Scorer 재보정 (refit_on_update, s_ref 계산·캐싱)
  7. 평가 (experience 0..T-1 test split, threshold는
     anomaly_scorer.threshold_needs_labels로 분기 — base/anomaly_scorer.py)
  8. 다음 라운드 준비 (anti_forgetting.on_task_end)

Step 4→5 순서는 리플레이 버퍼 계약(BaseMemoryManager.get_replay_batch)이
강제한다 — 5를 4보다 앞에 두면 이번 라운드 selected_data가 먼저 버퍼에
들어가 같은 라운드 안에서 자기 자신을 리플레이하게 된다(SPIDER/
CNDIDSMemoryManager에서 치명적). SSF 원문(`ssf.py:236-291`)은 대표 표본
재선택을 먼저 하고 갱신된 세트로 학습하는 반대 순서다 — SSF는 "메모리"와
"이번 라운드 학습 데이터"가 하나의 객체라 이 구분 자체가 없다
(docs/metric_justification.md "SSF 대표 표본 재선택" 절).

`dd=cade`+`as=cade_mad` 조합에서 `__init__`이 `CADEMADScorer.
set_private_encoder()`로 둘을 연결한다(`uses_shared_representation` 플래그로
Step 6/7 인코딩 경로 분기 — `components/cade/cade_anomaly_scorer.py`).
NSL-KDD 순정 CADE 콤보 A/B: f1 0.6482→0.7898, bwt
-0.1403→-0.0810.

`run_experience()`는 `data/dataset_loader.py`의 `train_category`를
`selected_idx`로 슬라이싱해, `drift_detector.fit_with_category()`가 있으면
(CADEDriftDetector 전용) family 단위 pairing/centroid를 만든다 — 없으면
이진 `fit()`으로 폴백(`components/cade/cade_drift_detector.py`).

**전역 RNG 오염(2026-09-14 재검토로 발견, 수정)**: `torch.manual_seed(seed)`는
콤보당 1회만(`grid_runner.py`) 호출되고 라운드마다 재시드하지 않는다.
`_compute_ssf_masks()`가 호출하는 `optimize_ssf_masks()`(`ssf_masks.py`)
내부의 `M_c`/`M_t` 초기화(`torch.rand`)가 전역 RNG를 소비하는데, 이 호출은
`ss=ssf`/`mm=ssf`일 때만 발동한다(Step 2) — 즉 이 슬롯 값 하나 때문에
그 뒤 Step 4의 메인 공유 모델 셔플이 보는 난수 시퀀스 자체가 달라진다.
같은 문제가 `CADEDriftDetector`(사설 encoder 초기화·매 라운드 학습)와
GPM/SPIDER의 `randperm`에도 있어(각 파일 docstring 참고) 국지적 버그가
아니라 설계 패턴 자체의 문제였다 — "슬롯 값만 바꿔 통제된 비교를 한다"는
이 벤치마크의 핵심 전제를 직접 위협하는 문제라 우선 수정했다.
`torch.random.fork_rng()`로 각 사설 랜덤성 소비를 격리해, 그 컴포넌트
자신의 결과는 재현 가능하게 유지하면서 바깥 RNG 스트림은 그 컴포넌트가
있든 없든 동일하게 만든다. 파생 시드는 `testbed/common/rng_utils.py`의
`derived_seed(seed, tag, counter)`(해시 기반, 2026-09-14부로 정수 오프셋
방식에서 교체 — `rng_utils.py` docstring 참고)를 쓴다.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from testbed.base.models import BaseCLModel
from testbed.common.rng_utils import derived_seed
from testbed.components.ssf.ssf_masks import SSFMaskContext, optimize_ssf_masks
from testbed.pipeline.component_registry import build


class CLClient:
    def __init__(self, model: BaseCLModel, combo: Dict[str, Any],
                 global_hparams: Dict[str, Any],
                 component_hparams: Optional[Dict[str, Dict[str, Any]]] = None,
                 device: str = "cpu",
                 held_out_normal_reference: Optional[List[torch.Tensor]] = None):
        """
        Args:
            model: BaseCLModel(FCLAutoEncoder) 인스턴스.
            combo: {'track': 'A'|'B', 'drift_detector':, 'sample_selector':,
                    'memory_manager':, 'anti_forgetting':, 'anomaly_scorer':}
                   (common.compatibility.enumerate_valid_combos()가 만든 형식).
            global_hparams: configs/global_hparams.yaml 로드 결과.
            component_hparams: {'cade': {...}, 'gpm': {...}, 'cndids': {...},
                                 'ssf': {...}} — configs/component_hparams/*.yaml.
            device: torch device 문자열.
            held_out_normal_reference: CND-IDS N_c(원문 `init_normal`) 이식 —
                `data/dataset_loader.py`의 `held_out_normal_reference` 키
                (라운드별 List[Tensor], 없으면 None). `consumes_held_out_
                reference=True`를 선언한 anomaly_scorer(PCAScorer)만
                소비한다 — Step 6 참고.
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.combo = combo
        self.track = combo["track"]
        component_hparams = component_hparams or {}

        input_dim = global_hparams.get("_input_dim")  # dataset_loader가 채워 넣음
        hidden_dim = global_hparams["hidden_dim"]
        latent_dim = global_hparams["latent_dim"]

        # component_hparams/*.yaml 전체 병합. build()가 생성자 시그니처로
        # 필터링(component_registry.py). input_dim/hidden_dim/latent_dim은
        # global_hparams가 항상 우선(10.1절).
        merged_component_kwargs: Dict[str, Any] = {}
        for hp in component_hparams.values():
            merged_component_kwargs.update(hp)
        merged_component_kwargs.update(
            input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
            batch_size=global_hparams["batch_size"],
            seed=global_hparams.get("seed", 42))
        # seed(2026-09-14 추가) — CADEDriftDetector(사설 encoder 초기화·매
        # 라운드 학습)/SPIDERMemoryManager/GPMAntiForgetting처럼 전역 RNG를
        # 소비하는 컴포넌트가 자기 전용 torch.random.fork_rng() 파생 시드를
        # 만드는 데 쓴다(각 컴포넌트 docstring 참고) — 이 값 자체를 그대로
        # 쓰는 게 아니라 컴포넌트별 고정 오프셋을 더해 파생한다.
        # batch_size는 CADEDriftDetector 사설 encoder 미니배치 학습에만 쓰임
        # . 다른 컴포넌트는 생성자에 이 인자가 없어 무시됨.

        self.drift_detector = build(
            "drift_detector", combo["drift_detector"], **merged_component_kwargs)
        # CADEDriftDetector만 자기 소유 nn.Module(사설 ContrastiveAutoEncoder)이
        # 있어 명시적 device 이동 필요(components/cade/cade_drift_detector.py).
        if hasattr(self.drift_detector, "to"):
            self.drift_detector.to(self.device)
        self.sample_selector = build(
            "sample_selector", combo["sample_selector"], **merged_component_kwargs)
        # 2026-09-14 재검토로 발견·수정 — 다른 4개 슬롯은 전부
        # **merged_component_kwargs를 넘기는데 이 줄만 빠져 있었다. 그
        # 결과 SPIDERMemoryManager/SSFMemoryManager/CNDIDSMemoryManager가
        # 추가한 `seed` 파라미터(전역 RNG 오염 격리용)가 실제로는 전달되지
        # 않고 항상 생성자 기본값(42)에 고정돼 있었다 — 지금(seed=42)
        # 단일 시드 실행에는 하드코딩 기본값과 우연히 같아 결과에 영향이
        # 없었지만, 멀티시드 재실행 시 이 세 클래스의 무작위성(버퍼
        # 교체·리플레이 샘플링)만 조용히 seed=42에 머물렀을 것이다.
        self.memory_manager = build(
            "memory_manager", combo["memory_manager"], **merged_component_kwargs)
        self.anti_forgetting = build(
            "anti_forgetting", combo["anti_forgetting"], **merged_component_kwargs)
        self.anomaly_scorer = build(
            "anomaly_scorer", combo["anomaly_scorer"], **merged_component_kwargs)
        # NoAnomalyScorer 전용 훅 — z 대신 model 참조가 필요
        # (pipeline/common_baselines.py).
        if hasattr(self.anomaly_scorer, "set_model"):
            self.anomaly_scorer.set_model(self.model)
        # SSFMemoryManager 전용 훅(Item 1+2) — 드리프트 시 버퍼 부족분을
        # 현재 모델로 pseudo-label해서 채울 때 필요(ssf_memory_manager.py
        # 참고). 누락되면 그 backfill 경로(드리프트+버퍼가 목표치 아래로
        # 줄어드는 라운드)에서만 크래시하는데, NSL-KDD에서는 안 걸리고
        # UNSW-NB15의 `dd=ssf/ss=ssf/mm=ssf` 콤보에서 실제로 걸린 걸
        # 스모크 테스트로 확인했다(2026-09-11).
        if hasattr(self.memory_manager, "set_model"):
            self.memory_manager.set_model(self.model)
        # CADEMADScorer 전용 훅 — dd=cade와 함께 선택됐을 때만 연결
        # (components/cade/cade_anomaly_scorer.py).
        if hasattr(self.anomaly_scorer, "set_private_encoder") and hasattr(self.drift_detector, "min_anomaly_score"):
            self.anomaly_scorer.set_private_encoder(self.drift_detector)

        optimizer_name = global_hparams.get("optimizer", "adam")
        lr = global_hparams["learning_rate"]
        if optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        self.batch_size = global_hparams["batch_size"]
        self.epochs_per_experience = global_hparams["epochs_per_experience"]

        # 정상 참조 표본은 별도 고정 세트가 아니라 이번 라운드 selected_data
        # 중 label=0인 것(docs/metric_justification.md).
        # 정상 라벨 선택 샘플이 없는 라운드는 마지막 성공한 self._s_ref로 폴백.
        self._s_ref: Optional[torch.Tensor] = None
        self._round = 0
        self._held_out_normal_reference = held_out_normal_reference
        # 전역 RNG 오염 격리(2026-09-14, _compute_ssf_masks() 참고)에 쓸
        # 이 인스턴스 전용 시드.
        self._seed = global_hparams.get("seed", 42)

    def forward_batched(self, x: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """model(x)를 self.batch_size 단위로 나눠 실행하고 (z, x_hat, logit)을
        이어붙여 반환한다. experience당 행 수가 큰 경우 한 번에 통과시키면
        CUDA OOM이 날 수 있어(GPU 서버 실측) 배치 단위로 나눈다. 호출부가
        이미 `torch.no_grad()`로 감싸므로 그래디언트 관리는 하지 않는다."""
        n = len(x)
        if n <= self.batch_size:
            return self.model(x)
        zs, x_hats, logits = [], [], []
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            z, x_hat, logit = self.model(x[start:end])
            zs.append(z)
            x_hats.append(x_hat)
            logits.append(logit)
        return torch.cat(zs, dim=0), torch.cat(x_hats, dim=0), torch.cat(logits, dim=0)

    def _compute_ssf_masks(self, new_data: torch.Tensor, buf_data: Optional[torch.Tensor],
                            new_logit: Optional[torch.Tensor] = None,
                            buf_logit: Optional[torch.Tensor] = None) -> SSFMaskContext:
        """SSF 원문(`ssf_masks.py`가 이식한 utils.py:96-190)을 라운드당
        1회 계산 — drift 여부와 무관하게 계산하고 sample_selector/
        memory_manager가 공유해서 쓴다(Item 1+2). control_res/treatment_res는
        현재 모델의 sigmoid(classifier(x_hat)) 점수(Item 10 적용 후) —
        Step 2가 `dd=ssf`처럼 이미 이 값을 계산해뒀으면(`new_logit`/
        `buf_logit` 인자) 재사용해 중복 forward를 피한다."""
        self.model.eval()
        with torch.no_grad():
            if new_logit is None:
                _, _, new_logit = self.forward_batched(new_data)
            treatment_res = torch.sigmoid(new_logit).squeeze(-1).detach()
            if buf_data is not None and len(buf_data) > 0:
                if buf_logit is None:
                    _, _, buf_logit = self.forward_batched(buf_data.to(self.device))
                control_res = torch.sigmoid(buf_logit).squeeze(-1).detach()
            else:
                control_res = treatment_res.new_zeros(0)
        # 전역 RNG 오염 격리(2026-09-14, 모듈 docstring "전역 RNG 오염" 참고)
        # — optimize_ssf_masks() 내부(_optimize_old_mask/_optimize_new_mask)의
        # M_c/M_t 초기화(torch.rand)가 전역 RNG를 소비해, ss=ssf/mm=ssf 여부에
        # 따라 그 뒤 메인 모델 학습 루프가 보는 셔플 시퀀스가 달라지던 문제.
        # 라운드마다(self._round) 다른 파생 시드로 격리한다.
        with torch.random.fork_rng():
            torch.manual_seed(derived_seed(self._seed, "ssf_masks", self._round))
            M_c_bin, M_t_bin, M_t_cont, M_c_cont = optimize_ssf_masks(
                control_res, treatment_res, self.device)
        return SSFMaskContext(
            buf_data=buf_data, control_res=control_res, M_c_bin=M_c_bin,
            new_data=new_data, treatment_res=treatment_res, M_t_bin=M_t_bin,
            M_t_cont=M_t_cont, M_c_cont=M_c_cont)

    @staticmethod
    def _label_budget_int(n: int, labeling_budget: Dict[str, Any]) -> int:
        if labeling_budget["mode"] == "fixed_count":
            return int(labeling_budget["value"])
        return round(labeling_budget["value"] * n)

    def run_experience(self, exp_idx: int,
                        train_data: torch.Tensor, train_labels: torch.Tensor,
                        all_test_splits: List[Tuple[torch.Tensor, torch.Tensor]],
                        labeling_budget: Dict[str, Any],
                        train_category: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """experience exp_idx에 대해 13절 8단계를 순서대로 수행한다.

        Args:
            exp_idx: 현재 experience 인덱스 (0-based).
            train_data, train_labels: experience exp_idx의 train split (X_i, y_i).
            all_test_splits: experience 0..T-1 전부의 (test_X, test_y) 리스트
                              (길이 T, 데이터 로딩 시 고정된 그대로 — 9.2절).
            labeling_budget: {'mode': 'fixed_count'|'fixed_ratio', 'value': ...}
            train_category: experience exp_idx의 다중클래스 category(정상/공격
                family 문자열, `data/dataset_loader.py`의 `train_category`).
                CADEDriftDetector가 있으면(`fit_with_category`) 이걸로 CADE의
                실제 family 단위 pairing/centroid를 재현한다(선택적, None이면
                기존 이진 라벨 방식으로 폴백 — 다른 컴포넌트는 이 인자를 모른다).

        Returns:
            {'round', 'exp_idx', 'drift_detected', 'drift_score',
             'avg_train_loss', 'threshold', 'eval_scores' (T개 Tensor),
             'eval_labels' (T개 Tensor), 'n_selected', 'label_budget_int',
             'n_optimizer_steps', 'first_epoch_avg_loss',
             'last_epoch_avg_loss'} — 마지막 5개는 15절 스모크 테스트
            정량 게이트(experiments/smoke_test.py)가 읽는 진단 필드.
        """
        self._round += 1
        new_data = train_data.to(self.device)
        new_labels = train_labels.to(self.device)

        # ---- Step 2: Drift 감지 (buf_ref = 이전 experience까지의 버퍼) ----
        buf_data, _ = self.memory_manager.get_buffer()
        new_logit = buf_logit = None  # dd=cade처럼 공유 표현을 안 쓰면 None 유지
        if self.drift_detector.uses_shared_representation:
            self.model.eval()
            with torch.no_grad():
                _, _, new_logit = self.forward_batched(new_data)
                if buf_data is not None:
                    _, _, buf_logit = self.forward_batched(buf_data.to(self.device))
            drift_score = self.drift_detector.get_drift_score(new_logit, buf_logit)
            drift_detected = self.drift_detector.detect(new_logit, buf_logit)
        else:
            buf_raw = buf_data.to(self.device) if buf_data is not None else None
            drift_score = self.drift_detector.get_drift_score(new_data, buf_raw)
            drift_detected = self.drift_detector.detect(new_data, buf_raw)

        # Item 1+2 — SSF 공유 마스크(M_c/M_t)를 라운드당 1회 계산해
        # sample_selector/memory_manager가 공유한다(둘 중 하나라도 소비할
        # 때만). Step 2가 이미 shared-representation 경로로 new_logit/
        # buf_logit을 계산했다면 재사용 — dd=cade와 짝지어진 경우만 별도
        # forward(`_compute_ssf_masks` 내부에서 처리).
        ssf_ctx: Optional[SSFMaskContext] = None
        if (hasattr(self.sample_selector, "set_ssf_masks")
                or hasattr(self.memory_manager, "set_ssf_masks")):
            ssf_ctx = self._compute_ssf_masks(new_data, buf_data, new_logit, buf_logit)
            if hasattr(self.sample_selector, "set_ssf_masks"):
                self.sample_selector.set_ssf_masks(ssf_ctx)
            if hasattr(self.memory_manager, "set_ssf_masks"):
                self.memory_manager.set_ssf_masks(ssf_ctx)

        # ---- Step 3: 샘플 선택과 라벨 예산 확정 ----
        if self.track == "B":
            # CND-IDS 원 논문(Fuhrman et al., Algorithm 1)은 label_budget
            # 없이 experience 전체를 학습에 쓴다. CNDIDSAntiForgetting.
            # compute_loss()도 selected_labels를 쓰지 않는 라벨-프리 설계
            # (사용자 결정, docs/metric_justification.md).
            label_budget_int = len(new_data)
            selected_data = new_data
            selected_labels = new_labels
            selected_category = train_category
        else:
            label_budget_int = self._label_budget_int(len(new_data), labeling_budget)
            # SSFSampleSelector 전용 훅 — 다중클래스 쿼터 채택 근거는
            # ssf_sample_selector.py "재검증" 절.
            if hasattr(self.sample_selector, "select_with_category") and train_category is not None:
                sel_idx = self.sample_selector.select_with_category(
                    new_data, new_labels, train_category, label_budget_int, drift_score)
            else:
                sel_idx = self.sample_selector.select(
                    new_data, new_labels, label_budget_int, drift_score)
            if len(sel_idx) == 0:
                sel_idx = list(range(min(label_budget_int, len(new_data))))
            selected_data = new_data[sel_idx]
            selected_labels = new_labels[sel_idx]
            selected_category = train_category[sel_idx] if train_category is not None else None
            if ssf_ctx is not None:
                # SSFMemoryManager가 "대표적이지만 미선택인 잔여"를 가려낼
                # 때 씀(ssf_memory_manager.py 참고) — new_data 기준 인덱스.
                ssf_ctx.sel_idx = torch.as_tensor(
                    sel_idx, dtype=torch.long, device=new_data.device)
            # Item 8b — SPIDER self-training(Algorithm 1의 (c) 요소).
            # mm=spider일 때만, GPMAntiForgetting처럼 이 훅을 구현한
            # anti_forgetting에만 sel_idx의 여집합(라벨 예산 밖 잔여)을
            # 전달한다 — Track B는 selected_data가 new_data 전체라 여집합이
            # 항상 비고, af=cndids는 애초에 이 훅이 없어 자연히 no-op.
            if self.combo["memory_manager"] == "spider" and hasattr(self.anti_forgetting, "set_self_training_pool"):
                sel_mask = torch.zeros(len(new_data), dtype=torch.bool, device=new_data.device)
                sel_mask[torch.as_tensor(sel_idx, dtype=torch.long, device=new_data.device)] = True
                self.anti_forgetting.set_self_training_pool(new_data[~sel_mask])
            # GPMAntiForgetting 전용 훅(2026-09-14, gpm_anti_forgetting.py
            # "라벨예산 면제" 절 참고) — CADEDriftDetector/SPIDERMemoryManager와
            # 같은 근거(원 논문에 라벨예산 개념 없음)로, SVD 기저 계산에 쓸
            # activation 표본을 selected_data(라벨예산 서브셋)가 아니라
            # new_data(라운드 전체)에서 직접 뽑도록 라운드당 1회 넘겨준다 —
            # 분류기 학습 자체(task loss)는 여전히 selected_data만 쓴다.
            if (getattr(self.anti_forgetting, "consumes_full_round_data", False)
                    and hasattr(self.anti_forgetting, "set_full_round_data")):
                self.anti_forgetting.set_full_round_data(new_data, new_labels)
        # selected_data 중 label=0 서브셋 — anomaly_scorer 재보정(step 6)과
        # CND-IDS 클러스터링의 "정상 참조"로 씀.
        normal_subset = selected_data[selected_labels == 0]

        if self.drift_detector.uses_shared_representation:
            self.model.eval()
            with torch.no_grad():
                _, _, sel_logit = self.forward_batched(selected_data)
            self.drift_detector.fit(sel_logit, selected_labels)
        elif hasattr(self.drift_detector, "fit_with_category") and train_category is not None:
            # CADEDriftDetector 전용 — family 단위 pairing/centroid.
            # consumes_full_round_data(2026-09-12, docs/design_decisions.md
            # 5절) — CADE 원문에 라벨 예산 개념이 없어 selected_data(라벨 예산 통과분)
            # 대신 new_data(그 라운드 전체)로 학습시킨다.
            if getattr(self.drift_detector, "consumes_full_round_data", False):
                self.drift_detector.fit_with_category(new_data, new_labels, train_category)
            else:
                self.drift_detector.fit_with_category(selected_data, selected_labels, selected_category)
        else:
            self.drift_detector.fit(selected_data, selected_labels)

        # CND-IDS 전용 훅 — 미니배치 학습 전 라운드당 1회 clustering
        # (components/cndids/cndids_anti_forgetting.py). 고정 정상 참조가
        # 있으면(held_out_normal_reference) 매 라운드 그 시점 스케일로
        # 먼저 전달한다 — 컴포넌트 내부의 저수지(`_normal_ref_pool`) 로직을
        # 대체한다(set_held_out_reference가 없거나 참조가 없으면 기존
        # 저수지 로직 유지, cndids_anti_forgetting.py 참고).
        has_fixed_ref = False
        if (self._held_out_normal_reference is not None
                and hasattr(self.anti_forgetting, "set_held_out_reference")):
            fixed_ref = self._held_out_normal_reference[exp_idx].to(self.device)
            self.anti_forgetting.set_held_out_reference(fixed_ref)
            has_fixed_ref = len(fixed_ref) > 0
        if hasattr(self.anti_forgetting, "on_experience_start") and (has_fixed_ref or len(normal_subset) > 0):
            self.anti_forgetting.on_experience_start(selected_data, normal_subset)

        # ---- Step 4: 모델 학습 (selected_data만, replay_batch는 "이전" 버퍼) ----
        self.model.train()
        n_sel = len(selected_data)

        total_loss = 0.0
        n_steps = 0
        # 첫/마지막 epoch 평균 손실을 따로 기록 — smoke_test의 발산 감지용.
        first_epoch_loss_sum, first_epoch_steps = 0.0, 0
        last_epoch_loss_sum, last_epoch_steps = 0.0, 0
        for epoch_idx in range(self.epochs_per_experience):
            perm = torch.randperm(n_sel, device=self.device)
            shuffled_data = selected_data[perm]
            shuffled_labels = selected_labels[perm]
            epoch_loss_sum, epoch_steps = 0.0, 0
            for start in range(0, n_sel, self.batch_size):
                end = min(start + self.batch_size, n_sel)
                batch_data = shuffled_data[start:end]
                batch_labels = shuffled_labels[start:end]

                replay_batch = None
                r_data, r_labels = self.memory_manager.get_replay_batch(self.batch_size)
                if r_data is not None:
                    replay_batch = (r_data.to(self.device), r_labels.to(self.device))

                # Item 8b — SPIDER self-training. GPMAntiForgetting만 이
                # 훅을 구현하므로 다른 anti_forgetting은 이 분기 자체를
                # 타지 않는다(hasattr 게이팅).
                loss_kwargs = {}
                if hasattr(self.anti_forgetting, "get_self_training_batch"):
                    self_training_batch = self.anti_forgetting.get_self_training_batch(
                        self.model, self.batch_size)
                    if self_training_batch is not None:
                        loss_kwargs["self_training_batch"] = self_training_batch

                self.optimizer.zero_grad()
                loss = self.anti_forgetting.compute_loss(
                    self.model, (batch_data, batch_labels), replay_batch, **loss_kwargs)
                loss.backward()
                self.anti_forgetting.project_gradients(self.model)
                self.optimizer.step()

                loss_value = float(loss.item())
                total_loss += loss_value
                n_steps += 1
                epoch_loss_sum += loss_value
                epoch_steps += 1

            if epoch_idx == 0:
                first_epoch_loss_sum, first_epoch_steps = epoch_loss_sum, epoch_steps
            last_epoch_loss_sum, last_epoch_steps = epoch_loss_sum, epoch_steps

        first_epoch_avg_loss = first_epoch_loss_sum / max(first_epoch_steps, 1)
        last_epoch_avg_loss = last_epoch_loss_sum / max(last_epoch_steps, 1)

        # ---- Step 5: 메모리 갱신 (학습 이후) ----
        # SSFMemoryManager는 이진 라벨 쿼터만 쓴다 — 다중클래스 쿼터는 A/B로
        # 더 나쁨을 확인(ssf_memory_manager.py).
        # SPIDERMemoryManager(Item 8a)는 라벨 예산 서브셋이 아니라 이번
        # 라운드 전체(new_data)에서 무작위 표본을 뽑는다 — 원문의 "이전
        # 태스크의 무작위 샘플"과 일치(spider_memory_manager.py 참고).
        if getattr(self.memory_manager, "consumes_full_round_data", False):
            self.memory_manager.update(new_data, new_labels, drift_detected)
        else:
            self.memory_manager.update(selected_data, selected_labels, drift_detected)

        # ---- Step 6: Anomaly Scorer 재보정 ----
        # Item 5 — PCAScorer(consumes_held_out_reference=True)는 원문의
        # 고정 참조(init_normal)를 쓴다. held_out_normal_reference가 없는
        # 호출(단위 테스트 등)은 기존처럼 이번 라운드 normal_subset으로
        # 폴백한다 — 둘 다 없으면(비어있으면) 재보정을 건너뛰고 self._s_ref를
        # 그대로 쓴다(빈 텐서로 score()를 호출하면 median()이 크래시).
        if (getattr(self.anomaly_scorer, "consumes_held_out_reference", False)
                and self._held_out_normal_reference is not None):
            ref_raw = self._held_out_normal_reference[exp_idx].to(self.device)
        elif getattr(self.anomaly_scorer, "consumes_full_round_data", False):
            # CADEMADScorer 전용(2026-09-12, docs/design_decisions.md 5절) —
            # dd=cade와 연결된 경우 fit()이 no-op이라 사실상 영향 없고, as=cade_mad가
            # dd=cade 없이 단독일 때만 실제로 라벨 예산 밖 정상 표본까지 쓴다.
            ref_raw = new_data[new_labels == 0]
        else:
            ref_raw = normal_subset
        if len(ref_raw) > 0:
            # CADEMADScorer가 사설 인코더에 연결된 경우 원본 데이터를 그대로
            # 넘긴다 — 그 안에서 재인코딩.
            if self.anomaly_scorer.uses_shared_representation:
                self.model.eval()
                with torch.no_grad():
                    current_normal_encoded, _, _ = self.forward_batched(ref_raw)
                current_normal_encoded = current_normal_encoded.detach()
            else:
                current_normal_encoded = ref_raw
            self.anomaly_scorer.refit_on_update(current_normal_encoded)
            self._s_ref = self.anomaly_scorer.score(current_normal_encoded).detach()

        # ---- Step 7: 평가 (experience 0..T-1 전부) ----
        eval_scores: List[torch.Tensor] = []
        eval_labels: List[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for test_x, test_y in all_test_splits:
                if self.anomaly_scorer.uses_shared_representation:
                    z, _, _ = self.forward_batched(test_x.to(self.device))
                    scores = self.anomaly_scorer.score(z.detach())
                else:
                    # CADEMADScorer가 사설 인코더에 연결된 경우 — 원본
                    # 데이터를 그대로 넘긴다(Step 6과 동일).
                    scores = self.anomaly_scorer.score(test_x.to(self.device))
                eval_scores.append(scores.cpu())
                eval_labels.append(test_y.cpu())

        # threshold_needs_labels(scorer 속성, base/anomaly_scorer.py)로 분기 —
        # track이 아니라 scorer 자체 속성인 이유: Track B의 as=cade_mad
        # 추가로 "Track B인데 라벨 불필요"인 경우가 생겼기 때문.
        if not self.anomaly_scorer.threshold_needs_labels and self._s_ref is not None:
            threshold = self.anomaly_scorer.compute_threshold(self._s_ref, None)
        else:
            # threshold 계산을 eval_scores[:exp_idx+1]
            # (0..exp_idx, 미래 라운드 제외)로 한정한다. 이전엔 전체
            # eval_scores(0..T-1)를 썼는데, 이러면 아직 등장하지 않은 라운드의
            # test 라벨로 threshold를 고르는 셈이 되어(PCAScorer의 Best-F가
            # 라벨을 직접 소비) R-matrix 대각선(R[i,i])과 그걸 쓰는 bwt()가
            # 미래 정보로 오염됐다. R-matrix 행 자체는 그대로 T개 전부
            # 채점(forward transfer 리포팅 유지), threshold 보정 기준만 분리.
            causal_scores = torch.cat(eval_scores[:exp_idx + 1])
            causal_labels = torch.cat(eval_labels[:exp_idx + 1])
            threshold = self.anomaly_scorer.compute_threshold(causal_scores, causal_labels)

        # ---- Step 8: 다음 라운드 준비 ----
        self.anti_forgetting.on_task_end(self.model)
        # SPIDERMemoryManager 전용 훅 — 직전 태스크 모델 스냅샷 필요
        # (components/spider_gpm/spider_memory_manager.py).
        if hasattr(self.memory_manager, "set_snapshot_model"):
            self.memory_manager.set_snapshot_model(self.model)

        return {
            "round": self._round,
            "exp_idx": exp_idx,
            "drift_detected": bool(drift_detected),
            "drift_score": float(drift_score),
            "avg_train_loss": total_loss / max(n_steps, 1),
            "threshold": float(threshold),
            "eval_scores": eval_scores,
            "eval_labels": eval_labels,
            # 15절 스모크 테스트 정량 게이트 검증용 진단 필드
            "n_selected": n_sel,
            "label_budget_int": label_budget_int,
            "n_optimizer_steps": n_steps,
            "first_epoch_avg_loss": first_epoch_avg_loss,
            "last_epoch_avg_loss": last_epoch_avg_loss,
        }
