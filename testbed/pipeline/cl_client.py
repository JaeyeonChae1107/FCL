"""CLClient — PRD 13절의 8단계 실행 흐름을 그대로 구현한다.

  1. 새 데이터 도착
  2. Drift 감지 (buf_ref = 이전 experience까지의 버퍼)
  3. 샘플 선택과 라벨 예산 확정 (label_budget_int → select → slice →
     drift_detector.fit(selected_data, selected_labels))
  4. 모델 학습 (epochs_per_experience, replay_batch는 "이전" 버퍼에서,
     selected_data만 사용 — experience 전체가 아니다)
  5. 메모리 갱신 (학습 이후, selected_data 그대로)
  6. Anomaly Scorer 재보정 (refit_on_update, s_ref 계산·캐싱)
  7. 평가 (experience 0..T-1 전부의 test split, anomaly_scorer.threshold_needs_labels
     에 따른 threshold 결정방식 — 2026-09-01 이전엔 Track별로 분기했으나
     Track B에 as=cade_mad가 추가되며 scorer 자체의 속성으로 일반화했다,
     base/anomaly_scorer.py 참고)
  8. 다음 라운드 준비 (anti_forgetting.on_task_end)

Step 4→5 순서(학습 후 메모리 갱신)는 "각 논문에서 그대로 도출"된 것이 아니라,
이 파이프라인이 공유하는 리플레이 버퍼 계약(BaseMemoryManager.get_replay_batch)이
강제하는 순서다: Step 4의 매 미니배치가 get_replay_batch()로 "이전" 버퍼를
읽는데, 만약 Step 5를 Step 4보다 앞에 두면 그 라운드 자신의 selected_data가
먼저 버퍼에 들어가 버려 같은 라운드 안에서 자기 자신을 리플레이하는 꼴이 된다
(SPIDER/CNDIDSMemoryManager처럼 get_replay_batch를 실제로 소비하는 컴포넌트에서
치명적). 이 순서는 SPIDER·CND-IDS 방향 메모리 매니저의 실제 사용 패턴과는
맞지만, SSF 원문(`ssf.py:236-291`)과는 반대다 — SSF는 대표 표본 재선택
(select_and_update_representative_samples[_when_drift]())을 먼저 수행해
`x_train_this_epoch`(메모리이자 곧 이번 라운드 학습 데이터 그 자체)를 갱신한
뒤 그 갱신된 세트로 학습한다. SSF에서는애초에 "메모리"와 "이번 라운드
학습 데이터"가 하나의 객체라 이 구분 자체가 없다. 이 구조적 차이는 의도적으로
되돌리지 않았다(docs/metric_justification.md "SSF 대표 표본 재선택" 절 참고) —
공유 리플레이 계약을 깨지 않으면서 4개 논문 전부를 하나의 파이프라인에 태우기
위한 불가피한 절충이다.

**2026-08-14 추가 — CADEMADScorer/CADEDriftDetector 연결**: 5-슬롯 독립
설계(drift_detector/anomaly_scorer가 서로 다른 축)의 부작용으로, "순정
CADE" 조합에서도 CADE의 실제 발명(대조학습 latent space 위에서 MAD
판정)이 한 번도 통합되어 실행되지 않는 문제를 구조 전수 감사에서 발견했다
— `CADEDriftDetector`가 학습시키는 사설 대조학습 인코더의 출력이 어디에도
안 쓰이고, `CADEMADScorer`는 무관한 공유 backbone의 z에 MAD 공식만
적용하고 있었다. `dd=cade`와 `as=cade_mad`가 함께 선택된 콤보에서만
`__init__`이 `CADEMADScorer.set_private_encoder()`로 둘을 연결한다
(`drift_detector.uses_shared_representation`과 대칭인
`anomaly_scorer.uses_shared_representation` 플래그로 Step 6/7의 인코딩
경로를 분기 — `components/cade/cade_anomaly_scorer.py` 참고). A/B
실측(NSL-KDD, 순정 CADE 콤보)으로 f1 0.6482→0.7898(+22%), bwt
-0.1403→-0.0810로 전 지표가 크게 개선됨을 확인했다.

**2026-08-25 추가 — CADE 다중클래스(family) 연결**: 위 연결은 인코더/
centroid 파이프라인 자체는 이었지만, `drift_detector.fit()`에 넘기는
`selected_labels`가 여전히 이진이라 CADE의 실제 단위(정상 + 공격 family)가
빠져 있었다(`components/cade/cade_drift_detector.py` 모듈 docstring
"2026-08-25" 절 참고). `run_experience()`가 이제 `data/dataset_loader.py`가
노출하는 `train_category`를 받아 `selected_idx`와 같은 인덱스로 슬라이싱한
뒤, `drift_detector.fit_with_category()`가 있으면(hasattr, CADEDriftDetector
전용) 그걸로 pairing/centroid를 만든다 — 없으면 기존 이진 `fit()`으로
폴백한다(다른 drift_detector는 이 인자를 모른 채 그대로 동작). A/B 실측
결과는 위 파일과 `docs/metric_justification.md`에 기록한다.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from testbed.base.models import BaseCLModel
from testbed.pipeline.component_registry import build


class CLClient:
    def __init__(self, model: BaseCLModel, combo: Dict[str, Any],
                 global_hparams: Dict[str, Any],
                 component_hparams: Optional[Dict[str, Dict[str, Any]]] = None,
                 device: str = "cpu"):
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
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.combo = combo
        self.track = combo["track"]
        component_hparams = component_hparams or {}

        input_dim = global_hparams.get("_input_dim")  # dataset_loader가 채워 넣음
        hidden_dim = global_hparams["hidden_dim"]
        latent_dim = global_hparams["latent_dim"]

        # 모든 component_hparams/*.yaml을 하나로 병합한다. build()가 각 클래스의
        # 실제 생성자 시그니처로 필터링하므로(component_registry.py), 서로 다른
        # 컴포넌트의 하이퍼파라미터가 섞여 있어도 안전하다 — 파라미터 이름이
        # 컴포넌트 간에 겹치지 않기 때문. 단, input_dim/hidden_dim/latent_dim은
        # global_hparams(10.1절, 모든 조합에 동일 적용)가 항상 우선하도록
        # 마지막에 덮어쓴다(cndids.yaml의 latent_dim=30은 참고용 기록일 뿐).
        merged_component_kwargs: Dict[str, Any] = {}
        for hp in component_hparams.values():
            merged_component_kwargs.update(hp)
        merged_component_kwargs.update(
            input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
            batch_size=global_hparams["batch_size"])
        # batch_size는 CADEDriftDetector의 사설 encoder 미니배치 학습에만 쓰인다
        # (2026-08-11 추가) — 다른 컴포넌트는 이 이름의 생성자 인자가 없어
        # build()의 시그니처 필터링으로 자동으로 무시된다.

        self.drift_detector = build(
            "drift_detector", combo["drift_detector"], **merged_component_kwargs)
        # CADEDriftDetector 전용 훅 — 유일하게 자기 소유의 nn.Module(사설
        # ContrastiveAutoEncoder)을 갖는 컴포넌트라, self.model처럼 명시적으로
        # 디바이스를 옮겨줘야 한다(components/cade/cade_drift_detector.py 참고).
        if hasattr(self.drift_detector, "to"):
            self.drift_detector.to(self.device)
        self.sample_selector = build(
            "sample_selector", combo["sample_selector"], **merged_component_kwargs)
        self.memory_manager = build("memory_manager", combo["memory_manager"])
        self.anti_forgetting = build(
            "anti_forgetting", combo["anti_forgetting"], **merged_component_kwargs)
        self.anomaly_scorer = build(
            "anomaly_scorer", combo["anomaly_scorer"], **merged_component_kwargs)
        # NoAnomalyScorer 전용 훅 — 표준 계약(z 소비) 밖에서 model 참조가
        # 필요한 유일한 컴포넌트 (pipeline/common_baselines.py 참고).
        if hasattr(self.anomaly_scorer, "set_model"):
            self.anomaly_scorer.set_model(self.model)
        # CADEMADScorer 전용 훅 — dd=cade와 함께 선택됐을 때만 CADEDriftDetector에
        # 연결한다(components/cade/cade_anomaly_scorer.py "2026-08-14"/"2026-08-25"
        # 절 참고, 2026-08-25부터 인코더 nn.Module이 아니라 detector 객체 전체를
        # 넘긴다 — score()가 detector의 min_anomaly_score()에 위임하기 위해).
        # hasattr 이중 체크로 두 컴포넌트가 우연히 같은 이름의 속성/메서드를
        # 가진 경우와 구분한다.
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

        # 2026-07-30 재설계: 이전에는 experience 0에서 한 번 뽑은 고정
        # 정상 참조 표본(_normal_reference_raw)을 실험 내내 재사용했다(논문에
        # 없는 개념, docs/metric_justification.md 참고). CADE 원 논문의
        # median/MAD 계산은 "그 시점에 라벨이 있는 정상 데이터 전체"를 쓰지만,
        # CADE 자체에는 이 테스트베드처럼 반복되는 라운드 개념이 없어 "매
        # 라운드 뭘 기준으로 재보정할지"는 애초에 답이 없는 질문이었다. 별도
        # 고정 표본을 새로 만드는 대신, 이미 라벨 예산으로 선택된 이번 라운드
        # 데이터(selected_data) 중 label=0인 것만 걸러 쓰기로 했다 —
        # normal_reference_size라는 별도 파라미터가 필요 없어지고,
        # labeling_budget 비율에 자동으로 비례한다. 이번 라운드에 정상 라벨
        # 선택 샘플이 하나도 없는 극단적 경우를 위해 마지막으로 성공한
        # 재보정 결과(self._s_ref)를 캐시해 재사용한다(run_experience 참고).
        self._s_ref: Optional[torch.Tensor] = None
        self._round = 0

    def forward_batched(self, x: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """model(x)를 self.batch_size 단위로 나눠 실행하고 (z, x_hat, logit)을
        이어붙여 반환한다. NSL-KDD/UNSW-NB15는 experience 하나가 수천~수만 행이라
        한 번에 통과시켜도 문제없었지만, CICIDS2018은 experience 하나가 수백만
        행이라 그대로 통과시키면 GPU 메모리가 터진다(실측: CUDA OOM, GPU 서버에서
        확인). Step 4(학습)는 이미 self.batch_size로 나눠 돌고 있었으므로, 그
        외 forward 호출(step 2/3/7의 drift 감지·fit·평가)에도 동일하게 배치를
        적용한다 — 호출부가 이미 `torch.no_grad()`로 감싸므로 여기서는 그래디언트
        관리를 하지 않는다(기존 호출 패턴과 동일)."""
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
            {'round', 'drift_detected', 'drift_score', 'avg_train_loss',
             'threshold', 'eval_scores' (T개 Tensor), 'eval_labels' (T개 Tensor)}
        """
        self._round += 1
        new_data = train_data.to(self.device)
        new_labels = train_labels.to(self.device)

        # ---- Step 2: Drift 감지 (buf_ref = 이전 experience까지의 버퍼) ----
        buf_data, _ = self.memory_manager.get_buffer()
        if self.drift_detector.uses_shared_representation:
            self.model.eval()
            with torch.no_grad():
                _, _, new_logit = self.forward_batched(new_data)
                buf_logit = None
                if buf_data is not None:
                    _, _, buf_logit = self.forward_batched(buf_data.to(self.device))
            drift_score = self.drift_detector.get_drift_score(new_logit, buf_logit)
            drift_detected = self.drift_detector.detect(new_logit, buf_logit)
        else:
            buf_raw = buf_data.to(self.device) if buf_data is not None else None
            drift_score = self.drift_detector.get_drift_score(new_data, buf_raw)
            drift_detected = self.drift_detector.detect(new_data, buf_raw)

        # ---- Step 3: 샘플 선택과 라벨 예산 확정 ----
        if self.track == "B":
            # CND-IDS 원 논문(Fuhrman et al., Algorithm 1: "Get Xtrain from
            # experience data Ei" -> "Fit CFE to Xtrain")은 label_budget
            # 개념 없이 experience 전체를 그대로 학습에 쓴다.
            # CNDIDSAntiForgetting.compute_loss()도 selected_labels를 전혀
            # 쓰지 않는 라벨-프리 설계다(12.5절, "라벨-프리 준수" 참고) —
            # 즉 Track B는 애초에 라벨을 한 개도 소비하지 않으므로,
            # "라벨링 비용 절약"을 명목으로 한 label_budget 제한을 적용하면
            # 아낀 라벨 비용 없이 원 논문 대비 데이터만 1/10로 줄어드는
            # 결과가 된다(실측 확인 후 사용자 결정,
            # docs/metric_justification.md 참고). Track B는 label_budget
            # 게이트를 건너뛰고 new_data 전체를 그대로 쓴다.
            label_budget_int = len(new_data)
            selected_data = new_data
            selected_labels = new_labels
            selected_category = train_category
        else:
            label_budget_int = self._label_budget_int(len(new_data), labeling_budget)
            # SSFSampleSelector 전용 훅 — 이진 라벨 쿼터보다 다중클래스 쿼터가
            # 전체 성능과 U2R 라운드 성능 둘 다 더 낫다는 걸 재검증 후 확인해
            # 채택했다(ssf_sample_selector.py 모듈 docstring "2026-08-26
            # 재검증" 절 참고 — 한 번 기각했다가 낡은 A/B 비교였음이 밝혀져
            # 다시 채택함).
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
        # selected_data(Track A는 라벨 예산 안의 데이터, Track B는 위에서
        # 정한 대로 experience 전체) 중 실제로 "정상(label=0)"이라고 알려진
        # 서브셋 — CADE-MAD/PCA 재보정(step 6)과 CND-IDS 클러스터링(아래)
        # 양쪽에서 "정상 참조"로 쓴다. 별도로 고정된 참조 표본을 만들지 않고
        # selected_data에서 바로 걸러내므로 데이터 규모에 자동으로 비례한다
        # (docs/metric_justification.md 참고).
        normal_subset = selected_data[selected_labels == 0]

        if self.drift_detector.uses_shared_representation:
            self.model.eval()
            with torch.no_grad():
                _, _, sel_logit = self.forward_batched(selected_data)
            self.drift_detector.fit(sel_logit, selected_labels)
        elif hasattr(self.drift_detector, "fit_with_category") and selected_category is not None:
            # CADEDriftDetector 전용 훅 — 다중클래스 family로 pairing/centroid를
            # 만든다(components/cade/cade_drift_detector.py "2026-08-25" 절 참고).
            self.drift_detector.fit_with_category(selected_data, selected_labels, selected_category)
        else:
            self.drift_detector.fit(selected_data, selected_labels)

        # CND-IDS 전용 훅 — CND_IDS.py:fit() 진입부와 동일하게, 미니배치 학습이
        # 시작되기 전 experience(라운드)당 한 번만 clustering을 수행한다
        # (components/cndids/cndids_anti_forgetting.py 참고). 이번 라운드에
        # 정상 라벨 선택 샘플이 하나도 없으면(현실적인 NIDS 데이터에서는
        # 극히 드묾) 건너뛴다 — 그러면 이전 라운드에 학습된 K-Means/클러스터
        # 상태가 그대로 유지되고(첫 라운드부터 그런 경우면
        # CNDIDSAntiForgetting의 기존 폴백대로 전부 "정상"으로 간주).
        if hasattr(self.anti_forgetting, "on_experience_start") and len(normal_subset) > 0:
            self.anti_forgetting.on_experience_start(selected_data, normal_subset)

        # ---- Step 4: 모델 학습 (selected_data만, replay_batch는 "이전" 버퍼) ----
        self.model.train()
        total_loss = 0.0
        n_steps = 0
        n_sel = len(selected_data)
        # 2026-08-14 추가 — 첫 epoch과 마지막 epoch의 평균 손실을 따로
        # 기록한다. 기존엔 전체 epoch 평균(avg_train_loss) 하나만 반환해서,
        # smoke_test가 "학습 루프가 실행됐는가"(파라미터 변화량, optimizer
        # step 횟수)는 검증해도 "손실이 실제로 줄어드는 방향으로 갔는가"는
        # 전혀 검증하지 못했다(구조 전수 감사에서 발견) — first/last를 함께
        # 반환해 smoke_test 쪽에서 발산 여부를 직접 판단할 수 있게 한다.
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

                self.optimizer.zero_grad()
                loss = self.anti_forgetting.compute_loss(
                    self.model, (batch_data, batch_labels), replay_batch)
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
        # 2026-08-26 재검증 후에도 기각 유지 — SSFMemoryManager의 다중클래스
        # category 쿼터는 재검증(최신 코드 기준 4-way A/B) 후에도 이진 쿼터
        # 보다 나쁨을 재확인했다(ssf_memory_manager.py 모듈 docstring 참고).
        # 이진 라벨 쿼터만 쓴다.
        self.memory_manager.update(selected_data, selected_labels, drift_detected)

        # ---- Step 6: Anomaly Scorer 재보정 ----
        # normal_subset이 비어있으면(이번 라운드 라벨 예산에 정상 샘플이 하나도
        # 없었던 경우 — 현실적인 NIDS 데이터에서는 극히 드묾) 재보정을
        # 건너뛰고 마지막으로 성공한 self._s_ref/scorer 내부 상태를 그대로
        # 쓴다. score()를 빈 텐서에 호출하면 Track A의 compute_threshold가
        # median()을 빈 텐서에 호출해 크래시하므로, 애초에 빈 입력으로
        # score()/재보정을 시도하지 않는다.
        if len(normal_subset) > 0:
            # CADEMADScorer가 사설 인코더에 연결된 경우(위 __init__ 참고)는
            # 원본 데이터를 그대로 넘긴다 — 그 안에서 스스로 재인코딩한다.
            if self.anomaly_scorer.uses_shared_representation:
                self.model.eval()
                with torch.no_grad():
                    current_normal_encoded, _, _ = self.forward_batched(normal_subset)
                current_normal_encoded = current_normal_encoded.detach()
            else:
                current_normal_encoded = normal_subset
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
                    # 데이터를 그대로 넘긴다(Step 6과 동일한 이유).
                    scores = self.anomaly_scorer.score(test_x.to(self.device))
                eval_scores.append(scores.cpu())
                eval_labels.append(test_y.cpu())

        # 2026-09-01 수정 — self.track이 아니라 anomaly_scorer 자체가
        # 선언하는 threshold_needs_labels로 분기한다(base/anomaly_scorer.py
        # "2026-09-01" 절 참고). 지금까지는 track과 완전히 겹쳤지만(Track
        # A=cade_mad/none=False, Track B=pca=True), Track B에 as=cade_mad가
        # 추가되면서 "Track B이지만 라벨 불필요"인 경우가 처음 생겼다 —
        # 그 경우도 Track A의 dd=none+as=cade_mad와 동일하게 s_ref 방식을
        # 써야 한다.
        if not self.anomaly_scorer.threshold_needs_labels and self._s_ref is not None:
            threshold = self.anomaly_scorer.compute_threshold(self._s_ref, None)
        else:
            # threshold_needs_labels=True(pca)는 원래도 pooled eval score
            # 방식. threshold_needs_labels=False인데 self._s_ref가 여태
            # 한 번도 채워지지 않은 극단적 예외 상황(지금까지 모든 라운드에서
            # 정상 라벨 선택 샘플이 없었던 경우)에도 크래시하지 않도록 같은
            # 폴백을 쓴다.
            #
            # **2026-08-26 발견·수정 — 미래 라운드 test 라벨이 threshold
            # 보정에 새어 들어가던 문제**: `eval_scores`/`eval_labels`는
            # `all_test_splits`(experience 0..T-1 전부) 순서 그대로 쌓이는데,
            # 여기서 그 전체(`torch.cat(eval_scores)`)를 threshold 계산에
            # 썼다 — 즉 라운드 `exp_idx`의 모델이 아직 등장하지도 않은
            # 라운드 `exp_idx+1..T-1`의 실제 라벨까지 미리 보고 그 라벨들
            # 기준으로 최적 threshold를 고르는 셈이었다(`PCAScorer`의
            # Best-F, `precision_recall_curve` 기반 오라클 탐색이라 라벨을
            # 직접 소비한다 — 이 버그 발견 당시엔 Track B의 유일한
            # anomaly_scorer가 이 경로였다). 이 threshold가 R-matrix의
            # **대각선(R[i,i], 그 라운드 자신의 test 성능)까지** 결정하므로,
            # `bwt()`가 쓰는 모든 대각선 값이 미래 정보로 오염된 threshold로
            # 계산되고 있었다 — 당시 Track B 3개 조합(pca) 전부의 BWT가
            # 이 영향을 받았다.
            # eval "범위"(모든 T개 라운드에 대해 채점하는 것, forward
            # transfer 측정용 — 정당한 리포팅)와 threshold "보정 기준"
            # (지금까지 실제로 등장한 라운드만 — 인과적으로 정당해야 함)을
            # 분리한다: threshold는 `eval_scores[:exp_idx+1]`(0..exp_idx,
            # 이번 라운드 자신의 test 포함, 그 이후는 제외)로만 계산하고,
            # R-matrix 행 자체(`grid_runner.py`가 `out["eval_scores"]`
            # 전체로 구성)는 그대로 T개 전부를 채점해 forward transfer
            # 리포팅은 그대로 유지한다.
            causal_scores = torch.cat(eval_scores[:exp_idx + 1])
            causal_labels = torch.cat(eval_labels[:exp_idx + 1])
            threshold = self.anomaly_scorer.compute_threshold(causal_scores, causal_labels)

        # ---- Step 8: 다음 라운드 준비 ----
        self.anti_forgetting.on_task_end(self.model)
        # SPIDERMemoryManager 전용 훅 — 표준 계약(모델 접근 없음) 밖에서
        # "직전 태스크까지 학습된 모델" 스냅샷이 필요한 유일한 memory_manager
        # (components/spider_gpm/spider_memory_manager.py 참고).
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
