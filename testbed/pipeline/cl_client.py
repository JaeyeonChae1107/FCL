"""CLClient — PRD 13절의 8단계 실행 흐름을 그대로 구현한다.

이 순서는 SSF/SPIDER 등 각 논문의 알고리즘이 실제로 동작하는 순서에서 그대로
도출된 것이며, 임의로 바꾸지 않는다:

  1. 새 데이터 도착
  2. Drift 감지 (buf_ref = 이전 experience까지의 버퍼)
  3. 샘플 선택과 라벨 예산 확정 (label_budget_int → select → slice →
     drift_detector.fit(selected_data, selected_labels))
  4. 모델 학습 (epochs_per_experience, replay_batch는 "이전" 버퍼에서,
     selected_data만 사용 — experience 전체가 아니다)
  5. 메모리 갱신 (학습 이후, selected_data 그대로)
  6. Anomaly Scorer 재보정 (refit_on_update, s_ref 계산·캐싱)
  7. 평가 (experience 0..T-1 전부의 test split, Track별 threshold 결정방식)
  8. 다음 라운드 준비 (anti_forgetting.on_task_end)
"""

from typing import Any, Dict, List, Optional, Tuple

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
            input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)

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

        optimizer_name = global_hparams.get("optimizer", "adam")
        lr = global_hparams["learning_rate"]
        if optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        self.batch_size = global_hparams["batch_size"]
        self.epochs_per_experience = global_hparams["epochs_per_experience"]

        self._normal_reference_raw: Optional[torch.Tensor] = None
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

    def set_normal_reference(self, normal_data: torch.Tensor) -> None:
        """12.6절 — experience 0의 train split에서 뽑은 정상(label=0) 참조
        데이터(고정 크기, 실험 내내 불변). dataset_loader가 준비해 1회 주입한다."""
        self._normal_reference_raw = normal_data

    @staticmethod
    def _label_budget_int(n: int, labeling_budget: Dict[str, Any]) -> int:
        if labeling_budget["mode"] == "fixed_count":
            return int(labeling_budget["value"])
        return round(labeling_budget["value"] * n)

    def run_experience(self, exp_idx: int,
                        train_data: torch.Tensor, train_labels: torch.Tensor,
                        all_test_splits: List[Tuple[torch.Tensor, torch.Tensor]],
                        labeling_budget: Dict[str, Any]) -> Dict[str, Any]:
        """experience exp_idx에 대해 13절 8단계를 순서대로 수행한다.

        Args:
            exp_idx: 현재 experience 인덱스 (0-based).
            train_data, train_labels: experience exp_idx의 train split (X_i, y_i).
            all_test_splits: experience 0..T-1 전부의 (test_X, test_y) 리스트
                              (길이 T, 데이터 로딩 시 고정된 그대로 — 9.2절).
            labeling_budget: {'mode': 'fixed_count'|'fixed_ratio', 'value': ...}

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
        label_budget_int = self._label_budget_int(len(new_data), labeling_budget)
        sel_idx = self.sample_selector.select(
            new_data, new_labels, label_budget_int, drift_score)
        if len(sel_idx) == 0:
            sel_idx = list(range(min(label_budget_int, len(new_data))))
        selected_data = new_data[sel_idx]
        selected_labels = new_labels[sel_idx]

        if self.drift_detector.uses_shared_representation:
            self.model.eval()
            with torch.no_grad():
                _, _, sel_logit = self.forward_batched(selected_data)
            self.drift_detector.fit(sel_logit, selected_labels)
        else:
            self.drift_detector.fit(selected_data, selected_labels)

        # CND-IDS 전용 훅 — CND_IDS.py:fit() 진입부와 동일하게, 미니배치 학습이
        # 시작되기 전 experience(라운드)당 한 번만 clustering을 수행한다
        # (components/cndids/cndids_anti_forgetting.py 참고).
        if hasattr(self.anti_forgetting, "on_experience_start"):
            self.anti_forgetting.on_experience_start(
                selected_data, self._normal_reference_raw.to(self.device))

        # ---- Step 4: 모델 학습 (selected_data만, replay_batch는 "이전" 버퍼) ----
        self.model.train()
        total_loss = 0.0
        n_steps = 0
        n_sel = len(selected_data)
        for _ in range(self.epochs_per_experience):
            perm = torch.randperm(n_sel, device=self.device)
            shuffled_data = selected_data[perm]
            shuffled_labels = selected_labels[perm]
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

                total_loss += float(loss.item())
                n_steps += 1

        # ---- Step 5: 메모리 갱신 (학습 이후) ----
        self.memory_manager.update(selected_data, selected_labels, drift_detected)

        # ---- Step 6: Anomaly Scorer 재보정 ----
        self.model.eval()
        with torch.no_grad():
            current_normal_encoded, _, _ = self.forward_batched(
                self._normal_reference_raw.to(self.device))
        current_normal_encoded = current_normal_encoded.detach()
        self.anomaly_scorer.refit_on_update(current_normal_encoded)
        s_ref = self.anomaly_scorer.score(current_normal_encoded).detach()

        # ---- Step 7: 평가 (experience 0..T-1 전부) ----
        eval_scores: List[torch.Tensor] = []
        eval_labels: List[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for test_x, test_y in all_test_splits:
                z, _, _ = self.forward_batched(test_x.to(self.device))
                scores = self.anomaly_scorer.score(z.detach())
                eval_scores.append(scores.cpu())
                eval_labels.append(test_y.cpu())

        if self.track == "A":
            threshold = self.anomaly_scorer.compute_threshold(s_ref, None)
        else:
            all_scores = torch.cat(eval_scores)
            all_labels = torch.cat(eval_labels)
            threshold = self.anomaly_scorer.compute_threshold(all_scores, all_labels)

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
        }
