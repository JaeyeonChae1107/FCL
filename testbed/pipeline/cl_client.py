"""CLClient — pluggable continual-learning client.

Orchestrates: Drift Detector → Sample Selector → Memory Manager
              → Anti-Forgetting → Anomaly Scorer
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from testbed.pipeline.component_registry import build


# ---------------------------------------------------------------------------
# 파이프라인 단계 가이드 (각 단계의 역할·입출력 계약)
# ---------------------------------------------------------------------------
STAGE_GUIDE: Dict[str, Dict[str, Any]] = {
    "1_drift_detector": {
        "역할": (
            "메모리 버퍼(과거 정상 데이터)와 새 배치 간의 분포 변화(concept drift)를 감지한다. "
            "메모리 update() 호출 이전에 실행하여 '이전 분포'를 참조 기준으로 사용한다."
        ),
        "입력": {
            "new_data": "torch.Tensor (N, D) — 현재 라운드의 새 샘플",
            "buf_ref":  "torch.Tensor (M, D) or None — 메모리 버퍼 전체 (이전 분포 참조)",
        },
        "출력": {
            "drift_score":    "float — 표류 강도 (0=표류 없음, 높을수록 강한 표류)",
            "drift_detected": "bool  — 표류 여부 (threshold 적용 후)",
        },
        "보장사항": [
            "버퍼가 None이면 반드시 False/0.0 반환",
            "memory_manager.update() 이전에 실행되므로 이전 분포를 기준으로 비교 가능",
        ],
        "구현체": ["NoDriftDetector(none)", "SSFDriftDetector(ssf)", "CADEDriftDetector(cade)"],
    },
    "2_sample_selector": {
        "역할": (
            "label_budget 내에서 학습에 가장 유익한 샘플 인덱스를 선택한다 (능동 학습). "
            "drift_score를 활용해 표류 시 더 공격적인 전략을 적용할 수 있다."
        ),
        "입력": {
            "new_data":     "torch.Tensor (N, D)",
            "new_labels":   "torch.Tensor (N,) — 정수 레이블 (0=정상, 1=공격)",
            "label_budget": "int — 이번 라운드 최대 레이블 수",
            "drift_score":  "float — Stage 1에서 반환된 표류 강도",
        },
        "출력": {
            "sel_idx": "List[int] — new_data 인덱스, 길이 ≤ label_budget",
        },
        "보장사항": [
            "반환 리스트 비어있으면 CLClient가 range(label_budget)으로 대체",
            "선택된 샘플만 Stage 3(메모리 업데이트)·Stage 5(손실 계산)에 사용됨",
        ],
        "구현체": ["AllSampleSelector(all)", "RandomSelector(random)", "SSFSampleSelector(ssf)"],
    },
    "3_memory_manager": {
        "역할": (
            "선택된 샘플을 replay 버퍼에 추가/교체하여 이전 태스크 경험을 유지한다. "
            "drift 발생 시 오래된 샘플을 더 공격적으로 교체할 수 있다."
        ),
        "입력": {
            "selected_data":   "torch.Tensor (K, D) — Stage 2에서 선택된 샘플",
            "selected_labels": "torch.Tensor (K,)",
            "drift_detected":  "bool — Stage 1의 출력",
        },
        "출력": {
            "버퍼(side-effect)": "내부 버퍼 갱신. get_buffer()/get_replay_batch()로 접근",
        },
        "보장사항": [
            "update() 완료 후 get_buffer()는 최신 버퍼를 반환",
            "Stage 4(replay batch 조회)는 update() 이후 실행됨",
        ],
        "구현체": ["NoMemoryManager(none)", "FIFOMemoryManager(fifo)", "SSFMemoryManager(ssf)"],
    },
    "4_replay_retrieval": {
        "역할": (
            "[CLClient 내부 단계 — 별도 플러그인 없음] "
            "메모리 버퍼에서 replay mini-batch를 샘플링하여 Stage 5(anti-forgetting)에 전달한다."
        ),
        "입력": {"batch_size": "int — label_budget과 동일"},
        "출력": {"replay_batch": "Tuple(data, labels) or None (버퍼 비어있을 경우)"},
        "보장사항": [
            "버퍼가 비어있으면 replay_batch=None이 Stage 5로 전달됨",
            "SSFAntiForgetting은 replay_batch=None을 drift 모드로 간주 (LwF 생략)",
        ],
        "구현체": ["CLClient.update() 내부 로직"],
    },
    "5_anti_forgetting": {
        "역할": (
            "new_batch와 replay_batch를 결합하여 catastrophic forgetting을 방지하는 "
            "스칼라 손실을 계산한다. loss.backward() 이후 선택적으로 "
            "project_gradients()를 통해 gradient를 이전 태스크 직교 방향으로 제한한다."
        ),
        "입력": {
            "model":        "nn.Module — 훈련 중인 현재 모델",
            "new_batch":    "Tuple(sel_data, sel_labels) — 현재 라운드 선택 샘플",
            "replay_batch": "Tuple(r_data, r_labels) or None — 버퍼 샘플",
        },
        "출력": {"loss": "torch.Tensor (scalar, requires_grad=True)"},
        "보장사항": [
            "on_task_end(model) 훅이 optimizer.step() 이후 반드시 호출됨",
            "project_gradients()는 backward() 이후, step() 이전에 호출됨 (GPM만 해당)",
        ],
        "구현체": ["ReplayOnlyLoss(none)", "CNDIDSAntiForgetting(cndids)", "GPMAntiForgetting(gpm)", "SSFAntiForgetting(lwf_ssf)"],
    },
    "anomaly_scorer": {
        "역할": (
            "[update() 파이프라인 외부 단계] "
            "인코더 출력(latent representation)에서 이상 점수를 계산하고 "
            "임계값으로 이진 분류한다. fit()은 정상 데이터로만 호출된다."
        ),
        "입력": {
            "fit":     "normal_data: torch.Tensor (N, D) — 정상(label=0) 인코더 출력",
            "score":   "data: torch.Tensor (N, D) — 추론 대상",
            "predict": "data + threshold: float (자동 설정됨)",
        },
        "출력": {
            "score":   "torch.Tensor (N,) float — 높을수록 이상",
            "predict": "torch.Tensor (N,) long  — 0=정상, 1=이상",
        },
        "보장사항": [
            "fit_anomaly_scorer()가 정상 데이터의 95th percentile을 임계값으로 자동 설정",
            "인코더 forward는 model.eval() + torch.no_grad() 하에서 실행됨",
        ],
        "구현체": ["PCAAnomalyScorer(pca)", "CADEAnomalyScorer(cade_mad)"],
    },
}


class CLClient:
    """Federated / continual learning client with swappable components.

    Config example::

        config = {
            "drift_detector":  {"name": "ssf", "drift_threshold": 0.05},
            "sample_selector": {"name": "ssf", "mask_threshold": 0.5},
            "memory_manager":  {"name": "ssf", "max_size": 1000},
            "anti_forgetting": {"name": "lwf_ssf", "lwf_lambda": 0.5},
            "anomaly_scorer":  {"name": "pca"},
            "label_budget": 50,
            "lr": 1e-3,
        }
        client = CLClient(model=my_model, config=config, device='cpu')
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any],
                 device: str = 'cpu'):
        """
        Args:
            model: PyTorch model. Expected forward: x → (z, recon) or z.
            config: Dict with keys for each component slot plus 'label_budget'
                    and 'lr'.
            device: Torch device string.
        """
        self.model = model.to(device)
        self.device = torch.device(device)
        self.config = config
        self.label_budget: int = config.get('label_budget', 50)
        self.lr: float = config.get('lr', 1e-3)

        # Build components from registry
        self.drift_detector = build(
            'drift_detector', **config.get('drift_detector', {'name': 'none'}))
        self.sample_selector = build(
            'sample_selector', **config.get('sample_selector', {'name': 'random'}))
        self.memory_manager = build(
            'memory_manager', **config.get('memory_manager', {'name': 'none'}))
        self.anti_forgetting = build(
            'anti_forgetting', **config.get('anti_forgetting', {'name': 'none'}))
        self.anomaly_scorer = build(
            'anomaly_scorer', **config.get('anomaly_scorer', {'name': 'pca'}))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self._anomaly_threshold: float = 0.5
        self._round: int = 0
        self.batch_size: int = config.get('batch_size', 64)
        self.n_epochs: int = config.get('n_epochs', 1)

    # ------------------------------------------------------------------
    def update(self, new_data: torch.Tensor,
               new_labels: torch.Tensor) -> Dict[str, Any]:
        """Run one continual-learning round.

        Pipeline stages (STAGE_GUIDE 참조):
          Stage 1 — DriftDetector : 이전 메모리 버퍼 대비 분포 변화 감지
          Stage 2 — SampleSelector: label_budget 내 최적 샘플 인덱스 선택
          Stage 3 — MemoryManager : 선택 샘플을 replay 버퍼에 추가
          Stage 4 — [내부]        : 버퍼에서 replay mini-batch 샘플링
          Stage 5 — AntiForgetting: new_batch + replay_batch로 손실 계산 및 역전파
          Hook    — on_task_end() : 교사 모델 스냅샷 등 태스크 종료 처리

        Args:
            new_data: Incoming data batch. Shape (N, D).
            new_labels: Corresponding labels. Shape (N,).

        Returns:
            Dict with keys 'loss', 'drift', 'drift_score', 'round'.
        """
        new_data = new_data.to(self.device)
        new_labels = new_labels.to(self.device)
        self._round += 1

        # 1. Drift detection
        buf_data, _ = self.memory_manager.get_buffer()
        buf_ref = buf_data.to(self.device) if buf_data is not None else None
        drift_score = self.drift_detector.get_drift_score(new_data, buf_ref)
        drift_detected = self.drift_detector.detect(new_data, buf_ref)

        # 2. Sample selection
        sel_idx = self.sample_selector.select(
            new_data, new_labels, self.label_budget, drift_score)
        if not sel_idx:
            sel_idx = list(range(min(self.label_budget, len(new_data))))
        sel_data = new_data[sel_idx]
        sel_labels = new_labels[sel_idx]

        # 3. Memory update
        self.memory_manager.update(sel_data, sel_labels, drift_detected)

        # 3b. Refit stateful drift detectors (e.g. CADE) on current task data
        #     so next round compares against this task's distribution.
        if hasattr(self.drift_detector, 'fit'):
            self.drift_detector.fit(new_data.cpu(), new_labels.cpu())

        # 4-5. Mini-batch training over ALL new_data for n_epochs
        self.model.train()
        N = len(new_data)
        total_loss = 0.0
        n_steps = 0

        for _ in range(self.n_epochs):
            perm = torch.randperm(N, device=self.device)
            shuffled_data = new_data[perm]
            shuffled_labels = new_labels[perm]

            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                batch_data = shuffled_data[start:end]
                batch_labels = shuffled_labels[start:end]

                replay_batch = None
                if self.memory_manager.size() > 0:
                    r_data, r_labels = self.memory_manager.get_replay_batch(
                        self.batch_size)
                    if r_data is not None:
                        replay_batch = (r_data.to(self.device),
                                        r_labels.to(self.device))

                self.optimizer.zero_grad()
                loss = self.anti_forgetting.compute_loss(
                    self.model, (batch_data, batch_labels), replay_batch)
                loss.backward()

                if hasattr(self.anti_forgetting, 'project_gradients'):
                    self.anti_forgetting.project_gradients(self.model)

                self.optimizer.step()
                total_loss += loss.item()
                n_steps += 1

        # 6. Task-end hook (once per update call, not per batch)
        self.anti_forgetting.on_task_end(self.model)

        return {
            'loss': total_loss / max(n_steps, 1),
            'drift': drift_detected,
            'drift_score': drift_score,
            'round': self._round,
        }

    # ------------------------------------------------------------------
    def fit_anomaly_scorer(self, normal_data: torch.Tensor) -> None:
        """Fit the anomaly scorer on normal (inlier) data and set threshold.

        Trains the scorer on normal samples, then sets the decision threshold
        to the 95th percentile of normal scores. This ensures the threshold is
        data-adaptive rather than a fixed constant (which would cause F1=0 for
        reconstruction-error scorers like PCA and DeepSVDD).

        Args:
            normal_data: Normal samples. Shape (N, D).
        """
        normal_data = normal_data.to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(normal_data)
            encoded = out[0] if isinstance(out, (tuple, list)) else out
        encoded_cpu = encoded.cpu()
        self.anomaly_scorer.fit(encoded_cpu)
        # 정상 데이터 점수의 95th percentile을 임계값으로 자동 설정 (B1 버그 수정)
        # 고정값 0.5는 PCA/DeepSVDD 등 재구성 오차 기반 scorer에서 F1=0을 유발했음
        with torch.no_grad():
            val_scores = self.anomaly_scorer.score(encoded_cpu)
        if len(val_scores) > 0:
            self._anomaly_threshold = float(
                torch.quantile(val_scores.float(), 0.95).item()
            )

    # ------------------------------------------------------------------
    def infer(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Score and classify a batch of samples.

        Args:
            data: Input samples. Shape (N, D).

        Returns:
            Dict with 'scores' (float, shape N) and 'predictions' (long, shape N).
        """
        data = data.to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(data)
            encoded = out[0] if isinstance(out, (tuple, list)) else out
        encoded = encoded.cpu()
        scores = self.anomaly_scorer.score(encoded)
        preds = self.anomaly_scorer.predict(encoded, self._anomaly_threshold)
        return {'scores': scores, 'predictions': preds}

    # ------------------------------------------------------------------
    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """Return model state dict for FL aggregation.

        Returns:
            state_dict compatible with nn.Module.load_state_dict().
        """
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def load_model_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load a global model received from the FL server.

        Args:
            state_dict: Model weights dict from the server.
        """
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in state_dict.items()})

    def set_anomaly_threshold(self, threshold: float) -> None:
        """Update the decision threshold for anomaly classification.

        Args:
            threshold: New threshold value.
        """
        self._anomaly_threshold = threshold
