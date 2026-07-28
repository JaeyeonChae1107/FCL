"""CADEDriftDetector — 독립 표현 소유자 (PRD 4절/12.1절/12.2절).

CADE 원 논문 근거: 클래스별 centroid=학습셋 latent 평균(CADE/cade/detect.py:62),
MAD=1.4826*median(|d-median(d)|)(detect.py:150-158), 샘플의 MAD 정규화 거리
A(x,i)=|‖z_x-centroid_i‖-median(dis_i)|/mad_i(detect.py:91), 최소값이
T_MAD(기본 3.5, utils.py:77-78)를 넘으면 drift로 판정(detect.py:97-104).

12.1절 — uses_shared_representation=False. CLClient는 메인 모델의 z가 아니라
원본 data를 그대로 넘긴다. fit()은 자기 소유의 ContrastiveAutoEncoder를
직접 학습시킨다(PRD 13절 step 3d — 라벨이 공개된 selected_data로만 호출).

**GPU 이식성**: 이 컴포넌트는 메인 모델(BaseCLModel)과 별개로 자기 소유의
`nn.Module`(ContrastiveAutoEncoder)을 갖는 유일한 컴포넌트다 — CLClient는
`self.model`만 `.to(device)`로 옮기므로, 이 사설 encoder는 CLClient가 명시적으로
`to(device)`를 호출해줘야 올바른 디바이스로 옮겨간다(`pipeline/cl_client.py`
참고, 다른 컴포넌트에는 없는 훅).
"""

from typing import Dict, Optional

import torch

from testbed.base.drift_detector import BaseDriftDetector
from testbed.components.cade.contrastive_ae import ContrastiveAutoEncoder, train_step


class CADEDriftDetector(BaseDriftDetector):
    uses_shared_representation = False

    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 32,
                 t_mad: float = 3.5, contrastive_margin: float = 10.0,
                 contrastive_lambda: float = 0.1, encoder_epochs: int = 5,
                 encoder_lr: float = 1e-3):
        self.t_mad = t_mad
        self.margin = contrastive_margin
        self.lam = contrastive_lambda
        self.epochs = encoder_epochs
        self._device = torch.device("cpu")
        self._encoder = ContrastiveAutoEncoder(input_dim, hidden_dim, latent_dim)
        self._optimizer = torch.optim.Adam(self._encoder.parameters(), lr=encoder_lr)
        self._centroids: Dict[int, torch.Tensor] = {}
        self._median: Dict[int, torch.Tensor] = {}
        self._mad: Dict[int, torch.Tensor] = {}

    def to(self, device) -> "CADEDriftDetector":
        """CLClient가 생성 직후 호출하는 선택적 훅 — 사설 encoder를 메인
        모델과 같은 디바이스로 옮긴다(`hasattr(component, 'to')` 패턴,
        `NoAnomalyScorer.set_model` 등과 같은 방식)."""
        self._device = torch.device(device)
        self._encoder.to(self._device)
        return self

    def fit(self, data: torch.Tensor, labels: torch.Tensor) -> None:
        if len(data) < 2:
            return
        for _ in range(self.epochs):
            train_step(self._encoder, self._optimizer, data, labels, self.margin, self.lam)

        self._encoder.eval()
        with torch.no_grad():
            z, _ = self._encoder(data)
        for c in labels.unique():
            mask = labels == c
            if mask.sum() == 0:
                continue
            centroid = z[mask].mean(dim=0)
            dist = torch.norm(z[mask] - centroid, dim=1)
            median = dist.median()
            mad = 1.4826 * (dist - median).abs().median()
            key = int(c.item())
            self._centroids[key] = centroid
            self._median[key] = median
            self._mad[key] = mad if mad > 1e-8 else torch.tensor(1e-8, device=self._device)

    def _min_anomaly_score(self, data: torch.Tensor) -> Optional[torch.Tensor]:
        if not self._centroids:
            return None
        self._encoder.eval()
        with torch.no_grad():
            z, _ = self._encoder(data)
        scores = []
        for c, centroid in self._centroids.items():
            dist = torch.norm(z - centroid, dim=1)
            anomaly = (dist - self._median[c]).abs() / self._mad[c]
            scores.append(anomaly)
        stacked = torch.stack(scores, dim=1)
        min_scores, _ = stacked.min(dim=1)
        return min_scores

    def detect(self, new_data: torch.Tensor, buf_ref: Optional[torch.Tensor]) -> bool:
        if buf_ref is None:
            return False
        min_scores = self._min_anomaly_score(new_data)
        if min_scores is None:
            return False
        return bool((min_scores > self.t_mad).float().mean().item() > 0.5)

    def get_drift_score(self, new_data: torch.Tensor, buf_ref: Optional[torch.Tensor]) -> float:
        if buf_ref is None:
            return 0.0
        min_scores = self._min_anomaly_score(new_data)
        if min_scores is None:
            return 0.0
        return float(min_scores.mean().item())
