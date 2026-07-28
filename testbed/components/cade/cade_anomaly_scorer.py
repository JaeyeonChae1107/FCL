"""CADEMADScorer — CADE MAD 정규화 거리 기반 anomaly scorer (PRD 12.6절).

12.1절 참고 — 이 클래스는 CADEDriftDetector와 이름만 CADE를 공유할 뿐 완전히
별개의 클래스이며 상태를 공유하지 않는다. 이 스코어러는 (A) 공유 표현
소비자로, 메인 분류기(BaseCLModel)의 z 공간에서 직접 centroid/MAD를 계산한다.
fit()은 항상 정상(label=0) 데이터만 받으므로 centroid는 단일(정상) centroid다.

Threshold: Track A 원 논문 방식 그대로 — median(s_ref) + t_mad*MAD(s_ref)
(PRD 3.4/3.5절, CADE/cade/detect.py:91,150-158, 기본 t_mad=3.5).
"""

from typing import Optional

import torch

from testbed.base.anomaly_scorer import BaseAnomalyScorer


class CADEMADScorer(BaseAnomalyScorer):
    required_backbone = "classifier"

    def __init__(self, t_mad: float = 3.5):
        self.t_mad = t_mad
        self._centroid: Optional[torch.Tensor] = None
        self._median: Optional[torch.Tensor] = None
        self._mad: Optional[torch.Tensor] = None

    def fit(self, normal_data: torch.Tensor) -> None:
        if len(normal_data) == 0:
            return
        self._centroid = normal_data.mean(dim=0)
        dist = torch.norm(normal_data - self._centroid, dim=1)
        self._median = dist.median()
        mad = 1.4826 * (dist - self._median).abs().median()
        self._mad = mad if mad > 1e-8 else torch.tensor(1e-8, device=normal_data.device)

    def score(self, data: torch.Tensor) -> torch.Tensor:
        if self._centroid is None:
            return torch.zeros(len(data), device=data.device)
        dist = torch.norm(data - self._centroid, dim=1)
        return (dist - self._median).abs() / self._mad

    def compute_threshold(self, eval_scores: torch.Tensor,
                           eval_labels: Optional[torch.Tensor]) -> float:
        # Track A(cade_mad): eval_scores는 정상 참조 데이터의 score(s_ref).
        # eval_labels는 쓰지 않는다(통계적 threshold, 라벨 불필요 — PRD 3.5절).
        median = eval_scores.median()
        mad = 1.4826 * (eval_scores - median).abs().median()
        mad = mad if mad > 1e-8 else torch.tensor(1e-8, device=eval_scores.device)
        return float(median + self.t_mad * mad)
