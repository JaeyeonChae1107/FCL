"""SSF DriftDetector — K-S 검정 기반 drift 탐지 (PRD 4절/12.2절).

SSF 원 논문 근거: SSF-Strategic-Selection-and-Forgetting/utils.py의
detect_drift()가 scipy.stats.ks_2samp(control, window)를 사용하고,
p_value < drift_threshold(기본 0.05)면 drift로 판정한다(ssf.py:49, utils.py:646-658).
UNSW 실험에서는 classifier logit 분포를 직접 비교한다(ssf.py:217-225) — 이
테스트베드의 표현 의존성 계약(12.1절)에서 SSFDriftDetector는 공유 표현 소비자
(uses_shared_representation=True)이므로, CLClient가 넘기는 현재 모델의 logit을
그대로 사용한다.
"""

from typing import Optional

import torch
from scipy.stats import ks_2samp

from testbed.base.drift_detector import BaseDriftDetector


class SSFDriftDetector(BaseDriftDetector):
    uses_shared_representation = True

    def __init__(self, drift_threshold: float = 0.05):
        self.drift_threshold = drift_threshold

    def detect(self, new_data: torch.Tensor, buf_ref: Optional[torch.Tensor]) -> bool:
        if buf_ref is None or len(buf_ref) == 0 or len(new_data) == 0:
            return False
        _, p_value = self._ks(new_data, buf_ref)
        return bool(p_value < self.drift_threshold)

    def get_drift_score(self, new_data: torch.Tensor, buf_ref: Optional[torch.Tensor]) -> float:
        if buf_ref is None or len(buf_ref) == 0 or len(new_data) == 0:
            return 0.0
        stat, _ = self._ks(new_data, buf_ref)
        return float(stat)

    @staticmethod
    def _ks(new_data: torch.Tensor, buf_ref: torch.Tensor):
        a = new_data.detach().cpu().reshape(-1).numpy()
        b = buf_ref.detach().cpu().reshape(-1).numpy()
        result = ks_2samp(a, b)
        return result.statistic, result.pvalue
