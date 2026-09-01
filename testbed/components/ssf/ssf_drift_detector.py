"""SSF DriftDetector — K-S 검정 기반 drift 탐지 (PRD 4절/12.2절).

SSF 원 논문 근거: SSF-Strategic-Selection-and-Forgetting/utils.py의
detect_drift()가 scipy.stats.ks_2samp(control, window)를 사용하고,
p_value < drift_threshold(기본 0.05)면 drift로 판정한다(ssf.py:49, utils.py:646-658).
UNSW 실험에서는 classifier logit 분포를 직접 비교한다(ssf.py:217-225) — 이
테스트베드의 표현 의존성 계약(12.1절)에서 SSFDriftDetector는 공유 표현 소비자
(uses_shared_representation=True)이므로, CLClient가 넘기는 현재 모델의 logit을
그대로 사용한다.

**인용 범위 주의(2026-09-01 보완)**: 위 "logit 비교" 근거는 SSF 원문의
UNSW-NB15 분기(ssf.py:217-225)다. NSL-KDD 분기(ssf.py:202-214)는 이와
달리 재구성 확률 비율(`pdf1_probe`/`pdf11_probe`)을 drift 신호로 쓴다 —
원문 자체가 데이터셋마다 다른 신호로 drift를 감지한다. 이 테스트베드는
공유 backbone 하나로 모든 데이터셋을 다루므로 logit 비교로 통일했다
(ssf_anti_forgetting.py 모듈 docstring에 이미 문서화된 것과 같은 종류의
"공유 backbone이 강제하는 데이터셋 간 절충").
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
