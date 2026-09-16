"""SSF DriftDetector — K-S 검정 기반 drift 탐지 (PRD 4절/12.2절).

SSF 원 논문 근거: SSF-Strategic-Selection-and-Forgetting/utils.py의
detect_drift()가 scipy.stats.ks_2samp(control, window)를 사용하고,
p_value < drift_threshold(기본 0.05)면 drift로 판정한다(ssf.py:49, utils.py:646-658).
UNSW 실험에서는 classifier logit 분포를 직접 비교한다(ssf.py:217-225) — 이
테스트베드의 표현 의존성 계약(12.1절)에서 SSFDriftDetector는 공유 표현 소비자
(uses_shared_representation=True)이므로, CLClient가 넘기는 현재 모델의 logit을
그대로 사용한다.

**인용 범위 주의**: 위 "logit 비교" 근거는 SSF 원문의
UNSW-NB15 분기(ssf.py:217-225)다. NSL-KDD 분기(ssf.py:202-214)는 이와
달리 재구성 확률 비율(`pdf1_probe`/`pdf11_probe`)을 drift 신호로 쓴다 —
원문 자체가 데이터셋마다 다른 신호로 drift를 감지한다. 이 테스트베드는
공유 backbone 하나로 모든 데이터셋을 다루므로 logit 비교로 통일했다
(ssf_anti_forgetting.py 모듈 docstring에 이미 문서화된 것과 같은 종류의
"공유 backbone이 강제하는 데이터셋 간 절충").

**서브윈도우 분할(원문 `utils.py:644-658` `detect_drift()`) 의도적 미이식
(2026-09-14 재검토)**: 원문은 `new_data`를 `window_size` 단위로 잘라
여러 서브윈도우 각각에 대해 KS 검정을 돌리고, 그중 하나라도 drift로
판정되면 즉시 True를 반환한다(OR 판정) — 마지막 조각이 `window_size`보다
작으면 그 조각은 검정 없이 루프를 끝낸다(`break`). 그런데 원문에서
`window_size`로 넘기는 값(`sample_interval`, 기본 20000)은 애초에 원문이
"라운드" 자체를 만드는 데 쓰는 고정 슬라이싱 크기와 동일하다
(`ssf.py:194-200` — 전체 스트림을 20000행씩 순서대로 잘라 라운드를
만듦). 즉 `detect_drift()`에 들어오는 `new_data`는 거의 항상 정확히
`window_size`와 같은 길이라 이 for 루프는 사실상 매 라운드 1회만
실행되고(원문이 의도한 "여러 윈도우 중 하나라도" 판정이 실질적으로
거의 발동하지 않음), `break`가 걸리는 유일한 경우는 "전체 스트림의
맨 마지막 라운드가 20000행보다 적게 남았을 때"뿐이다 — 통계적 타당성을
근거로 한 설계가 아니라 "고정 크기 스트림 슬라이싱"이라는 원문 고유의
라운드 구성 방식이 낳은 부산물이다. 이 테스트베드는 라운드를 고정
행 수 슬라이싱이 아니라 카테고리 라운드로빈 + i-Blurry로 구성하므로
`window_size`에 대응하는 값 자체가 존재하지 않는다 — 억지로 하나
정하면 근거 없는 임의값이 되므로, 이 서브윈도우/스킵 로직은 이식하지
않고 라운드 전체를 대상으로 한 번만 검정한다(현재 구현 그대로 유지).
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
