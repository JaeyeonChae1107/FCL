"""SSF MemoryManager — Strategic Forgetting (PRD 4절/12.4절).

SSF 원 논문 근거: utils.py의 select_and_update_representative_samples()가
비대표(M_c_bin==0) 샘플을 우선 제거하고, 부족하면 대표 샘플 중 가장 낮은
M_c 점수부터 추가 제거한다(utils.py:192-257,259-388) — FIFO가 아니라
"대표성 낮은 순 우선 제거"가 핵심 성질이다. drift 시에는 더 공격적으로
교체한다(utils.py:259-388의 별도 drift 분기).

이 테스트베드의 목적은 원본 코드를 그대로 재현하는 게 아니라, "drift_detector가
만드는 신호(drift_detected)를 실제로 소비하는 컴포넌트가 있어야 drift_detector
슬롯(none/ssf/cade)이 조합 비교에서 의미 있는 축이 된다"는 것이다(사용자
지시 — 이 소비 관계가 없는 (sample_selector, memory_manager) 조합은
common/compatibility.py의 TRACK_A_DD_ACTIVE_SS_MM에서 애초에 제외했다).
원문의 마스크 점수(M_c)를 그대로 전달받지 않으므로, 매 update 시점에
(기존 버퍼 + 신규 선택 샘플)을 합친 뒤 "자기 클래스 centroid까지의 거리가
가까울수록 대표적"이라는 대표성 개념으로 점수를 재계산해 하위 점수부터
제거하되, drift_detected=True인 라운드에는 유지 개수를
max_size*drift_retention_ratio로 줄여 더 공격적으로 교체한다 — "평시엔
비대표만 솎아내고, drift 시엔 과거를 더 강하게 잊는다"는 원문의 정성적 성질을
보존한 것이다.
"""

from typing import Optional, Tuple

import torch

from testbed.base.memory_manager import BaseMemoryManager


class SSFMemoryManager(BaseMemoryManager):
    def __init__(self, max_size: int = 1000, drift_retention_ratio: float = 0.7):
        self.max_size = max_size
        # drift 감지 시 버퍼를 이 비율만큼만 유지해 더 공격적으로 교체한다.
        # PRD 6절 "테스트베드 기본값" 성격 — 특정 논문 수치가 아니라, drift_detector
        # 슬롯이 실제로 결과에 영향을 주도록 만드는 최소한의 원칙적 조정.
        self.drift_retention_ratio = drift_retention_ratio
        self._buf_data: Optional[torch.Tensor] = None
        self._buf_labels: Optional[torch.Tensor] = None

    def update(self, selected_data: torch.Tensor, selected_labels: torch.Tensor,
               drift_detected: bool = False) -> None:
        if self._buf_data is None:
            combined_data, combined_labels = selected_data.clone(), selected_labels.clone()
        else:
            combined_data = torch.cat([self._buf_data, selected_data], dim=0)
            combined_labels = torch.cat([self._buf_labels, selected_labels], dim=0)

        keep_size = self.max_size
        if drift_detected:
            keep_size = max(1, int(self.max_size * self.drift_retention_ratio))

        if len(combined_data) <= keep_size:
            self._buf_data, self._buf_labels = combined_data, combined_labels
            return

        scores = self._representativeness(combined_data, combined_labels)
        keep_idx = torch.topk(scores, keep_size).indices
        self._buf_data = combined_data[keep_idx]
        self._buf_labels = combined_labels[keep_idx]

    @staticmethod
    def _representativeness(data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        scores = torch.zeros(len(data), device=data.device)
        for c in labels.unique():
            mask = labels == c
            centroid = data[mask].mean(dim=0, keepdim=True)
            dist = torch.norm(data[mask] - centroid, dim=1)
            scores[mask] = -dist  # 가까울수록(비대표성 낮을수록) 높은 점수
        return scores

    def get_replay_batch(self, batch_size: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self._buf_data is None:
            return None, None
        n = min(batch_size, len(self._buf_data))
        idx = torch.randperm(len(self._buf_data), device=self._buf_data.device)[:n]
        return self._buf_data[idx], self._buf_labels[idx]

    def get_buffer(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._buf_data, self._buf_labels

    def size(self) -> int:
        return 0 if self._buf_data is None else len(self._buf_data)
