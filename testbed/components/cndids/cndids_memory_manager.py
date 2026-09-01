"""CNDIDSMemoryManager — 정상(label=0) 전용 FIFO 버퍼 (PRD 4절/12.4절).

주의(2026-08-12 정정): CND-IDS의 실제 제안 방법(`CND_IDS.py`, 196줄)에는
메모리/리플레이 버퍼가 전혀 없다 — "CND-IDS 원문 근거"로 이 컴포넌트를
정당화할 수 없다. 정상 라벨만 필터링해 저장하는 이 클래스(max_size=1000
포함)는 실제로는 같은 저장소의 `CFE.py`(ADCN 베이스라인용 별도 피처
추출기, CND-IDS와 무관)의 `Memory(mode, 1000, ...)` 클래스와 닮아 있고,
그쪽조차 라벨 필터링 없이 저장한다. 즉 "정상 전용 필터링"은 CND-IDS
원문에서 가져온 것이 아니라 이 테스트베드가 memory_manager 슬롯에
cndids 조합을 채우기 위해 자체적으로 고안한 설계다(PRD 4절/12.4절 —
테스트베드 자체 문서, 논문 문서 아님). CND-IDS는 라벨-프리 접근이라는
점(PRD 5.2절)과는 별개로, 공격 라벨을 학습 손실에 쓰지 않는 것은
cndids_anti_forgetting.py 쪽 설계이고, 이 클래스가 버퍼 편입 여부를
label==0으로 거르는 것은 그와 독립적인 테스트베드 자체 판단이다.

**2026-08-26 발견·수정 — "FIFO"가 사실상 "직전 라운드만 기억"으로
퇴화**: Track B는 label_budget 없이 experience 전체를 쓰므로(`cl_client.py`
참고) 한 라운드의 정상 표본 수(NSL-KDD 13468건+)가 `max_size`(1000)를
이미 훨씬 넘는다 — 즉 매 라운드 `update()`가 끝나면 버퍼는 100% "이번
라운드에서 막 들어온 것"이고, 그 이전 라운드의 표본은 전부 밀려난다
(FIFO 자체는 정의대로 정확히 동작한 것이지만, "여러 라운드에 걸친
리플레이로 망각을 완화한다"는 memory_manager 슬롯의 취지에서 보면 사실상
"직전 한 라운드만 리플레이"로 퇴화한 것 — `cndids_anti_forgetting.py`의
"2026-08-26" 절에서 발견한 것과 같은 근본 원인). 여러 라운드에 걸친 표본이
실제로 섞여 남도록 꼬리 슬라이싱 대신 무작위 표본으로 바꾼다.
"""

from typing import Optional, Tuple

import torch

from testbed.base.memory_manager import BaseMemoryManager


class CNDIDSMemoryManager(BaseMemoryManager):
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._buf_data: Optional[torch.Tensor] = None
        self._buf_labels: Optional[torch.Tensor] = None

    def update(self, selected_data: torch.Tensor, selected_labels: torch.Tensor,
               drift_detected: bool = False) -> None:
        normal_mask = selected_labels == 0
        if normal_mask.sum() == 0:
            return
        normal_data = selected_data[normal_mask]
        normal_labels = selected_labels[normal_mask]

        if self._buf_data is None:
            self._buf_data, self._buf_labels = normal_data.clone(), normal_labels.clone()
        else:
            self._buf_data = torch.cat([self._buf_data, normal_data], dim=0)
            self._buf_labels = torch.cat([self._buf_labels, normal_labels], dim=0)

        if len(self._buf_data) > self.max_size:
            # 2026-08-26 발견·수정 — 꼬리 슬라이싱은 라운드당 표본 수가
            # max_size를 넘으면(Track B는 항상 그렇다) 매번 100% 이번
            # 라운드 것으로 덮어써진다(위 모듈 docstring 참고). 무작위
            # 표본으로 바꿔 여러 라운드의 표본이 비율대로 실제로 섞여
            # 남게 한다.
            perm = torch.randperm(len(self._buf_data), device=self._buf_data.device)
            keep = perm[:self.max_size]
            self._buf_data = self._buf_data[keep]
            self._buf_labels = self._buf_labels[keep]

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
