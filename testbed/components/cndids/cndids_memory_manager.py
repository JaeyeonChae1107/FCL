"""CNDIDSMemoryManager — 정상(label=0) 전용 FIFO 버퍼 (PRD 4절/12.4절).

CND-IDS는 라벨-프리 접근이지만 "이 샘플이 정상이라고 알려진 것"이라는 정보는
이용한다(PRD 5.2절) — 이 메모리 매니저가 그 지점이다. 공격 라벨은 학습 손실에
쓰지 않지만(cndids_anti_forgetting.py), 버퍼에 넣을지 여부를 정할 때는
label==0 여부만 참조한다.
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
            excess = len(self._buf_data) - self.max_size
            self._buf_data = self._buf_data[excess:]
            self._buf_labels = self._buf_labels[excess:]

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
