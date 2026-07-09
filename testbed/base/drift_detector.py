from abc import ABC, abstractmethod
from typing import Optional
import torch


class BaseDriftDetector(ABC):
    """파이프라인 Stage 1 — Concept Drift 탐지기 추상 기반 클래스.

    메모리 버퍼(이전 분포)와 새 배치(현재 분포)를 비교하여 분포 변화를 감지한다.
    반드시 memory_manager.update() 이전에 호출되어 이전 분포를 참조해야 한다.

    구현 시 보장사항:
    - memory_buffer가 None이면 항상 detect()=False, get_drift_score()=0.0 반환
    - get_drift_score() 반환값은 0 이상의 float (normalize 권장, 범위 [0, 1])

    등록 키: 'none' | 'ssf' | 'cade' | 'ddm'
    """

    @abstractmethod
    def detect(self, new_data: torch.Tensor,
               memory_buffer: Optional[torch.Tensor]) -> bool:
        """Detect whether distribution drift has occurred.

        Args:
            new_data: Incoming data batch. Shape (N, D).
            memory_buffer: Reference distribution from memory. Shape (M, D) or None.

        Returns:
            True if drift is detected, False otherwise.

        Raises:
            ValueError: If new_data is empty or has wrong shape.
        """

    @abstractmethod
    def get_drift_score(self, new_data: torch.Tensor,
                        memory_buffer: Optional[torch.Tensor]) -> float:
        """Compute a scalar drift intensity score.

        Args:
            new_data: Incoming data batch. Shape (N, D).
            memory_buffer: Reference distribution. Shape (M, D) or None.

        Returns:
            Drift score as a float. Higher values indicate stronger drift.
        """

    def reset(self):
        """Reset internal state. Override if the detector is stateful.

        Returns:
            None
        """
        pass
