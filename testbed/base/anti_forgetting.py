from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch


class BaseAntiForgetting(ABC):
    """파이프라인 Stage 5 — Catastrophic Forgetting 방지 전략 추상 기반 클래스.

    new_batch(현재 태스크)와 replay_batch(메모리 버퍼)를 결합하여 이전 태스크 망각을
    방지하는 스칼라 손실을 계산한다.

    학습 순서 (CLClient.update() 내):
      1. compute_loss(model, new_batch, replay_batch) → loss
      2. loss.backward()
      3. [선택] project_gradients(model)  ← GPM 전용, hasattr 체크 후 호출
      4. optimizer.step()
      5. on_task_end(model)  ← 교사 모델 스냅샷, 메모리 업데이트 등

    replay_batch=None 처리:
    - SSFAntiForgetting: LwF 생략, InfoNCE만 계산 (drift 모드로 간주)
    - CNDIDSAntiForgetting / CFEExtractor / GPMAntiForgetting: 미사용 (LwF 방식)

    등록 키: 'none' | 'lwf_ssf' | 'cfe' | 'cndids' | 'gpm'
    """

    @abstractmethod
    def compute_loss(self,
                     model: torch.nn.Module,
                     new_batch: Tuple[torch.Tensor, torch.Tensor],
                     replay_batch: Optional[Tuple[torch.Tensor, torch.Tensor]],
                     old_model: Optional[torch.nn.Module] = None) -> torch.Tensor:
        """Compute the total training loss for one step.

        Args:
            model: The current (student) model being trained.
            new_batch: Tuple (data, labels) for the current new mini-batch.
            replay_batch: Tuple (data, labels) sampled from the replay buffer,
                          or None if the buffer is empty.
            old_model: Frozen teacher model snapshot, or None.

        Returns:
            Scalar loss tensor with requires_grad=True.

        Raises:
            ValueError: If new_batch contains tensors of mismatched shapes.
        """

    def on_task_end(self, model: torch.nn.Module) -> None:
        """Hook called after each task / round ends.

        Override to implement teacher-model updates, gradient projection
        memory updates, etc.

        Args:
            model: The current model after the task.

        Returns:
            None
        """
        pass
