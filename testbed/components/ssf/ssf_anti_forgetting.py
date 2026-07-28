"""SSF AntiForgetting — LwF 스타일 distillation + task loss (PRD 4절/12.5절).

SSF 원 논문 근거: ssf.py의 distillation은 MSE(현재 출력, teacher 출력)이며
(reconstruction/classifier 출력에 직접 적용, 온도 스케일링 없음), 총 손실은
task_loss + lwf_lambda * distillation_loss (ssf.py:45,292-334, 기본
lwf_lambda=0.5). Teacher는 이전 태스크 종료 시점 모델 스냅샷이다
(ssf.py:149,336, on_task_end에서 갱신).
"""

import copy
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from testbed.base.anti_forgetting import BaseAntiForgetting
from testbed.base.models import BaseCLModel


class SSFAntiForgetting(BaseAntiForgetting):
    backbone_type = "classifier"

    def __init__(self, lwf_lambda: float = 0.5):
        self.lwf_lambda = lwf_lambda
        self._teacher: Optional[BaseCLModel] = None

    def compute_loss(self, model: BaseCLModel,
                      new_batch: Tuple[torch.Tensor, torch.Tensor],
                      replay_batch: Optional[Tuple[torch.Tensor, torch.Tensor]]
                      ) -> torch.Tensor:
        data, labels = new_batch
        _, _, logit = model(data)
        loss = F.binary_cross_entropy_with_logits(logit.squeeze(-1), labels.float())

        if replay_batch is not None and replay_batch[0] is not None:
            r_data, r_labels = replay_batch
            _, _, r_logit = model(r_data)
            loss = loss + F.binary_cross_entropy_with_logits(
                r_logit.squeeze(-1), r_labels.float())

        # teacher가 없으면(experience 0) LwF 항을 생략한다 — replay_batch=None인
        # 경우와 무관하게, teacher 존재 여부가 유일한 게이팅 조건이다.
        if self._teacher is not None:
            with torch.no_grad():
                _, _, teacher_logit = self._teacher(data)
            distill = F.mse_loss(logit, teacher_logit)
            loss = loss + self.lwf_lambda * distill

        return loss

    def on_task_end(self, model: BaseCLModel) -> None:
        self._teacher = copy.deepcopy(model)
        self._teacher.eval()
        for p in self._teacher.parameters():
            p.requires_grad_(False)
