"""GPM (Gradient Projection Memory) anti-forgetting.

SPIDER 저장소 코드를 확보하지 못해(testbed/docs/metric_justification.md 참고),
GPM 원 논문(Saha, Garg, Roy, "Gradient Projection Memory for Continual
Learning", ICLR 2021, https://openreview.net/forum?id=3AOj0RCNC2)의 알고리즘
설명을 직접 근거로 이 테스트베드의 BaseAntiForgetting 계약(PRD 12.5절)에 맞춰
새로 작성했다. 이전 testbed의 components/gpm/gpm_anti_forgetting.py는 참고하지
않았다(사용자 지시).

핵심 알고리즘 (GPM 원 논문 Algorithm 1):
  1. 태스크(experience) 종료 시(on_task_end), 그 태스크에서 실제로 학습에
     쓰인 selected_data로 각 Linear 레이어의 입력 activation 행렬을 수집한다.
  2. 레이어별 activation 행렬에 SVD를 적용해, 누적 에너지 비율이
     activation_threshold 이상이 되는 최소 개수의 우특이벡터를 이번 태스크의
     기저로 채택한다.
  3. 기존에 저장된 기저와 concat 후 QR 분해로 재직교화해 누적 기저(GPM
     memory)를 갱신한다.
  4. 이후 태스크 학습 시 backward() 직후 project_gradients()가 각 레이어
     가중치의 gradient에서 누적 기저 방향 성분을 제거한다:
       grad_proj = grad - grad @ basis @ basis.T

레지스트리 키는 "gpm" (PRD 4.1절 — 폴더는 components/spider_gpm/).
GPM은 명시적 replay/정규화 항 없이 gradient projection만으로 이전 태스크를
보호하므로(원 논문 설계), compute_loss는 task loss만 계산한다.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from testbed.base.anti_forgetting import BaseAntiForgetting
from testbed.base.models import BaseCLModel


class GPMAntiForgetting(BaseAntiForgetting):
    backbone_type = "classifier"

    def __init__(self, activation_threshold: float = 0.97):
        self.activation_threshold = activation_threshold
        self._basis: Dict[str, torch.Tensor] = {}
        self._pending_data: List[torch.Tensor] = []

    def compute_loss(self, model: BaseCLModel,
                      new_batch: Tuple[torch.Tensor, torch.Tensor],
                      replay_batch: Optional[Tuple[torch.Tensor, torch.Tensor]]
                      ) -> torch.Tensor:
        data, labels = new_batch
        self._pending_data.append(data.detach())
        _, _, logit = model(data)
        loss = F.binary_cross_entropy_with_logits(logit.squeeze(-1), labels.float())
        return loss

    def project_gradients(self, model: BaseCLModel) -> None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in self._basis:
                if module.weight.grad is None:
                    continue
                basis = self._basis[name].to(module.weight.device)
                grad = module.weight.grad
                proj = grad @ basis @ basis.T
                module.weight.grad = grad - proj

    def on_task_end(self, model: BaseCLModel) -> None:
        if not self._pending_data:
            return
        all_data = torch.cat(self._pending_data, dim=0)
        self._update_basis(model, all_data)
        self._pending_data = []

    def _update_basis(self, model: BaseCLModel, data: torch.Tensor) -> None:
        activations: Dict[str, torch.Tensor] = {}
        handles = []

        def make_hook(name):
            def hook(module, inp, out):
                activations[name] = inp[0].detach()
            return hook

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(make_hook(name)))

        model.eval()
        with torch.no_grad():
            model(data)
        for h in handles:
            h.remove()

        for name, act in activations.items():
            new_basis = self._compute_basis(act)
            if name in self._basis:
                combined = torch.cat([self._basis[name], new_basis], dim=1)
            else:
                combined = new_basis
            Q, _ = torch.linalg.qr(combined, mode="reduced")
            self._basis[name] = Q

    def _compute_basis(self, activation: torch.Tensor) -> torch.Tensor:
        centered = activation - activation.mean(dim=0, keepdim=True)
        _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        energy = S ** 2
        cumulative = torch.cumsum(energy, dim=0) / energy.sum().clamp(min=1e-10)
        k = int((cumulative < self.activation_threshold).sum().item()) + 1
        k = max(1, min(k, Vh.shape[0]))
        return Vh[:k].T
