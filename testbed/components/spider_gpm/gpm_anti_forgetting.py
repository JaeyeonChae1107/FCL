"""GPM (Gradient Projection Memory) anti-forgetting.

SPIDER 저장소 코드를 확보하지 못해(testbed/docs/metric_justification.md 참고),
GPM 원 논문(Saha, Garg, Roy, "Gradient Projection Memory for Continual
Learning", ICLR 2021)을 근거로 이 테스트베드의 BaseAntiForgetting 계약
(PRD 12.5절)에 맞춰 작성했다. 2026-07-30 최종 재검토에서 GPM 저자 공식
코드(https://github.com/sahagobinda/GPM, `main_pmnist.py`)를 실제로 찾아
대조했다 — 이전 기록은 "원 논문 Algorithm 1"이라고 표현했지만 실제로는
아래 두 지점에서 공식 코드와 달랐다:

핵심 알고리즘 (공식 코드 `main_pmnist.py` 대조 완료):
  1. 태스크(experience) 종료 시(on_task_end), 그 태스크에서 실제로 학습에
     쓰인 selected_data로 각 Linear 레이어의 입력 activation 행렬을 수집한다.
  2. 레이어별 activation 행렬에 SVD를 적용해(공식 코드: **평균을 빼지 않은
     원본 activation**에 바로 `np.linalg.svd` — 이전 기록은 "PCA처럼
     평균을 뺀 뒤 SVD"라고 잘못 구현했었다. 2026-07-30에 평균-빼기를
     제거해 공식 코드와 동일하게 맞췄다), 누적 에너지 비율이
     activation_threshold **미만**인 개수만큼의 좌특이벡터를 이번 태스크의
     기저로 채택한다(`r = sum(cumsum(S**2/sum) < threshold)`, `U[:,0:r]`
     — 이전 기록은 `+1`을 더해 공식 코드보다 벡터 하나를 더 채택하고
     있었다. 2026-07-30에 제거해 동일하게 맞췄다).
  3. **공식 코드는 기존 기저와 새 기저를 `np.hstack`으로 이어붙이기만 하고
     차원이 넘치면 그냥 앞부분만 잘라낼 뿐, QR 재직교화를 하지 않는다.**
     이 테스트베드는 5개 experience에 걸쳐 기저가 계속 누적되는데, 서로
     다른 태스크의 SVD 결과를 단순히 이어붙이기만 하면(태스크 간 직교성이
     보장되지 않아) `basis @ basis.T`가 더 이상 참된 직교 사영행렬이 되지
     못하는 문제가 생길 수 있다 — 그래서 이 부분만은 공식 코드를 그대로
     따르지 않고 QR 분해로 재직교화하는 것을 의도적으로 유지한다(공식
     코드에 없는 이 테스트베드의 보정 — "원 논문 그대로"라고 오기하지
     않도록 정정).
  4. 이후 태스크 학습 시 backward() 직후 project_gradients()가 각 레이어
     가중치의 gradient에서 누적 기저 방향 성분을 제거한다:
       grad_proj = grad - grad @ basis @ basis.T
     (공식 코드: `Uf = feature_list[i] @ feature_list[i].T`을 미리 계산해
     동일한 사영을 수행 — 이 부분은 이미 일치했다.)

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
        # GPM 공식 코드(main_pmnist.py)는 평균을 빼지 않은 원본 activation에
        # 바로 SVD를 적용한다 — PCA식 중심화를 하지 않는다.
        _, S, Vh = torch.linalg.svd(activation, full_matrices=False)
        energy = S ** 2
        cumulative = torch.cumsum(energy, dim=0) / energy.sum().clamp(min=1e-10)
        # 공식 코드: r = sum(cumsum(energy_ratio) < threshold); U[:, 0:r].
        k = int((cumulative < self.activation_threshold).sum().item())
        k = max(1, min(k, Vh.shape[0]))
        return Vh[:k].T
