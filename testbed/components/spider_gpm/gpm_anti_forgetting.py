"""GPM (Gradient Projection Memory) anti-forgetting.

SPIDER 저장소 코드를 확보하지 못해(testbed/docs/metric_justification.md 참고),
GPM 원 논문(Saha, Garg, Roy, "Gradient Projection Memory for Continual
Learning", ICLR 2021)을 근거로 이 테스트베드의 BaseAntiForgetting 계약
(PRD 12.5절)에 맞춰 작성했다. 2026-07-30 최종 재검토에서 GPM 저자 공식
코드(https://github.com/sahagobinda/GPM, `main_pmnist.py`)를 실제로 찾아
대조했고, 한 번은 문자 그대로 맞췄다가 실측 회귀를 발견해 다시 되돌렸다 —
`_compute_basis()`의 주석에 그 경위가 자세히 남아있다. 최종적으로 아래
1·4번은 공식 코드와 일치하고, 2·3번은 의도적으로 다르다:

핵심 알고리즘 (공식 코드 `main_pmnist.py` 대조 완료):
  1. 태스크(experience) 종료 시(on_task_end), 그 태스크에서 실제로 학습에
     쓰인 selected_data로 각 Linear 레이어의 입력 activation 행렬을 수집한다.
  2. 레이어별 activation 행렬에 SVD를 적용해 누적 에너지 비율이
     activation_threshold 이상이 되는 최소 개수(+1 여유)의 우특이벡터를
     이번 태스크의 기저로 채택한다. **공식 코드는 평균을 빼지 않은 원본
     activation에 바로 SVD를 적용하고 `+1` 여유도 없다.** 문자 그대로
     맞춰봤더니 NSL-KDD에서 `af=gpm+as=cade_mad` 일부 조합(ss=ssf 계열)이
     exp1에서 분류기가 완전히 한 클래스로 퇴화하는 회귀가 실측으로
     확인됐다(30개 gpm 조합 전수검사: nsl-kdd 2/30 실패, unsw-nb15 0/30) —
     평균 중심화와 `+1` 여유 중 어느 하나만 복원해도 통과했다. 이 얕은
     은닉층 1개짜리 공유 아키텍처에서는 공식 코드의 문자 그대로의 계산이
     기저를 지나치게 작게(또는 ReLU의 항상-양수 편향 방향으로) 만들어
     분류기 학습에 필요한 그래디언트까지 사영으로 지워버리는 경계 사례를
     만든다 — 그래서 평균 중심화와 `+1` 여유 둘 다 원래대로 유지한다.
  3. 기존에 저장된 기저와 새 기저를 concat 후 QR 분해로 재직교화해 누적
     기저(GPM memory)를 갱신한다. **공식 코드는 `np.hstack`으로 이어붙이기만
     하고 QR 재직교화를 하지 않는다** — 이 테스트베드는 5개 experience에
     걸쳐 기저가 계속 누적되는데, 서로 다른 태스크의 SVD 결과를 단순히
     이어붙이기만 하면(태스크 간 직교성이 보장되지 않아) `basis @ basis.T`가
     더 이상 참된 직교 사영행렬이 되지 못하는 문제가 생길 수 있어 QR
     재직교화를 의도적으로 유지한다.
  4. 이후 태스크 학습 시 backward() 직후 project_gradients()가 각 레이어
     가중치의 gradient에서 누적 기저 방향 성분을 제거한다:
       grad_proj = grad - grad @ basis @ basis.T
     (공식 코드: `Uf = feature_list[i] @ feature_list[i].T`을 미리 계산해
     동일한 사영을 수행 — 이 부분은 일치한다.)

**요약**: 이 테스트베드의 목적(0절 — 논문 재현이 아니라 재조합·비교)상,
"공식 코드와 문자 그대로 같다"보다 "조합이 실제로 붕괴하지 않고 작동한다"를
우선했다. 공식 코드와 다른 지점(2·3번)은 전부 실측 근거와 함께 정직하게
기록해뒀다 — 몰라서 다른 게 아니라 알고도 다르게 한 것이다.

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
        # 2026-07-30: 공식 코드(main_pmnist.py, 평균 미중심화 + `+1` 없는
        # r=sum(cumsum<threshold))에 정확히 맞췄다가, 실측으로 회귀를
        # 발견해 되돌렸다 — NSL-KDD에서 af=gpm+as=cade_mad 조합 중 일부
        # (ss=ssf/mm=ssf 또는 spider, dd=ssf 또는 cade)가 exp1에서 예측이
        # 완전히 한 클래스로 퇴화했다(30개 gpm 조합 전수 검사로 확인,
        # unsw-nb15는 0/30, nsl-kdd는 2/30 실패). 원인 분리 실험 결과 두
        # 변경(중심화 제거, +1 제거) 중 **어느 한쪽만 되돌려도** 통과했다 —
        # 즉 공식 코드가 이 테스트베드의 (얕은 은닉층 1개 + 작은 분류기
        # head) 아키텍처에서는 기저가 지나치게 작아지거나(또는 ReLU 출력의
        # 항상-양수 편향 방향을 그대로 기저로 채택해) 분류기 학습에 필요한
        # 그래디언트 방향까지 사영으로 제거해버리는 경계 사례를 만든다.
        # 공식 코드를 문자 그대로 재현하는 것보다 "조합이 실제로 붕괴하지
        # 않고 작동하는 것"이 이 테스트베드의 목적(0절 — 재현이 아니라
        # 재조합·비교)에 더 부합하므로, 평균 중심화 + `+1` 여유 둘 다
        # 원래대로 유지한다(공식 코드와 다르다는 점만 정직하게 기록).
        centered = activation - activation.mean(dim=0, keepdim=True)
        _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        energy = S ** 2
        cumulative = torch.cumsum(energy, dim=0) / energy.sum().clamp(min=1e-10)
        k = int((cumulative < self.activation_threshold).sum().item()) + 1
        k = max(1, min(k, Vh.shape[0]))
        return Vh[:k].T
