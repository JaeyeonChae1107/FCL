"""SSF의 공유 마스크(M_c/M_t) 계산 — 원문 `utils.py:96-190`
(`optimize_old_mask`/`optimize_new_mask`) 이식 (PRD 4절/12.3-12.4절, Item 1+2).

원문은 이 두 함수를 매 라운드 순서대로 호출해 M_c(버퍼 표본의 대표성
마스크)/M_t(유입 표본의 대표성 마스크)를 얻고, `sample_selector`(선택)와
`memory_manager`(버퍼 갱신) 양쪽이 같은 마스크를 공유해서 쓴다 — 이 파일이
그 계산 부분만 떼어낸 것이고, `CLClient._compute_ssf_masks()`가 라운드당
1회 호출한다.

- Stage 1(`optimize_old_mask`, init='0.5-1', lr=1.0): `M_c`(버퍼 크기)를
  100-step SGD로 최적화 — `M_c` 가중 히스토그램(10-bin, [0,1])이
  `treatment_res` 히스토그램에 맞춰지도록(KL divergence).
- Stage 2(`optimize_new_mask`, init='0-0.5', lr=50.0): `M_c` 고정, `M_t`
  (유입 데이터 크기)를 100-step SGD로 최적화 — (`M_c`가중 old + `M_t`가중
  new) 결합 히스토그램이 같은 `treatment_res` 타깃에 맞춰지도록.
- control_res/treatment_res는 `histc(..., min=0., max=1.)`가 전제하는 [0,1]
  구간 값이어야 한다 — 이 테스트베드의 공유 backbone은 raw logit을
  반환하므로, 호출부(`cl_client.py`)가 `torch.sigmoid()`로 감싼 값을
  넘긴다.

  **인용 범위 주의(2026-09-14 재검토로 발견, 이전 서술 정정)**: 위 "원문에서
  이미 sigmoid를 통과한 classifier 출력"이라는 근거는 SSF 원문의 UNSW-NB15
  분기(`ssf.py:217-225`, classifier가 `nn.Sigmoid()`로 끝남, `utils.py:
  53-56`)에만 해당한다. NSL-KDD 분기(`ssf.py:202-214`)는 `control_res`/
  `treatment_res`를 classifier sigmoid가 아니라 `evaluate()`가 계산하는
  두 가우시안 확률밀도 비율(`pdf1/(pdf1+pdf2)`)로 만든다 — classifier와
  전혀 무관한 신호다. 이 테스트베드는 `cl_client.py`가 데이터셋을 가리지
  않고 항상 `torch.sigmoid(logit)`을 넘기므로, 이 프로젝트의 실제 대상
  데이터셋인 NSL-KDD에서는 마스크 입력의 구성 방식 자체가 원문과 구조적으로
  다르다 — `ssf_drift_detector.py` 모듈 docstring이 드리프트 감지 신호에
  대해 이미 인정한 것과 정확히 같은 종류의 "공유 backbone이 강제하는
  데이터셋 간 절충"이며, 이 파일만 그 사실을 명시하지 않고 있었다.
- 원문의 `optimize_old_mask`가 계산만 하고 실제로는 안 쓰는 `control_hist`
  변수(사용되는 건 `treatment_hist`뿐)는 죽은 코드라 이식하지 않았다 —
  결과에 영향 없는 생략이다.

**버퍼가 비어있는 라운드(control_res=None, 원문에 없는 경계 케이스)**:
control_res/M_c를 길이 0인 텐서로 취급한다 — Stage 1은 자연히 건너뛰고
(빈 텐서에 대한 SGD 최적화가 무의미), Stage 2의 결합 히스토그램도 M_c
항이 전부 0이 되어 `treatment_res` 자신의 히스토그램만 타깃으로 최적화하는
것과 동일해진다(별도 분기 없이 텐서 연산 자체가 이 경계 케이스를 흡수).
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class SSFMaskContext:
    buf_data: Optional[torch.Tensor]
    control_res: torch.Tensor  # sigmoid(logit), 버퍼 없으면 길이 0
    M_c_bin: torch.Tensor      # 버퍼 없으면 길이 0
    new_data: torch.Tensor
    treatment_res: torch.Tensor  # sigmoid(logit)
    M_t_bin: torch.Tensor
    M_t_cont: torch.Tensor
    M_c_cont: torch.Tensor  # 버퍼 없으면 길이 0 — SSFMemoryManager가
                             # "대표 표본 중 최하위 M_c" 랭킹에 씀.
    # CLClient가 Step 3 선택 직후 채운다 — new_data 기준 이번 라운드
    # 선택된 인덱스(SSFMemoryManager가 "대표적이지만 미선택인 잔여"를
    # 가려낼 때 씀, ssf_memory_manager.py 참고).
    sel_idx: Optional[torch.Tensor] = None


def _optimize_old_mask(control_res: torch.Tensor, treatment_res: torch.Tensor,
                        device: torch.device, num_bins: int, lr: float, steps: int
                        ) -> torch.Tensor:
    """원문 `optimize_old_mask`(`utils.py:96-143`) 이식.

    **미고지 이탈 1건(2026-09-14 재검토로 발견)**: 아래 `F.kl_div((bin_obs_c
    + 1e-10).log(), ...)`의 `+ 1e-10`은 원문(`utils.py:139`, `bin_obs_c.log()`
    — epsilon 없음)에 없던 것이다. `bin_obs_c`의 어느 bin이 0이면
    `.log()`가 `-inf`가 되어 이후 backward가 NaN이 될 수 있는데, 원문은 이
    경우를 아예 다루지 않는다(반면 `_optimize_new_mask`/원문의 `optimize_
    new_mask`는 양쪽 다 `clamp(min=1e-10)`을 명시적으로 쓴다 — 원문 자체가
    old/new 두 함수 사이에서 이미 비대칭적이다). 이 epsilon을 없애 원문과
    완전히 똑같이 맞추면 실제 학습 중 특정 라운드에서 NaN으로 이어질 수
    있어(검증 없이 제거하는 게 더 위험) 그대로 유지한다 — 다만 지금까지
    이 사실 자체가 어디에도 적혀있지 않았다."""
    n = len(control_res)
    if n == 0:
        return control_res.new_zeros(0)
    M_c = torch.nn.Parameter(torch.rand(n, device=device) * 0.5 + 0.5)  # init '0.5-1'
    optimizer = torch.optim.SGD([M_c], lr=lr)
    delta = 1e-4
    bin_edges = torch.linspace(0., 1., num_bins + 1, device=device)
    treatment_hist = torch.histc(treatment_res, bins=num_bins, min=0., max=1.)
    n_treatment = len(treatment_res)
    for _ in range(steps):
        with torch.no_grad():
            M_c.clamp_(delta, 1 - delta)
        optimizer.zero_grad()
        bin_obs_c = torch.zeros(num_bins, device=device)
        bin_tgt_c = torch.zeros(num_bins, device=device)
        for i in range(num_bins):
            mask_c = (control_res >= bin_edges[i]) & (control_res < bin_edges[i + 1])
            bin_obs_c[i] = torch.sum(M_c * mask_c.float()) / torch.sum(M_c)
            bin_tgt_c[i] = treatment_hist[i] / n_treatment
        bin_obs_c = bin_obs_c / bin_obs_c.sum()
        bin_tgt_c = bin_tgt_c / bin_tgt_c.sum()
        loss = F.kl_div((bin_obs_c + 1e-10).log(), bin_tgt_c, reduction="sum")
        loss.backward()
        optimizer.step()
    return M_c.detach()


def _optimize_new_mask(control_res: torch.Tensor, treatment_res: torch.Tensor,
                        M_c: torch.Tensor, device: torch.device,
                        num_bins: int, lr: float, steps: int) -> torch.Tensor:
    n = len(treatment_res)
    M_t = torch.nn.Parameter(torch.rand(n, device=device) * 0.5)  # init '0-0.5'
    optimizer = torch.optim.SGD([M_t], lr=lr)
    delta = 1e-4
    bin_edges = torch.linspace(0., 1., num_bins + 1, device=device)
    treatment_hist = torch.histc(treatment_res, bins=num_bins, min=0., max=1.)
    n_treatment = len(treatment_res)
    for _ in range(steps):
        with torch.no_grad():
            M_t.clamp_(delta, 1 - delta)
        optimizer.zero_grad()
        bin_tgt_t = torch.zeros(num_bins, device=device)
        bin_combined = torch.zeros(num_bins, device=device)
        denom = torch.sum(M_t) + torch.sum(M_c)
        for i in range(num_bins):
            mask_c = (control_res >= bin_edges[i]) & (control_res < bin_edges[i + 1])
            mask_t = (treatment_res >= bin_edges[i]) & (treatment_res < bin_edges[i + 1])
            bin_tgt_t[i] = treatment_hist[i] / n_treatment
            bin_combined[i] = (torch.sum(M_t * mask_t.float())
                                + torch.sum(M_c * mask_c.float())) / denom
        bin_combined = torch.clamp(bin_combined / bin_combined.sum(), min=1e-10)
        bin_combined = bin_combined / bin_combined.sum()
        bin_tgt_t = torch.clamp(bin_tgt_t / bin_tgt_t.sum(), min=1e-10)
        bin_tgt_t = bin_tgt_t / bin_tgt_t.sum()
        loss = F.kl_div(bin_combined.log(), bin_tgt_t, reduction="sum")
        loss.backward()
        optimizer.step()
    return M_t.detach()


def optimize_ssf_masks(control_res: torch.Tensor, treatment_res: torch.Tensor,
                        device: torch.device, num_bins: int = 10,
                        old_lr: float = 1.0, new_lr: float = 50.0, steps: int = 100
                        ):
    """반환: (M_c_bin, M_t_bin, M_t_cont, M_c_cont). `control_res`가 길이
    0이면 Stage 1은 자동으로 건너뛰어진다(모듈 docstring 참고)."""
    M_c = _optimize_old_mask(control_res, treatment_res, device, num_bins, old_lr, steps)
    M_t = _optimize_new_mask(control_res, treatment_res, M_c, device, num_bins, new_lr, steps)
    M_c_bin = (M_c >= 0.5).float()
    M_t_bin = (M_t >= 0.5).float()
    return M_c_bin, M_t_bin, M_t, M_c
