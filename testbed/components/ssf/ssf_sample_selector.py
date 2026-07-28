"""SSF SampleSelector — KL-divergence 기반 마스크 최적화 (PRD 4절/12.3절).

SSF 원 논문 근거: utils.py의 optimize_old_mask/optimize_new_mask가 고정
steps=100(kl_max_iter, CLI 노출 없음)으로 각 샘플에 소프트 마스크를 부여하고,
10-bin 히스토그램 분포 간 KL divergence를 최소화하도록 SGD로 최적화한 뒤
0.5 임계값으로 이진화해 대표 샘플을 선택한다(utils.py:109-194).

PRD 12.3절 select() 인터페이스는 모델/로짓을 넘기지 않으므로(new_data,
new_labels, label_budget, drift_score만 입력) — SSF 원문이 로짓/재구성-유사도
분포에 적용하던 히스토그램을, 이 테스트베드에서는 new_data의 제1주성분
투영값(스칼라) 분포에 적용한다. KL-마스크 최적화라는 핵심 메커니즘은 동일하게
유지하되, 이 인터페이스가 제공하는 입력(원본 feature)으로 대체한 것이다.

**drift_score 소비**: 이 테스트베드의 목적은 원본 코드 재현이 아니라 "어떤
호환 조합이 최적인가"를 비교하는 것이다(사용자 지시) — 그러려면
drift_detector 슬롯(none/ssf/cade)이 실제로 결과를 바꿔야 비교 자체가
성립한다. 처음에는 KL 목표 분포 자체를 drift_score로 바꿔봤지만, bin별
가중치가 "합(sum)"으로 계산되는 구조상 표본이 적은 극단 bin은 목표를 아무리
올려도 top-k 결과 자체가 거의 안 바뀌는 것을 실제로 확인했다(최적화 loss는
달라져도 순위가 안 바뀜 — 미묘한 SGD 수렴 특성에 의존하는 취약한 설계였음).

그래서 **직접적이고 검증 가능한 방식**으로 다시 짰다: (1) 평시 목표(균등
분포)로 계산한 대표성 점수 `mask_final`과 (2) 분포 중심에서 얼마나 먼지를
직접 재는 `extremity` 점수를 각각 [0,1]로 정규화한 뒤, `drift_weight`
비율로 선형 블렌딩해서 top-k를 뽑는다. drift_weight=0이면 순수 대표성 선택,
1에 가까울수록 순수 극단값(새 패턴/이상치) 선택으로 수렴한다 — 두 기준의
선형 결합이므로 blending 비율이 바뀌면 top-k 결과도 실제로 바뀐다는 것을
보장할 수 있다.

**중요한 수정(실제 데이터에서 발견된 붕괴 재현 후 수정)**: UNSW-NB15처럼
experience 간 분포가 극심하게 흔들리는 데이터에서는 drift_detected=True가
연속된 여러 experience에 걸쳐 계속 발동한다. `extremity`는 라벨과 무관하게
"분포 중심에서 먼 정도"만 재기 때문에, drift_weight가 계속 높게 유지되는
상태로 여러 라운드가 누적되면 선택된 학습 데이터가 점점 라벨 균형을 잃은
이상치 위주로 치우쳐, 분류기가 완전히 붕괴하는 현상을 실제로 재현했다
(`A_dd=ssf_ss=ssf_mm=ssf_af=lwf_ssf`와 `A_dd=cade_ss=ssf_mm=ssf_af=lwf_ssf`
조합, UNSW-NB15, 마지막 라운드에서 F1=0으로 완전 붕괴 — ablation으로
memory_manager가 아니라 이 extremity 블렌딩이 원인임을 확인). CADE의
drift_score는 상한이 없어(수백까지 커짐) tanh가 거의 즉시 1로 포화되므로,
상한값 0.3까지도 붕괴를 막기에 부족했다 — 0.2까지도 두 실패 사례 모두에서
붕괴를 재현했고, 0.1에서 두 사례 모두 안정화되는 것을 직접 확인한 뒤 이
값으로 낮췄다(0.05는 오히려 0.1보다 불안정 — 이 근방에서 비단조적인 민감한
동역학이라는 뜻이므로, 정밀 튜닝보다 실제 최악 사례 2건에서 안정성이 확인된
값을 보수적으로 채택했다). `max_drift_influence` 기본값 0.1 — drift_detector가
결과에 영향을 주긴 하되, 대표성 선택을 압도해 학습을 불안정하게 만들지는
않도록 하는 안전장치다.
"""

from typing import List

import torch
import torch.nn.functional as F

from testbed.base.sample_selector import BaseSampleSelector


class SSFSampleSelector(BaseSampleSelector):
    def __init__(self, kl_max_iter: int = 100, num_bins: int = 10,
                 max_drift_influence: float = 0.1):
        self.kl_max_iter = kl_max_iter
        self.num_bins = num_bins
        # drift_weight의 상한 — 위 docstring의 붕괴 재현 참고. 대표성 선택을
        # 완전히 압도하지 못하도록 제한한다.
        self.max_drift_influence = max_drift_influence

    def _scalar_projection(self, data: torch.Tensor) -> torch.Tensor:
        centered = data - data.mean(dim=0, keepdim=True)
        try:
            _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
            proj = centered @ Vh[0]
        except Exception:
            proj = centered.mean(dim=1)
        return proj

    def _histogram_bins(self, scores: torch.Tensor) -> torch.Tensor:
        lo, hi = scores.min(), scores.max()
        if (hi - lo).abs() < 1e-12:
            return torch.zeros(len(scores), dtype=torch.long, device=scores.device)
        edges = torch.linspace(lo.item(), hi.item(), self.num_bins + 1, device=scores.device)
        bins = torch.bucketize(scores, edges[1:-1])
        return bins

    def select(self, new_data: torch.Tensor, new_labels: torch.Tensor,
               label_budget: int, drift_score: float) -> List[int]:
        n = len(new_data)
        if n == 0:
            return []
        k = min(label_budget, n)
        if k == n:
            return list(range(n))

        scores = self._scalar_projection(new_data)
        bins = self._histogram_bins(scores)

        # 목표 분포: bin마다 균등 — SSF의 "대표성" 개념(특정 구간에 쏠리지 않고
        # 전체 분포를 고르게 대표하는 샘플을 선호)을 균등 목표 분포로 표현한다.
        # (drift_score와 무관하게 항상 균등 — 아래에서 별도로 블렌딩한다.)
        target_dist = torch.full((self.num_bins,), 1.0 / self.num_bins, device=new_data.device)

        mask_logit = torch.zeros(n, requires_grad=True, device=new_data.device)
        optimizer = torch.optim.SGD([mask_logit], lr=1.0)

        for _ in range(self.kl_max_iter):
            optimizer.zero_grad()
            mask = torch.sigmoid(mask_logit)
            bin_weights = torch.zeros(self.num_bins, device=new_data.device)
            for b in range(self.num_bins):
                sel = bins == b
                if sel.any():
                    bin_weights[b] = mask[sel].sum()
            bin_dist = bin_weights / (bin_weights.sum() + 1e-10)
            loss = F.kl_div((bin_dist + 1e-10).log(), target_dist, reduction="sum")
            loss.backward()
            optimizer.step()

        mask_final = torch.sigmoid(mask_logit).detach()

        def _min_max_norm(x: torch.Tensor) -> torch.Tensor:
            lo, hi = x.min(), x.max()
            if (hi - lo).abs() < 1e-12:
                return torch.zeros_like(x)
            return (x - lo) / (hi - lo)

        representativeness = _min_max_norm(mask_final)
        extremity = _min_max_norm((scores - scores.mean()).abs())

        # drift_score(0 이상, base/drift_detector.py 계약)는 0일 때 순수
        # 대표성 선택이어야 한다. sigmoid(0)=0.5라 부적절 — tanh(0)=0이고 큰
        # 값에서 1로 수렴하는 tanh를 쓴다. math.tanh(float64)로 바꾸면 torch의
        # float32 tanh와 최하위 비트가 달라져 top-k 선택이 실제로 바뀌는 것을
        # 확인해서(회귀 테스트로 적발) 원래의 torch.tanh 계산을 그대로 두고
        # device만 명시한다.
        drift_weight = self.max_drift_influence * torch.tanh(
            torch.tensor(float(drift_score), device=new_data.device))
        combined = (1 - drift_weight) * representativeness + drift_weight * extremity

        # PRD 15.1절의 label_budget 5% 이내 일치 요구를 정확히 지키기 위해
        # top-k로 선택한다(이진화만 쓰면 budget을 넘거나 못 채울 수 있음).
        topk = torch.topk(combined, k).indices.tolist()
        return topk
