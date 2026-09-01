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

**2026-08-26 발견·수정 — 균일-히스토그램 대체가 라운드마다 예측 불가능하게
클래스/family 비율을 왜곡하는 문제**: 4개 논문 컴포넌트 전수 재감사에서,
위 KL-마스크 최적화가 "제1주성분 투영값의 분포"만 보고 어떤 표본이 어떤
클래스인지는 전혀 모른다는 걸 재확인했다 — 그 결과 어느 클래스가 우연히
어느 bin에 몰려 있는지에 따라 선택 비율이 라운드마다 완전히 달라진다.
실측(NSL-KDD)으로 R2L 라운드(공격 995건)는 비례 기대치(99.5건) 대비 82%
적은 18건만 선택됐는데, U2R 라운드(52건)는 거의 정확히 비례대로 선택되는
등 들쭉날쭉했다. `select()`를 `_quota_select()`로 리팩터링해 label_budget을
먼저 그룹(이진 라벨) 구성비대로(최소 1개) 배정하고, 그 쿼터 안에서만 위
KL+extremity 메커니즘을 적용하도록 했다. 이진 쿼터만으로는 "공격" 예산
안에서 서로 다른 family까지는 공평해지지 않는다는 것도 확인해(U2R 라운드
자체 학습 성능이 여전히 낮게 나옴 — `ssf_memory_manager.py`의 같은 절
참고), `train_category`(다중클래스)로 쿼터를 나누는 `select_with_category()`
를 추가해봤다.

**한 번 기각했다가, 잘못된 비교였음이 밝혀져 재검증 후 채택함**: 처음엔
`select_with_category()`/`SSFMemoryManager.update_with_category()`(둘 다
같은 이유로 시도) 각각을 A/B 실측(NSL-KDD, `dd=ssf/ss=ssf/mm=ssf/
af=lwf_ssf/as=cade_mad`)해 이진 쿼터가 가장 낫다고 결론 내리고 둘 다
되돌렸다. 그런데 재감사(4개 병렬 에이전트) 과정에서, 이 A/B가 그 사이
바뀐 `data/dataset_loader.py`의 MinMaxScaler 시간 유출 수정 **이전**
숫자와 비교된, 이미 낡은 비교였다는 게 밝혀졌다(같은 "이진 쿼터만"
콤보를 현재 코드로 다시 재보면 f1=0.7291이 아니라 0.6565가 나온다 —
전처리가 바뀌면 4개 변형 전부의 절대 수치가 같이 움직이므로 순위까지
바뀔 수 있다는 걸 놓쳤었다). 현재 코드 기준으로 4개 변형을 전부 다시
측정한 결과:
  - 이진 쿼터만: f1=0.6565, roc_auc=0.7150, diag-F1=[0.851,0.554,0.257,
    0.009,0.0]
  - **선택기만 category 쿼터: f1=0.7040, roc_auc=0.6451, diag-F1=
    [0.846,0.556,0.254,0.067,0.0]** ← 전체 f1도, U2R 라운드 자체 성능도
    이진 쿼터보다 낫다.
  - 버퍼만 category 쿼터: f1=0.5107, roc_auc=0.5101(거의 무작위) — 여전히
    나쁨.
  - 둘 다 category 쿼터: f1=0.5879, roc_auc=0.5276 — 여전히 이진보다 나쁨.
결론이 뒤집혔다: **선택기의 category 쿼터는 채택**(전체 성능과 U2R 둘 다
개선), **버퍼의 category 쿼터는 여전히 기각**(단독으로도, 선택기와
합쳐도 더 나쁨 — `ssf_memory_manager.py`의 같은 절 참고). "실측 우선"
원칙을 지키려면 실측 자체가 최신 코드 기준이어야 한다는 교훈이 남는다 —
그리드 전체에 영향 주는 변경(전처리 등) 이후에는 이전 A/B 결론을 그대로
믿지 않고 재확인해야 한다.
"""

from typing import List

import numpy as np
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
        return self._quota_select(new_data, new_labels, label_budget, drift_score)

    def select_with_category(self, new_data: torch.Tensor, new_labels: torch.Tensor,
                              category, label_budget: int, drift_score: float) -> List[int]:
        """CLClient 전용 훅 — `train_category`(다중클래스)가 있으면 이진
        라벨 대신 그걸로 쿼터를 나눈다. 한 번 기각했다가 재감사에서 잘못된
        비교였음이 밝혀져 다시 채택했다 — 아래 "2026-08-26 재검증" 절 참고."""
        codes = np.unique(np.asarray(category), return_inverse=True)[1]
        group = torch.tensor(codes, dtype=torch.long, device=new_data.device)
        return self._quota_select(new_data, group, label_budget, drift_score)

    def _quota_select(self, new_data: torch.Tensor, group: torch.Tensor,
                       label_budget: int, drift_score: float) -> List[int]:
        n = len(new_data)
        if n == 0:
            return []
        k = min(label_budget, n)
        if k == n:
            return list(range(n))

        # 2026-08-26 발견·수정 — 균일-히스토그램 대체 방식이 라운드별로
        # 예측 불가능하게 그룹(클래스/category) 비율을 왜곡하는 문제(4개
        # 논문 컴포넌트 전수 재감사에서 발견). KL 최적화는 "제1주성분
        # 투영값의 분포"만 보고 그룹은 전혀 모르므로, 어느 그룹이 어느
        # bin에 몰려 있는지에 따라 선택 비율이 완전히 달라진다 — 실측
        # (NSL-KDD)으로 R2L 라운드(공격 995건)는 비례 기대치(99.5건) 대비
        # 82% 적은 18건만 선택됐는데, U2R 라운드(52건)는 거의 정확히
        # 비례대로 선택되는 등 라운드마다 들쭉날쭉했다. label_budget을
        # 그룹 구성비대로(최소 1개) 먼저 배정하고, 그 쿼터 안에서만 아래
        # KL-마스크 최적화+extremity 블렌딩을 적용한다 — "대표성 있는
        # 표본을 뽑는다"는 핵심 메커니즘은 그룹 내부에서 그대로 작동하되,
        # 그룹 간 비율 자체는 더 이상 히스토그램 우연에 좌우되지 않는다.
        groups = group.unique().tolist()
        if len(groups) > 1:
            idx_by_group = {g: (group == g).nonzero(as_tuple=True)[0] for g in groups}
            counts = {g: len(idx_by_group[g]) for g in groups}
            quotas = {g: min(max(1, round(k * counts[g] / n)), counts[g]) for g in groups}
            diff = k - sum(quotas.values())
            order = sorted(groups, key=lambda g: counts[g], reverse=True)
            i = 0
            while diff != 0 and i < 10000:
                g = order[i % len(order)]
                if diff > 0 and quotas[g] < counts[g]:
                    quotas[g] += 1
                    diff -= 1
                elif diff < 0 and quotas[g] > 0:
                    quotas[g] -= 1
                    diff += 1
                i += 1

            selected: List[int] = []
            for g in groups:
                if quotas[g] <= 0:
                    continue
                grp_idx = idx_by_group[g]
                local_topk = self._select_within_group(
                    new_data[grp_idx], quotas[g], drift_score)
                selected.extend(grp_idx[local_topk].tolist())
            return selected

        return self._select_within_group(new_data, k, drift_score).tolist()

    def _select_within_group(self, new_data: torch.Tensor, k: int,
                              drift_score: float) -> torch.Tensor:
        """단일 그룹(클래스) 안에서 KL 마스크 최적화 + extremity 블렌딩으로
        top-k 인덱스(그 그룹 내부 기준)를 뽑는다 — `select()`가 클래스별로
        호출한다(위 "2026-08-26" 절 참고)."""
        n = len(new_data)
        if k >= n:
            return torch.arange(n, device=new_data.device)

        scores = self._scalar_projection(new_data)
        bins = self._histogram_bins(scores)

        # 목표 분포: bin마다 균등.
        #
        # **2026-08-14 정정 — "SSF의 대표성 개념을 표현한 것"이 아니다**:
        # 구조 전수 감사에서 utils.py:109-190을 다시 정독한 결과, SSF의
        # 실제 목표 분포는 균일분포가 전혀 아님을 확인했다 —
        # `optimize_old_mask`/`optimize_new_mask`의 `bin_tgt_c`/`bin_tgt_t`
        # (utils.py:134,175)는 **treatment_res(현재 윈도우의 실제 관측
        # 분포)의 경험적 히스토그램**이다. 즉 SSF의 진짜 메커니즘은
        # "선택된 표본이 지금 실제로 관측되는(드리프트됐을 수 있는) 분포를
        # 따라가도록" 마스크를 최적화하는 drift-추종형 선택이지, "값 범위
        # 전체에 고르게 퍼뜨리는" 다양성 극대화가 아니다. 이 균일분포
        # 대체는 인터페이스 제약상 불가피했다 — `select()`는 old/control
        # 분포에 접근할 방법이 없고(버퍼나 모델을 안 받음), SSF 원문의
        # M_t 최적화 자체도 M_c(old mask)에 의존하는 구조라(utils.py:177)
        # SampleSelector 혼자서는 애초에 원문 메커니즘을 재현할 수 없다.
        # 즉 이건 "SSF 개념의 단순화"가 아니라 "SSF의 핵심 메커니즘을
        # 포기하고 완전히 다른 대체 휴리스틱(균일 커버리지)을 쓴 것"이다.
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
        return torch.topk(combined, k).indices
