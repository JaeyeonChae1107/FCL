"""SSF SampleSelector — 공유 마스크(M_t) 기반 대표 표본 선택 (PRD 4절/12.3절,
Item 1+2).

SSF 원 논문 근거: `select_and_update_representative_samples[_when_drift]()`
(`utils.py:192-388`)가 `representative_new = x_test_this_epoch[M_t_bin.bool()]`
로 대표 표본을 고르고, 부족하면 `M_t` 점수 상위부터, 그것도 부족하면
무작위로 채운다. `M_t`(연속 마스크)는 `CLClient._compute_ssf_masks()`가
라운드당 1회 계산해(`ssf_masks.py`) `set_ssf_masks()`로 넘겨준다 —
`SSFMemoryManager`와 같은 마스크를 공유한다.

**이전 버전과의 차이(재검증 후 재작성)**: 이전에는 이 클래스가 `select()`
인터페이스(모델/로짓을 받지 않음) 제약 때문에 마스크를 자체적으로(제1주성분
투영값 분포로) 근사 계산했고, drift_score를 반영하기 위해 "대표성 점수"와
"분포 중심에서 먼 정도(extremity)"를 선형 블렌딩하는 자체 장치를 뒀다.
원문을 재대조한 결과 `optimize_old_mask`/`optimize_new_mask`(마스크 계산
자체)는 drift 여부와 완전히 무관하게 계산되고, drift 감지 결과는
memory_manager의 갱신 분기 선택에만 영향을 준다는 게 확인돼 — "선택
단계에서 drift로 점수를 블렌딩한다"는 개념 자체가 원문에 없었다. 이제
`CLClient`가 공유 마스크를 계산해 넘겨주므로 이 클래스는 자체 마스크
계산도, extremity/drift_weight 블렌딩도 하지 않고 `M_t_cont`(연속값)를
그대로 랭킹에 쓴다.

**category 쿼터는 유지**: 균일-히스토그램 대체 시절부터 있던 문제(선택
비율이 라운드마다 클래스/family 구성에 따라 왜곡되는 것)는 마스크 계산
방식과 무관하게 여전히 유효하다 — label_budget을 그룹(이진 라벨 또는
`train_category`)별로 먼저 배정하고, 그 쿼터 안에서 대표 표본을 뽑는다
(`_quota_select`). 이 쿼터 로직은 이전 버전과 동일하게 유지한다(A/B로
이미 검증된 테스트베드 확장 — 선택기의 category 쿼터가 이진보다 낫다는
결론, `docs/metric_justification.md` 참고).

**2026-09-14 정정 — "부족/과다" 분기 누락 발견 및 수정**: 원문
(`utils.py:192-257`)을 다시 정독한 결과, 지난 버전이 "언제나
`torch.topk(M_t_cont, k)`"로 단순화했던 게 원문의 실제 두 분기와 다름을
발견했다. 원문은:
  1. 먼저 이진마스크 통과분(`representative_new = x_test_this_epoch[
     M_t_bin.bool()]`)을 구한다.
  2. 통과분이 쿼터보다 **많으면**(`utils.py:247-251`) 그 통과분 **안에서만**
     `M_t`(연속) 점수로 상위 k개를 추린다.
  3. 통과분이 쿼터보다 **적으면**(`utils.py:235-246`) 통과분을 전부 쓰고,
     모자란 만큼을 마스크 미달(비통과) 표본 중에서 **무작위**로 채운다
     (연속 점수로 이어서 채우지 않는다).

쿼터(k) ≤ 이진마스크 통과 수일 때는 두 방식이 수학적으로 동일하다(이진
통과분이 전부 연속점수 0.5 이상이라, 전체에서 top-k를 뽑아도 결과가
같음) — 그래서 이전 A/B(재검증 절 위 문단)가 이 차이를 못 잡아냈을
가능성이 높다. 갈리는 건 쿼터가 통과 수보다 **클 때만**인데, 그 경우
지난 버전은 마스크 미달 후보 중 "점수가 높은 순"으로 결정론적으로
채웠지만 원문은 "무작위"로 채운다 — 마스크가 걸러낸 비대표 표본
중에서까지 점수로 편향된 선택을 하면, 원문이 의도한 "부족분은 편향 없이
채운다"는 취지와 어긋난다. `_quota_select()`가 이 두 분기를 그룹별로 정확히 재현하도록 수정했다
(`_select_within_group()` 참고).

**A/B 실측(NSL-KDD smoke, 2026-09-14)**: 결과가 갈렸다 —
`dd=ssf/ss=ssf/mm=ssf/af=lwf_ssf/as=cade_mad`(SSF 풀 조합)는 f1
0.6915→0.7749, bwt +0.0312→+0.0074, pr_auc 0.8499→0.8799로 **개선**.
반면 `dd=none/ss=ssf/mm=none/af=none/as=cade_mad`(선택기 단독)는 f1
0.7479→0.7210, bwt -0.0405→-0.0850으로 **악화**. 이번 수정은 "더 나은
방식을 새로 시도"가 아니라 "원문과 다르게 동작하던 실제 버그를 원문대로
고친 것"이라 결과가 갈려도 유지한다(사용자 지시) — 원인 분석은 추후
필요하면(예: 무작위 보충이 SSF 마스크의 의도된 표본 다양성을 방해하는
방향으로 상호작용하는지) 별도로.
"""

from typing import List

import numpy as np
import torch

from testbed.base.sample_selector import BaseSampleSelector
from testbed.common.rng_utils import derived_seed
from testbed.components.ssf.ssf_masks import SSFMaskContext


class SSFSampleSelector(BaseSampleSelector):
    """**전역 RNG 오염(2026-09-14 재검토로 발견, 수정)**: `_quota_select()`
    안의 `torch.rand`(폴백)/`_select_within_group()`의 `torch.randperm`
    (쿼터가 마스크 통과 수보다 클 때 무작위 보충)이 전역 RNG를 소비한다 —
    `ss=ssf`일 때만 발동하므로, 이 슬롯 값 하나로 그 뒤 메인 모델 학습이
    보는 난수 시퀀스가 갈라진다(`common_baselines.py`의 `RandomSelector`
    모듈 docstring 참고, 같은 문제 패턴). `torch.random.fork_rng()`로
    라운드당 한 번 격리한다."""

    def __init__(self, seed: int = 42):
        self._ctx: SSFMaskContext = None
        self._seed = seed
        self._round = 0

    def set_ssf_masks(self, ctx: SSFMaskContext) -> None:
        """CLClient가 Step 3 진입 시 라운드당 1회 호출(Item 1+2)."""
        self._ctx = ctx

    def select(self, new_data: torch.Tensor, new_labels: torch.Tensor,
               label_budget: int, drift_score: float) -> List[int]:
        return self._quota_select(new_data, new_labels, label_budget)

    def select_with_category(self, new_data: torch.Tensor, new_labels: torch.Tensor,
                              category, label_budget: int, drift_score: float) -> List[int]:
        """CLClient 전용 훅 — `train_category`(다중클래스)가 있으면 이진
        라벨 대신 그걸로 쿼터를 나눈다(모듈 docstring 참고)."""
        codes = np.unique(np.asarray(category), return_inverse=True)[1]
        group = torch.tensor(codes, dtype=torch.long, device=new_data.device)
        return self._quota_select(new_data, group, label_budget)

    def _quota_select(self, new_data: torch.Tensor, group: torch.Tensor,
                       label_budget: int) -> List[int]:
        n = len(new_data)
        if n == 0:
            return []
        k = min(label_budget, n)
        if k == n:
            return list(range(n))

        # 전역 RNG 오염 격리(클래스 docstring 참고) — 아래 torch.rand 폴백과
        # _select_within_group()의 randperm을 라운드당 하나의 파생 시드로
        # 감싼다.
        with torch.random.fork_rng():
            torch.manual_seed(derived_seed(self._seed, "ssf_quota_select", self._round))

            if self._ctx is None or self._ctx.new_data.shape[0] != n:
                # set_ssf_masks()가 호출되지 않았거나(단위 테스트 등 독립 호출)
                # new_data가 컨텍스트와 어긋나는 극단 상황 — 크래시 대신 무작위
                # 선택으로 폴백한다(원문에 없는 경계 케이스, common_baselines.py
                # RandomSelector와 동일한 성격). 이진마스크도 같은 방식(0.5 컷)
                # 으로 일관되게 파생시킨다.
                m_t_cont = torch.rand(n, device=new_data.device)
                m_t_bin = (m_t_cont >= 0.5)
            else:
                m_t_cont = self._ctx.M_t_cont
                m_t_bin = self._ctx.M_t_bin.bool()

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
                    chosen_local = self._select_within_group(
                        m_t_cont[grp_idx], m_t_bin[grp_idx], quotas[g])
                    selected.extend(grp_idx[chosen_local].tolist())
                result = selected
            else:
                result = self._select_within_group(m_t_cont, m_t_bin, k).tolist()

        self._round += 1
        return result

    @staticmethod
    def _select_within_group(cont: torch.Tensor, bin_mask: torch.Tensor, quota: int
                              ) -> torch.Tensor:
        """SSF 원문(`utils.py:192-257`)의 두 분기를 그대로 재현한다 —
        모듈 docstring "2026-09-14 정정" 절 참고. `cont`/`bin_mask`는 이미
        하나의 그룹(카테고리 쿼터 적용 시) 또는 전체(단일 그룹)로 슬라이싱된
        상태로 들어온다 — 반환값은 그 슬라이스 기준 **로컬** 인덱스다.
        """
        rep_local = bin_mask.nonzero(as_tuple=True)[0]
        n_rep = len(rep_local)
        if n_rep >= quota:
            # 원문 utils.py:247-251 — 통과분이 쿼터보다 많으면 그 안에서만
            # 연속점수(M_t) 상위 quota개.
            rep_scores = cont[rep_local]
            top_of_rep = torch.topk(rep_scores, quota).indices
            return rep_local[top_of_rep]
        # 원문 utils.py:235-246 — 통과분이 쿼터보다 적으면 전부 쓰고,
        # 모자란 만큼을 마스크 미달(비통과) 표본 중 무작위로 채운다
        # (연속점수로 이어서 채우지 않는다 — 여기가 이전 버전의 실제 버그).
        non_rep_local = (~bin_mask).nonzero(as_tuple=True)[0]
        need = min(quota - n_rep, len(non_rep_local))
        if need > 0:
            perm = torch.randperm(len(non_rep_local), device=cont.device)[:need]
            fallback_local = non_rep_local[perm]
            return torch.cat([rep_local, fallback_local])
        return rep_local
