"""SSF MemoryManager — Strategic Forgetting, 원문 그대로 이식 (PRD 4절/12.4절,
Item 1+2).

SSF 원 논문 근거(`utils.py:192-388`, `select_and_update_representative_
samples[_when_drift]()`)가 실제로 하는 일 — 같은 라운드에 `CLClient`가
계산한 공유 마스크(M_c/M_t, `ssf_masks.py`)를 `SSFSampleSelector`와
공유해서 쓴다:

**드리프트 없음**: 버퍼(old)에서 비대표(`M_c_bin==0`) 표본을 이번 라운드
새로 선택된 표본 수(`k`)만큼 제거한다(비대표가 모자라면 대표 중 `M_c`
최하위부터 추가 제거) — 그 자리에 이번 라운드 `selected_data`(원문의
`representative_new`, `SSFSampleSelector`가 이미 같은 `M_t` 마스크로
뽑아둔 것)를 무조건 추가한다.

**드리프트**: old의 비대표 표본을 **전부** 제거(제거 개수 상한 없음) —
그 결과 버퍼가 목표 크기(`max_size`) 아래로 줄면, (1) `M_t_bin==1`이지만
이번 라운드 라벨 예산으로 선택되지 않은 잔여 표본(`M_t` 점수 상위부터) →
(2) 그것도 부족하면 나머지 유입 데이터에서 무작위 표본을, **현재
실시간 학습 중인 모델**(스냅샷 아님)로 pseudo-label해서 순서로 채운다.

`M_c`/`M_t`가 없는 첫 라운드(버퍼가 비어있음)는 원문에 없는 경계
케이스로, `selected_data`를 그대로 초기 버퍼로 쓴다. `CLClient`가
공유 마스크를 계산하지 않는 상황(sample_selector/memory_manager 둘 다
`set_ssf_masks`가 없을 때, 즉 이 클래스가 안 쓰일 때)은 발생하지 않는다.

**테스트베드 고유의 안전장치**: label_budget이 SSF 원문의 고정
`num_labeled_sample=200`과 달리 라운드 크기에 비례해 커질 수 있어,
위 로직을 그대로 적용해도 combined 크기가 `max_size`를
넘는 극단적 경우가 이론상 있을 수 있다 — 원문에는 없는 최종 안전 캡
(무작위 서브샘플)을 추가했다(정상 경로에서는 발동하지 않음).

**A/B 실측(NSL-KDD) — 콤보에 따라 방향이 갈림, 유지**: `dd=none/ss=ssf/
mm=none/af=lwf_ssf/as=cade_mad`(마스크가 선택기만 소비): f1 0.654→0.779,
roc_auc 0.679→0.815(개선). `dd=ssf/ss=ssf/mm=ssf/af=lwf_ssf/as=cade_mad`
(선택기+버퍼 둘 다 소비, 원문 구조에 가장 가까운 콤보): f1 0.625→0.749,
bwt -0.0094→-0.0040, roc_auc 0.687→0.824(개선). `dd=none/ss=random/
mm=ssf/af=none/as=none`(마스크가 버퍼만 소비, 망각방지 자체가 없는
콤보): f1 0.621→0.366, bwt -0.215→-0.339(뚜렷한 악화). 이 콤보는
`dd=none`이라 `drift_detected`가 항상 False라 드리프트 분기(위 pseudo-
labeling 포함)는 전혀 발동하지 않는다 — 순수하게 평시(no-drift) 분기의
"이번 라운드 선택 개수만큼 M_c 최하위부터 제거" 방식이, 이전의 "클래스별
쿼터 안에서 centroid-거리 top-k 유지" 방식보다 `af=none`(망각방지
메커니즘이 아예 없어 버퍼 리플레이 품질에 결과가 더 민감한 상태)에서
약한 것으로 보인다. Item 3/4와 달리 전 콤보에서 일관되게 나빠지는 패턴이
아니라(오히려 원문 구조에 가장 가까운 두 콤보는 뚜렷이 개선) 유지한다 —
다만 "마스크가 버퍼만 소비하는" 이 특정 조합 유형은 후속 Track A 전체
검증에서 주의 깊게 볼 필요가 있다.

**드리프트 backfill의 무작위 풀이 원문과 다름(2026-09-14 재검토로 발견,
의도적으로 유지)**: 원문(`utils.py:348,367`)의 무작위 backfill은
`x_test_this_epoch` 전체(이미 이번 라운드에 라벨링돼 학습셋에 들어간
표본 포함)에서 `torch.randperm`으로 뽑아, 이미 선택된 표본이 pseudo-label
버전으로 버퍼에 중복 삽입될 수 있다(원문이 이를 배제하는 마스킹을 전혀
하지 않음을 재확인). 이 테스트베드는 `already_selected`(sel_idx +
leftover_repr_idx)를 명시적으로 제외한 풀에서만 뽑아 이 중복을 원천
차단한다. 원문 스케일(`sample_interval=20000`)에서는 충돌 확률이 낮아
실무적 영향이 작았겠지만, 이 테스트베드의 라운드 크기는 그보다 훨씬
작을 수 있어 원문 그대로 두면 충돌 확률이 원문보다 커질 수 있다 —
"이미 실측 라벨이 있는 표본을 모델 자신의 예측(pseudo-label)으로 덮어
버퍼에 중복 삽입"하는 것은 원문의 의도된 설계라기보다 원문 자신의
스케일에서만 무해했던 부수효과로 판단해, 중복을 배제하는 현재 방식을
유지한다(A/B 미실시 — 원문이 의도한 "무작위성" 자체는 보존되고 풀
구성원 자격만 더 엄격해진 것이라 원문의 핵심 메커니즘을 훼손하지
않는다고 판단).

**`ss=ssf`+`mm=ssf`+`af=lwf_ssf` 자기학습 피드백 루프(2026-09-14, Round-4
교차-슬롯 감사로 발견)**: 위 드리프트 backfill이 `_pseudo_label()`(현재
실시간 학습 중인 모델)로 채운 표본은 `self._buf_labels`에 실측 라벨과
구분 없이 저장된다. `get_replay_batch()`가 돌려주는 이 버퍼는
`cl_client.py` Step 4에서 `SSFAntiForgetting.compute_loss()`로 그대로
전달되는데, `ssf_anti_forgetting.py`의 `_task_loss()`가 이 replay_batch를
BCE 손실뿐 아니라 InfoNCE 대조 손실의 `labels==0` anchor 마스크
(`ssf_infonce.py`)에도 실측 라벨과 동일하게 사용한다 — 즉 모델이 방금 낸
예측이 그대로 "정상 anchor"인지 아닌지를 정하는 기준이 되어 자기 자신을
강화하는 구조다. `spider_memory_manager.py`가 `mm=spider`+`af=lwf_ssf`/
`af=none`에서 이미 정직하게 기록해둔 것과 같은 종류의 자기학습 피드백
루프이지만, 그 컴포넌트와는 완전히 독립된 별도 발생원(SPIDER의 버퍼
교체 pseudo-labeling이 아니라 SSF 자신의 드리프트 backfill)이다 — 지금까지
그 SPIDER 쪽 사례만 문서화돼 있었고 이 `mm=ssf` 쪽은 빠져 있었다.
크래시·눈에 띄는 붕괴는 관찰되지 않았고(위 A/B 실측 결과들이 이미 이
경로를 거쳐 나온 정상 수치다), 이 피드백 루프 자체의 장단점은 SPIDER
쪽과 마찬가지로 별도 분석된 적이 없다는 점을 정직하게 남겨둔다 —
`design_decisions.md` 7절 참고.
"""

from typing import Optional, Tuple

import torch

from testbed.base.memory_manager import BaseMemoryManager
from testbed.base.models import BaseCLModel
from testbed.common.rng_utils import derived_seed
from testbed.components.ssf.ssf_masks import SSFMaskContext


class SSFMemoryManager(BaseMemoryManager):
    """**전역 RNG 오염(2026-09-14 재검토로 발견, 수정)**: `update()`/
    `get_replay_batch()`의 `torch.randperm`이 전역 RNG를 소비한다 —
    `mm=ssf`일 때만 호출되므로, 이 슬롯 값 하나로 그 뒤 메인 모델
    학습이 보는 난수 시퀀스가 갈라진다(`spider_memory_manager.py`/
    `gpm_anti_forgetting.py` 모듈 docstring "전역 RNG 오염" 참고, 같은
    문제 패턴 — CADE/SSF마스크/GPM/SPIDER만 고치고 이 클래스를 그대로
    두면 "일부 슬롯 비교만 깨끗함"이라는 앞뒤 안 맞는 상태가 되므로
    함께 고친다). `torch.random.fork_rng()`로 격리한다."""

    def __init__(self, max_size: int = 1000, seed: int = 42):
        self.max_size = max_size
        self._buf_data: Optional[torch.Tensor] = None
        self._buf_labels: Optional[torch.Tensor] = None
        self._ctx: Optional[SSFMaskContext] = None
        self._model: Optional[BaseCLModel] = None
        self._seed = seed
        self._update_round = 0
        self._replay_call_count = 0

    def set_model(self, model: BaseCLModel) -> None:
        """CLClient가 매 experience 종료 시(step 8과 무관, __init__ 시점에
        한 번 연결 — 같은 model 객체가 매 라운드 그 자리에서 갱신되므로
        한 번만 받아두면 된다) 호출 — 드리프트 시 부족분 backfill의
        pseudo-labeling에 쓴다(NoAnomalyScorer.set_model()과 동일 패턴)."""
        self._model = model

    def set_ssf_masks(self, ctx: SSFMaskContext) -> None:
        """CLClient가 Step 3 진입 시 라운드당 1회 호출(Item 1+2) —
        SSFSampleSelector와 같은 마스크를 공유한다."""
        self._ctx = ctx

    def _pseudo_label(self, data: torch.Tensor) -> torch.Tensor:
        was_training = self._model.training
        self._model.eval()
        try:
            with torch.no_grad():
                _, _, logit = self._model(data)
        finally:
            if was_training:
                self._model.train()
        return (torch.sigmoid(logit).squeeze(-1) > 0.5).long()

    def update(self, selected_data: torch.Tensor, selected_labels: torch.Tensor,
               drift_detected: bool = False) -> None:
        if self._buf_data is None or len(self._buf_data) == 0:
            # 원문에 없는 경계 케이스(버퍼가 아직 없는 첫 라운드) — 그대로
            # 초기 버퍼로 쓴다.
            self._buf_data = selected_data.clone()
            self._buf_labels = selected_labels.clone()
            return

        # 전역 RNG 오염 격리(클래스 docstring 참고) — 이 메서드 안의 3곳
        # randperm(비대표 무작위 제거, 무작위 backfill, 안전 캡)을 전부
        # 라운드당 하나의 파생 시드로 감싼다.
        with torch.random.fork_rng():
            torch.manual_seed(derived_seed(self._seed, "ssf_memory_update", self._update_round))
            ctx = self._ctx
            old_data, old_labels = self._buf_data, self._buf_labels
            M_c_bin = ctx.M_c_bin.bool()
            M_c_cont = ctx.M_c_cont
            k = len(selected_data)

            old_indices = torch.arange(len(old_data), device=old_data.device)
            representative_old_indices = old_indices[M_c_bin]
            non_representative_old_indices = old_indices[~M_c_bin]

            if drift_detected:
                # 드리프트: 비대표 전부 제거(상한 없음).
                remove_indices = non_representative_old_indices
            else:
                # 평시: 이번 라운드 선택 개수(k)만큼만 제거(비대표 우선).
                if len(non_representative_old_indices) <= k:
                    remove_indices = non_representative_old_indices
                else:
                    perm = torch.randperm(len(non_representative_old_indices), device=old_data.device)
                    remove_indices = non_representative_old_indices[perm[:k]]

            if len(remove_indices) < k:
                # 비대표만으로 부족 — 대표 중 M_c 최하위부터 추가 제거.
                additional_needed = k - len(remove_indices)
                rep_scores = M_c_cont[representative_old_indices]
                lowest = torch.argsort(rep_scores)[:additional_needed]
                remove_indices = torch.cat([remove_indices, representative_old_indices[lowest]])

            keep_mask = torch.ones(len(old_data), dtype=torch.bool, device=old_data.device)
            keep_mask[remove_indices] = False
            old_data, old_labels = old_data[keep_mask], old_labels[keep_mask]

            combined_data = torch.cat([old_data, selected_data], dim=0)
            combined_labels = torch.cat([old_labels, selected_labels], dim=0)

            if drift_detected and len(combined_data) < self.max_size:
                shortfall = self.max_size - len(combined_data)
                new_data, M_t_bin = ctx.new_data, ctx.M_t_bin.bool()
                already_selected = torch.zeros(len(new_data), dtype=torch.bool, device=new_data.device)
                if ctx.sel_idx is not None:
                    already_selected[ctx.sel_idx] = True

                leftover_repr_idx = (M_t_bin & ~already_selected).nonzero(as_tuple=True)[0]
                if len(leftover_repr_idx) > shortfall:
                    order = torch.argsort(ctx.M_t_cont[leftover_repr_idx], descending=True)
                    leftover_repr_idx = leftover_repr_idx[order[:shortfall]]
                backfill_data = new_data[leftover_repr_idx]
                backfill_labels = self._pseudo_label(backfill_data) if len(backfill_data) > 0 else \
                    backfill_data.new_zeros(0, dtype=torch.long)

                remaining_shortfall = shortfall - len(leftover_repr_idx)
                if remaining_shortfall > 0:
                    already_selected[leftover_repr_idx] = True
                    remaining_pool_idx = (~already_selected).nonzero(as_tuple=True)[0]
                    n_take = min(remaining_shortfall, len(remaining_pool_idx))
                    if n_take > 0:
                        perm = torch.randperm(len(remaining_pool_idx), device=new_data.device)[:n_take]
                        random_idx = remaining_pool_idx[perm]
                        random_data = new_data[random_idx]
                        random_labels = self._pseudo_label(random_data)
                        backfill_data = torch.cat([backfill_data, random_data], dim=0)
                        backfill_labels = torch.cat([backfill_labels, random_labels], dim=0)

                combined_data = torch.cat([combined_data, backfill_data], dim=0)
                combined_labels = torch.cat([combined_labels, backfill_labels], dim=0)

            if len(combined_data) > self.max_size:
                # 원문에 없는 테스트베드 고유 안전 캡 — 모듈 docstring 참고.
                perm = torch.randperm(len(combined_data), device=combined_data.device)[:self.max_size]
                combined_data, combined_labels = combined_data[perm], combined_labels[perm]

        self._update_round += 1
        self._buf_data, self._buf_labels = combined_data, combined_labels

    def get_replay_batch(self, batch_size: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self._buf_data is None:
            return None, None
        n = min(batch_size, len(self._buf_data))
        with torch.random.fork_rng():
            torch.manual_seed(derived_seed(self._seed, "ssf_memory_replay", self._replay_call_count))
            idx = torch.randperm(len(self._buf_data), device=self._buf_data.device)[:n]
        self._replay_call_count += 1
        return self._buf_data[idx], self._buf_labels[idx]

    def get_buffer(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._buf_data, self._buf_labels

    def size(self) -> int:
        return 0 if self._buf_data is None else len(self._buf_data)
