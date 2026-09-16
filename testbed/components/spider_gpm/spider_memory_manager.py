"""SPIDERMemoryManager — SPIDER 원 논문의 "유한 버퍼 메모리(M)" (PRD 4절/12.4절).

레지스트리 키는 "spider"(anti_forgetting의 "gpm"과는 별개 키 — 4.1절 명명 규칙과
같은 정신으로, SPIDER 논문의 서로 다른 두 메커니즘을 각각 별도 슬롯/키로 둔다).

**근거(사용자가 SPIDER 원 논문을 직접 확인해 제공)**:
SPIDER는 GPM(anti_forgetting) 외에 별도의 유한 버퍼 메모리 M을 두는데, 기존
리플레이 방식과 두 가지가 다르다:
  1. **라벨 없는 샘플만 저장(privacy-preserving)** — 버퍼에는 라벨이 없는
     샘플만 저장하고, 이전 태스크(t-1)에서 무작위로 선택된 것이다.
  2. **단순 교체 정책(No MRP)** — 복잡한 memory reconstruction policy 없이,
     현재 태스크의 학습이 끝나면 버퍼 전체를 현재 태스크의 라벨 없는 무작위
     샘플들로 완전히 교체한다(누적/FIFO 방식이 아니다).
  3. **Pseudo-labeling** — 버퍼 데이터는 라벨이 없으므로, replay에 쓸 때
     바로 직전 태스크까지 학습된 모델(f_θ^(t-1))로 실시간 pseudo-label을
     생성해 학습에 사용한다.

**Item 8a — 버퍼가 라벨 예산 서브셋에서 채워지던 결함**: "1번(무작위 샘플)"이
실제로는 `update(selected_data, ...)`로 라벨 예산(Track A 기준 10%)만큼만
골라진 서브셋에서 뽑혀, 원문의 "이전 태스크 전체에서 무작위"보다 표본
다양성이 훨씬 좁았다. `consumes_full_round_data=True` 클래스 속성을 두고
`cl_client.py` Step 5가 이걸 보고 `selected_data` 대신 `new_data`(이번
라운드 전체)를 넘기도록 수정 — 이미 라벨을 버리는 컴포넌트라 본문 수정은
불필요했다. A/B(NSL-KDD): `mm=spider/af=none/as=cade_mad` f1 0.705→0.800,
bwt -0.060→+0.010; `mm=spider/af=gpm/as=cade_mad` f1 0.806→0.863(bwt
+0.007→-0.007, 거의 0 유지) — 둘 다 개선.

이 세 가지를 그대로 구현한다. 3번(pseudo-labeling)은 표준 BaseMemoryManager
계약(모델 접근 없음) 밖의 정보가 필요하므로, `NoAnomalyScorer.set_model()`/
`CNDIDSAntiForgetting.on_experience_start()`와 같은 패턴으로 선택적 훅
`set_snapshot_model()`을 둔다 — CLClient가 매 experience 종료 시(step 8,
anti_forgetting.on_task_end와 같은 시점) 그 라운드까지 학습된 모델의 스냅샷을
넘겨준다. 이 스냅샷이 바로 다음 라운드의 replay pseudo-labeling에 쓰인다.

Track B(CND-IDS)에서 쓰일 때는 `CNDIDSAntiForgetting.compute_loss()`가 애초에
replay_batch의 라벨을 쓰지 않으므로(라벨-프리 원칙) pseudo-label 값 자체는
소비되지 않는다 — "라벨 없는 무작위 교체 버퍼"라는 핵심 성질만 공유하면 되고,
Track A/B 양쪽 모두에서 이 메모리 매니저를 쓸 수 있다.

**위 "Track A/B 양쪽 모두" 근거는 라벨-프리(CND-IDS) 경로만 검증한
것이었다**: Track A의
`af=lwf_ssf`(`SSFAntiForgetting.compute_loss`)와 `af=none`
(`NoAntiForgetting.compute_loss`)는 CND-IDS와 달리 `replay_batch`의
라벨을 **실제로** BCE 손실에 쓴다는 걸 재확인했다. 그런데 `mm=spider`와
결합되면 그 "라벨"은 실측 정답이 아니라 `_pseudo_label()`이 스냅샷
모델의 예측으로 만든 것이다 — 즉 `mm=spider`+`af=lwf_ssf`(또는
`af=none`) 조합은 모델이 최근에 낸 예측을 스스로 정답처럼 다시 학습하는
자기학습(self-training) 피드백 루프가 된다. 이 조합은 크래시하거나
눈에 띄게 붕괴하지는 않았다(실행 확인 완료 — `mm=spider`+`af=gpm`이
발견 3에서 건강하게 확인된 것과 별개로 `af=none`/`af=lwf_ssf` 조합도
정상 실행됨). 다만 이 자기학습 루프 자체의 장단점(예: 모델이 자신 있게
틀린 예측을 계속 강화할 위험)은 아직 별도로 분석된 적이 없다 — "Track
A/B 양쪽 모두 쓸 수 있다"는 근거가 실제로는 라벨-프리 경로의 안전성만
증명했을 뿐, 이 self-training 경로는 다른 성격의 위험이라는 점을
정직하게 남겨둔다.
"""

import copy
from typing import Optional, Tuple

import torch

from testbed.base.memory_manager import BaseMemoryManager
from testbed.base.models import BaseCLModel
from testbed.common.rng_utils import derived_seed


class SPIDERMemoryManager(BaseMemoryManager):
    """**전역 RNG 오염(2026-09-14 재검토로 발견, 수정)**: `update()`/
    `get_replay_batch()`의 `torch.randperm`이 전역 RNG를 소비한다 —
    `mm=spider`일 때만 호출되므로, 이 슬롯 값 하나로 그 뒤 메인 모델
    학습이 보는 난수 시퀀스 전체가 갈라진다(`gpm_anti_forgetting.py`
    모듈 docstring "전역 RNG 오염" 참고, 같은 문제 패턴). `torch.random.
    fork_rng()`로 격리한다."""

    # Item 8a — SPIDER 원문의 버퍼는 "이전 태스크의 무작위 샘플"이지 라벨
    # 예산으로 골라진 서브셋이 아니다(이미 라벨을 버리는 컴포넌트이므로
    # 이번 라운드 전체 데이터를 받아도 라벨-프리 성질은 그대로 유지된다).
    # `cl_client.py` Step 5가 이 속성을 보고 `selected_data` 대신
    # `new_data`(라운드 전체)를 넘긴다.
    consumes_full_round_data = True

    def __init__(self, max_size: int = 1000, seed: int = 42):
        self.max_size = max_size
        self._buf_data: Optional[torch.Tensor] = None
        self._snapshot_model: Optional[BaseCLModel] = None
        self._seed = seed
        self._update_round = 0
        self._replay_call_count = 0

    def update(self, selected_data: torch.Tensor, selected_labels: torch.Tensor,
               drift_detected: bool = False) -> None:
        # "라벨 없는 샘플만 저장" — selected_labels는 의도적으로 버린다.
        # "단순 교체 정책(No MRP)" — 기존 버퍼를 누적하지 않고 통째로 교체한다.
        n = len(selected_data)
        k = min(self.max_size, n)
        with torch.random.fork_rng():
            torch.manual_seed(derived_seed(self._seed, "spider_update", self._update_round))
            idx = torch.randperm(n, device=selected_data.device)[:k]
        self._update_round += 1
        self._buf_data = selected_data[idx].clone()

    def set_snapshot_model(self, model: BaseCLModel) -> None:
        """CLClient가 매 experience 종료 시(step 8) 호출 — 그 시점의 모델을
        복제해 스냅샷으로 보관한다. 다음 라운드의 replay pseudo-labeling에
        "바로 직전 태스크까지 학습된 모델"로 쓰인다."""
        self._snapshot_model = copy.deepcopy(model)
        self._snapshot_model.eval()
        for p in self._snapshot_model.parameters():
            p.requires_grad_(False)

    def _pseudo_label(self, data: torch.Tensor) -> torch.Tensor:
        if self._snapshot_model is None:
            # 아직 스냅샷이 없는 첫 라운드 — 안전한 폴백으로 전부 정상(0) 취급.
            return torch.zeros(len(data), dtype=torch.long, device=data.device)
        with torch.no_grad():
            _, _, logit = self._snapshot_model(data)
        return (torch.sigmoid(logit).squeeze(-1) > 0.5).long()

    def get_buffer(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # drift_detector(step 2)는 buf_ref로 데이터만 쓰고 라벨은 버리므로
        # (CLClient: `buf_data, _ = self.memory_manager.get_buffer()`),
        # 여기서는 불필요한 pseudo-label 계산을 하지 않는다.
        return self._buf_data, None

    def get_replay_batch(self, batch_size: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self._buf_data is None:
            return None, None
        n = min(batch_size, len(self._buf_data))
        with torch.random.fork_rng():
            torch.manual_seed(derived_seed(self._seed, "spider_replay", self._replay_call_count))
            idx = torch.randperm(len(self._buf_data), device=self._buf_data.device)[:n]
        self._replay_call_count += 1
        data = self._buf_data[idx]
        return data, self._pseudo_label(data)

    def size(self) -> int:
        return 0 if self._buf_data is None else len(self._buf_data)
