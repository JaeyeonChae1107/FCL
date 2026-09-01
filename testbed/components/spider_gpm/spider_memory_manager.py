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

이 세 가지를 그대로 구현한다. 3번(pseudo-labeling)은 표준 BaseMemoryManager
계약(모델 접근 없음) 밖의 정보가 필요하므로, `NoAnomalyScorer.set_model()`/
`CNDIDSAntiForgetting.on_experience_start()`와 같은 패턴으로 선택적 훅
`set_snapshot_model()`을 둔다 — CLClient가 매 experience 종료 시(step 8,
anti_forgetting.on_task_end와 같은 시점) 그 라운드까지 학습된 모델의 스냅샷을
넘겨준다. 이 스냅샷이 바로 다음 라운드의 replay pseudo-labeling에 쓰인다.

Track B(CND-IDS)에서 쓰일 때는 `CNDIDSAntiForgetting.compute_loss()`가 애초에
replay_batch의 라벨을 쓰지 않으므로(라벨-프리 원칙) pseudo-label 값 자체는
소비되지 않는다 — "라벨 없는 무작위 교체 버퍼"라는 핵심 성질만 공유하면 되고,
Track A/B 양쪽 모두에서 이 메모리 매니저를 쓸 수 있다(사용자 지시).

**2026-08-26 발견 — 위 "Track A/B 양쪽 모두" 근거는 라벨-프리(CND-IDS)
경로만 검증한 것이었다**: 4개 논문 컴포넌트 전수 재감사에서, Track A의
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


class SPIDERMemoryManager(BaseMemoryManager):
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._buf_data: Optional[torch.Tensor] = None
        self._snapshot_model: Optional[BaseCLModel] = None

    def update(self, selected_data: torch.Tensor, selected_labels: torch.Tensor,
               drift_detected: bool = False) -> None:
        # "라벨 없는 샘플만 저장" — selected_labels는 의도적으로 버린다.
        # "단순 교체 정책(No MRP)" — 기존 버퍼를 누적하지 않고 통째로 교체한다.
        n = len(selected_data)
        k = min(self.max_size, n)
        idx = torch.randperm(n, device=selected_data.device)[:k]
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
        idx = torch.randperm(len(self._buf_data), device=self._buf_data.device)[:n]
        data = self._buf_data[idx]
        return data, self._pseudo_label(data)

    def size(self) -> int:
        return 0 if self._buf_data is None else len(self._buf_data)
