"""공통(파이프라인 인프라) 폴백 컴포넌트 — PRD 4절 compatibility table의 "공통"
칸(drift_detector='none', sample_selector='random', memory_manager='none',
anti_forgetting='none'). 특정 논문의 알고리즘이 아니라 "메커니즘 없음"/"무작위
선택" 같은 null-baseline이며, PRD 0절이 말하는 "논문 근거가 필요 없는 일반적인
소프트웨어 엔지니어링 관행"에 해당한다.

(이전에는 memory_manager='fifo'도 여기 있었으나, SPIDER 원 논문의 실제 유한
버퍼 메커니즘(components/spider_gpm/spider_memory_manager.py)으로 대체되어
삭제했다 — 사용자 지시.)

`NoAnomalyScorer`는 성격이 다르다 — "메커니즘 없음"이 아니라 **SSF/SPIDER
원 논문이 실제로 쓰는 방식 그 자체**(분류기 자신의 sigmoid(logit) 판정)다.
사용자 지시로 추가: CADE의 MAD 채점 방식(cade_mad)과 비교해 "CADE 채점을
빌려오는 게 실제로 도움이 되는가"를 검증하기 위한 Track A anomaly_scorer의
두 번째 축.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from testbed.base.anomaly_scorer import BaseAnomalyScorer
from testbed.base.anti_forgetting import BaseAntiForgetting
from testbed.base.drift_detector import BaseDriftDetector
from testbed.base.memory_manager import BaseMemoryManager
from testbed.base.models import BaseCLModel
from testbed.base.sample_selector import BaseSampleSelector


class NoDriftDetector(BaseDriftDetector):
    uses_shared_representation = True

    def detect(self, new_data, buf_ref) -> bool:
        return False

    def get_drift_score(self, new_data, buf_ref) -> float:
        return 0.0


class RandomSelector(BaseSampleSelector):
    def select(self, new_data: torch.Tensor, new_labels: torch.Tensor,
               label_budget: int, drift_score: float) -> List[int]:
        n = len(new_data)
        k = min(label_budget, n)
        if k <= 0:
            return []
        idx = torch.randperm(n)[:k]
        return idx.tolist()


class NoMemoryManager(BaseMemoryManager):
    def update(self, selected_data, selected_labels, drift_detected=False) -> None:
        pass

    def get_buffer(self) -> Tuple[None, None]:
        return None, None

    def get_replay_batch(self, batch_size: int) -> Tuple[None, None]:
        return None, None

    def size(self) -> int:
        return 0


class NoAntiForgetting(BaseAntiForgetting):
    """naive fine-tuning baseline — task loss만 사용, 명시적 망각방지 항 없음."""

    backbone_type = "classifier"

    def compute_loss(self, model: BaseCLModel,
                      new_batch: Tuple[torch.Tensor, torch.Tensor],
                      replay_batch: Optional[Tuple[torch.Tensor, torch.Tensor]]
                      ) -> torch.Tensor:
        data, labels = new_batch
        _, _, logit = model(data)
        loss = F.binary_cross_entropy_with_logits(logit.squeeze(-1), labels.float())

        if replay_batch is not None and replay_batch[0] is not None:
            r_data, r_labels = replay_batch
            _, _, r_logit = model(r_data)
            loss = loss + F.binary_cross_entropy_with_logits(
                r_logit.squeeze(-1), r_labels.float())

        return loss


class NoAnomalyScorer(BaseAnomalyScorer):
    """분류기 자체 판정 — SSF/SPIDER 원 논문의 실제 방식(PRD 12.1절 "anomaly_scorer
    전부 z 소비" 원칙의 유일한 예외 — 이 컴포넌트는 z가 아니라 classifier head가
    낸 logit을 직접 쓴다. 그러려면 z만으로는 부족해 model 참조가 필요하다).

    SSF/SPIDER는 별도 anomaly_scorer 없이 classifier head의 sigmoid(logit)로
    직접 판정한다(고정 0.5 threshold, 라벨 불필요). CADE의 MAD 채점(cade_mad)과
    달리 정상 참조 데이터로 별도 학습(fit)할 것이 없다 — 분류기 자체가 이미
    메인 학습 루프(anti_forgetting.compute_loss)에서 학습되기 때문이다.

    CLClient가 표준 계약 밖의 선택적 훅 `set_model()`로 모델 참조를 한 번
    넘겨준다(CNDIDSAntiForgetting.on_experience_start와 같은 패턴). 모델은
    같은 객체가 매 라운드 그 자리에서 갱신되므로 한 번만 받아두면 된다.
    """

    required_backbone = "classifier"

    def __init__(self):
        self._model: Optional[BaseCLModel] = None

    def set_model(self, model: BaseCLModel) -> None:
        self._model = model

    def fit(self, normal_data: torch.Tensor) -> None:
        pass  # 별도 학습 없음 — 분류기 자체가 이미 학습되어 있다.

    def score(self, data: torch.Tensor) -> torch.Tensor:
        # data는 z(잠재표현). classifier head로 logit을 복원해 sigmoid
        # 확률을 점수로 쓴다(높을수록 공격) — SSF/SPIDER가 실제로 보는 값.
        if self._model is None:
            raise RuntimeError("NoAnomalyScorer.set_model()이 호출되지 않았다.")
        with torch.no_grad():
            logit = self._model.classifier(data)
        return torch.sigmoid(logit).squeeze(-1)

    def compute_threshold(self, eval_scores: torch.Tensor,
                           eval_labels: Optional[torch.Tensor]) -> float:
        return 0.5  # SSF/SPIDER 원 논문의 고정 판정 기준 — 라벨도, 정상
                    # 참조 데이터도 쓰지 않는다.
