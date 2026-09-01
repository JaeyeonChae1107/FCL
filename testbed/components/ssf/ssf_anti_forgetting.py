"""SSF AntiForgetting — LwF 스타일 distillation + task loss (PRD 4절/12.5절).

SSF 원 논문 근거: ssf.py의 distillation은 MSE(현재 출력, teacher 출력)이며
(reconstruction/classifier 출력에 직접 적용, 온도 스케일링 없음), 총 손실은
task_loss + lwf_lambda * distillation_loss (ssf.py:45,292-334, 기본
lwf_lambda=0.5). Teacher는 이전 태스크 종료 시점 모델 스냅샷이다
(ssf.py:149,336, on_task_end에서 갱신).

**2026-08-26 발견 — 위 근거는 SSF 원문의 UNSW-NB15 전용 분기다, NSL-KDD
분기는 아예 다른 구조**: 4개 논문 컴포넌트 전수 재감사에서, `ssf.py`가
`if dataset == 'nsl': ... else: <위에서 인용한 코드>` 형태로 데이터셋별로
완전히 다른 모델/손실을 쓴다는 걸 재확인했다 — `dataset=='nsl'` 분기는
분류기 자체가 없는 순수 오토인코더(`AE`, `AE_classifier`가 아님)이고,
BCE도 logit 기반 distillation도 없다(reconstruction MSE distillation만
있음). 즉 위에서 인용한 "SSF의 실제 공식"은 SSF 원문 두 분기 중 하나
(UNSW-NB15용)일 뿐, NSL-KDD 자체 분기와는 다르다 — 이전 문서들이 이
구분 없이 "SSF의 실제 방식"이라고 인용해온 것은 부정확했다. 이 테스트베드는
3개 데이터셋에 공유 backbone(`base/models.py`, 분류기 있음)을 강제로
써야 하므로 데이터셋별 분기를 재현할 수 없다는 제약은 그대로다 — 다만
"이 구현이 NSL-KDD에서도 SSF 원문 그대로"라는 주장은 하지 않으며,
UNSW-NB15 분기를 3개 데이터셋 전부에 일반화 적용한 것임을 명확히
한다.

**2026-08-12 발견·수정 — InfoNCE 손실항 누락**: 위 task_loss는 BCE
하나였는데, SSF 원문의 실제 task_loss는
`weighted_con_loss.mean() + weighted_classification_loss.mean()`
(`ssf.py:310-318`, drift 분기는 `:281-288`)로 InfoNCE 기반 재구성-대조
손실(`ssf_infonce.py` 참고)이 통째로 빠져 있었다. 이 부분은 추가했다 —
A/B 실측(NSL-KDD, dd=none/ss=ssf/mm=none/af=lwf_ssf/as=cade_mad)으로
f1 0.6680→0.7107, bwt -0.0161→-0.0005(거의 완전한 망각 방지)로 뚜렷한
개선을 확인했다.

**new_sample_weight=100은 채택하지 않았다(의도적)**: SSF는 두 손실 항 모두
새로 선택된 대표 표본에 `new_sample_weight=100`(`ssf.py:26`)을 곱하고
나머지(replay)는 1을 쓴다. 이 테스트베드는 new_batch/replay_batch를 이미
크기가 비슷한 별개 배치로 분리하므로(PRD 13절 8단계 설계, 되돌리지 않음),
SSF의 "표본 단위" 가중치는 여기서 "배치 단위" 가중치로 자연스럽게 대응된다
— new_batch 손실 전체에 곱하고 replay_batch 손실 전체에는 1을 곱하는
구조 자체는 맞다. 문제는 **값** 100이다: SSF에서 100은 "누적 풀 전체
(~2.5만, NSL-KDD 기준) 대비 이번 라운드 신규 표본(~200개)"이라는 극단적으로
작은 비율(약 1:125)을 보정하려고 고른 값인데, 이 테스트베드는 new_batch와
replay_batch 크기가 비슷해(≈1:1) 같은 100을 곱하면 gradient 기여도가
약 99:1로 replay가 사실상 무력화된다 — A/B 실측으로 f1 0.7107→0.5655,
bwt -0.0005→-0.1341(망각 급증)까지 나빠짐을 확인했다. `labeling_budget`
(global_hparams.yaml 참고)과 정확히 같은 종류의 함정이라 같은 방식으로
처리했다: 배치 단위 가중치라는 **구조**는 SSF에서 그대로 가져오되, 값은
1.0(신규/과거 동등 취급 — 위 A/B에서 실측 최선)으로 이 테스트베드의
new:old 비율에 맞게 재보정했다. 자세한 수치와 근거는
`configs/component_hparams/ssf.yaml`의 `new_sample_weight` 주석 참고.

**2026-08-14 시도했다가 되돌림 — drift 시 LwF를 끄는 게이팅**: `ssf.py:
262-291`(drift 분기)을 다시 정밀 대조한 결과, teacher_model 호출도
distillation_loss 계산도 `lwf_lambda` 사용도 **전혀 없다** — SSF는 drift가
감지된 라운드에는 과거 지식 보존보다 빠른 적응을 우선하도록 의도적으로
LwF를 끈다. `on_experience_start`(CNDIDSAntiForgetting 참고)와 같은
선택적 훅으로 `set_drift_context()`를 추가해 이 게이팅을 재현해봤으나,
A/B 실측(NSL-KDD, `dd=ssf/ss=ssf/mm=ssf/af=lwf_ssf/as=cade_mad`)으로
f1 0.7306→0.4973, bwt +0.0198→-0.0727로 **크게 악화**됨을 확인해 되돌렸다.

원인으로 보이는 것: SSF 원문은 "drift"를 스트리밍 중 가끔 일어나는 큰
사건으로 가정하고 그때만 적응 우선 모드로 전환하는데, 이 테스트베드의
class-incremental 분할은 설계상 **매 experience가 새로운 공격 카테고리를
도입**한다(`docs/metric_justification.md` "Experience 분할" 절 참고) —
그 결과 라운드 간 분포 차이(K-S 검정 기준)가 거의 항상 유의미해
`drift_detected`가 대부분/전체 라운드에서 True가 되기 쉽고, LwF가 사실상
상시 꺼진 것과 같아진다. SSF가 가정하는 "대체로 안정, 가끔 드리프트"라는
전제 자체가 이 시나리오와 안 맞는 것으로 보인다 — GPM의 residual
projection, CADEMADScorer의 이중 MAD와 같은 종류의 함정(원 논문 메커니즘을
정확히 재현해도 이 테스트베드의 구조적 차이 때문에 오히려 해로운 경우)이라
같은 원칙(실측 우선)으로 처리했다. `set_drift_context()` 훅은 실제로
쓰이지 않아 제거했다 — dd(drift_detector) 슬롯은 여전히 memory_manager
쪽(SSFMemoryManager)에만 영향을 주고 af=lwf_ssf의 손실 자체에는 영향을
주지 않는 상태로 남는다(이 한계는 문서화된 채로 유지).
"""

import copy
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from testbed.base.anti_forgetting import BaseAntiForgetting
from testbed.base.models import BaseCLModel
from testbed.components.ssf.ssf_infonce import ssf_infonce_loss


class SSFAntiForgetting(BaseAntiForgetting):
    backbone_type = "classifier"

    def __init__(self, lwf_lambda: float = 0.5, new_sample_weight: float = 1.0,
                 infonce_temperature: float = 0.02):
        self.lwf_lambda = lwf_lambda
        self.new_sample_weight = new_sample_weight
        self.infonce_temperature = infonce_temperature
        self._teacher: Optional[BaseCLModel] = None

    def _task_loss(self, model: BaseCLModel, data: torch.Tensor,
                    labels: torch.Tensor, weight: float
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ssf.py:281-288/310-318 한 항(new 또는 replay)에 해당 — BCE +
        InfoNCE(recon)를 weight로 스케일한 뒤 더한다. LwF distillation이
        new_batch의 logit도 재사용할 수 있도록 logit을 같이 반환한다(중복
        forward 방지)."""
        _, x_hat, logit = model(data)
        bce = F.binary_cross_entropy_with_logits(logit.squeeze(-1), labels.float())
        loss = weight * bce
        infonce = ssf_infonce_loss(x_hat, labels, self.infonce_temperature)
        if infonce is not None:
            loss = loss + weight * infonce.mean()
        return loss, logit

    def compute_loss(self, model: BaseCLModel,
                      new_batch: Tuple[torch.Tensor, torch.Tensor],
                      replay_batch: Optional[Tuple[torch.Tensor, torch.Tensor]]
                      ) -> torch.Tensor:
        data, labels = new_batch
        loss, logit = self._task_loss(model, data, labels, self.new_sample_weight)

        if replay_batch is not None and replay_batch[0] is not None:
            r_data, r_labels = replay_batch
            r_loss, _ = self._task_loss(model, r_data, r_labels, 1.0)
            loss = loss + r_loss

        # teacher가 없으면(experience 0) LwF 항을 생략한다 — replay_batch=None인
        # 경우와 무관하게, teacher 존재 여부가 유일한 게이팅 조건이다. drift
        # 시 게이팅은 시도했다가 실측 회귀로 되돌렸다(위 모듈 docstring
        # "2026-08-14" 절 참고).
        #
        # **2026-08-26 시도했다가 되돌림 — replay_batch까지 distillation에
        # 포함**: SSF 원문(`ssf.py:296-330`)은 old+new를 합친 전체 배치에
        # distillation을 적용하므로(distillation_loss가 new_batch만이
        # 아니라 전체 배치 기준), replay_batch에도 teacher와의 MSE를
        # 더하도록 확장해봤다. A/B 실측(NSL-KDD, `dd=ssf/ss=ssf/mm=ssf/
        # af=lwf_ssf/as=cade_mad`) 결과 f1 0.6565→0.6128, roc_auc 0.7150→
        # 0.5981로 오히려 나빠졌다 — SSF의 teacher는 매 라운드 그 시점
        # 모델을 통째로 스냅샷한 것이라 replay_batch(과거 라운드 데이터)에
        # 대해서도 teacher와 거리를 좁히라고 강제하면, teacher 자신이
        # replay_batch 위에서 이미 어느 정도 학습된 상태라 distillation
        # 신호가 tautological(자기 자신 재확인)에 가까워지면서 new_batch
        # 쪽 신호(정작 이번 라운드 새로 배워야 할 것)에 갈 gradient
        # 용량을 깎아먹는 것으로 보인다 — GPM residual projection/
        # CADEMADScorer 단일 t_mad와 같은 "원문에 더 충실하지만 이
        # 테스트베드 구조와 안 맞는" 패턴. new_batch만 distillation하는
        # 원래 방식을 유지한다.
        if self._teacher is not None:
            with torch.no_grad():
                _, _, teacher_logit = self._teacher(data)
            distill = F.mse_loss(logit, teacher_logit)
            loss = loss + self.lwf_lambda * distill

        return loss

    def on_task_end(self, model: BaseCLModel) -> None:
        self._teacher = copy.deepcopy(model)
        self._teacher.eval()
        for p in self._teacher.parameters():
            p.requires_grad_(False)
