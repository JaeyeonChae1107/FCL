"""SSF AntiForgetting — LwF 스타일 distillation + task loss (PRD 4절/12.5절).

SSF 원 논문 근거: ssf.py의 classifier는 `nn.Sigmoid()`로 끝나고(utils.py:
53-56), distillation은 MSE(sigmoid(현재 출력), sigmoid(teacher 출력)) —
post-sigmoid 확률 위에서 계산된다(`ssf.py:330`의 변수명 `teacher_logits`는
오해 소지가 있으나 실제로는 sigmoid 통과 값). 총 손실은 task_loss +
lwf_lambda * distillation_loss (ssf.py:45,292-334, 기본 lwf_lambda=0.5).
Teacher는 이전 태스크 종료 시점 모델 스냅샷이다(ssf.py:149,336,
on_task_end에서 갱신).

**post-sigmoid MSE로 문자 그대로 맞췄다가 되돌림 — 시그모이드 포화로 인한
distillation 신호 소실**: `torch.sigmoid(logit)` 위에서 MSE를 계산하도록
고쳐 NSL-KDD 4개 조합(af=lwf_ssf, as∈{cade_mad,none} × mm∈{none,ssf})으로
A/B 실측했더니 4개 전부에서 bwt가 뚜렷이 악화됐다(예:
`dd=none/mm=none/as=none`: bwt -0.013→-0.603, f1 0.601→0.017까지 붕괴;
`dd=ssf/mm=ssf/as=cade_mad`: bwt +0.019→-0.142, f1 0.652→0.636).
`lwf_lambda`를 0.5→10.0까지 스윕해도 어떤 값도 원래 bwt/f1을 회복하지
못했다(예: `as=cade_mad` 조합, λ=10: f1 0.466, bwt -0.100 — 그나마 나은
값도 원본 -0.014에 못 미침). 원인: bce로 200 epoch 학습하면 classifier
logit이 빠르게 커져 sigmoid가 0/1 근처로 포화되고, `sigmoid'(x) =
sigmoid(x)(1-sigmoid(x))`가 0에 가까워지면서 MSE(sigmoid(·), sigmoid(·))의
그래디언트 자체가 소실된다 — teacher/student가 이미 확신에 찬(포화된)
예측을 할 때일수록(=과거 지식을 보존해야 할 때일수록) distillation
그래디언트가 사라지는 구조적 결함이라 `lwf_lambda`를 아무리 키워도
"거의 0을 상수배"한 값이라 보완이 안 된다. 반면 raw logit MSE는 포화가
없어 disagreement 크기에 비례하는 그래디언트를 항상 제공한다 — GPM
residual projection/CADEMADScorer 이중 MAD/드리프트 게이팅과 같은 종류의
"원문에 더 충실하지만 이 테스트베드 구조(200 epoch, 공유 backbone)와
안 맞는" 패턴이라 raw logit MSE를 유지한다.

**위 근거는 SSF 원문의 UNSW-NB15 전용 분기다, NSL-KDD 분기는 다른 구조**:
`ssf.py`가 `if dataset == 'nsl': ... else: <위에서 인용한 코드>` 형태로
데이터셋별로 완전히 다른 모델/손실을 쓴다 — `dataset=='nsl'` 분기는
분류기 자체가 없는 순수 오토인코더(`AE`, `AE_classifier`가 아님)이고,
BCE도 logit 기반 distillation도 없다(reconstruction MSE distillation만
있음). 즉 위에서 인용한 "SSF의 실제 공식"은 SSF 원문 두 분기 중 하나
(UNSW-NB15용)일 뿐, NSL-KDD 자체 분기와는 다르다. 이 테스트베드는 3개
데이터셋에 공유 backbone(`base/models.py`, 분류기 있음)을 강제로 써야
하므로 데이터셋별 분기를 재현할 수 없다는 제약은 그대로다 — "이 구현이
NSL-KDD에서도 SSF 원문 그대로"라는 주장은 하지 않으며, UNSW-NB15 분기를
3개 데이터셋 전부에 일반화 적용한 것임을 명확히 한다.

**InfoNCE 손실항 누락**: 위 task_loss는 BCE 하나였는데, SSF 원문의 실제
task_loss는 `weighted_con_loss.mean() + weighted_classification_loss.mean()`
(`ssf.py:310-318`, drift 분기는 `:281-288`)로 InfoNCE 기반 재구성-대조
손실(`ssf_infonce.py` 참고)이 빠져 있었다. 추가했다 — A/B 실측(NSL-KDD,
dd=none/ss=ssf/mm=none/af=lwf_ssf/as=cade_mad): f1 0.6680→0.7107, bwt
-0.0161→-0.0005(거의 완전한 망각 방지)로 뚜렷한 개선.

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

**drift 시 LwF를 끄는 게이팅을 시도했다가 되돌림**: `ssf.py:262-291`
(drift 분기)은 teacher_model 호출도 distillation_loss 계산도 `lwf_lambda`
사용도 전혀 없다 — SSF는 drift가 감지된 라운드에는 과거 지식 보존보다
빠른 적응을 우선하도록 의도적으로 LwF를 끈다. `set_drift_context()`를
추가해 이 게이팅을 재현해봤으나, A/B 실측(NSL-KDD, `dd=ssf/ss=ssf/mm=ssf/
af=lwf_ssf/as=cade_mad`)으로 f1 0.7306→0.4973, bwt +0.0198→-0.0727로
크게 악화됨을 확인해 되돌렸다.

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
        # 시 게이팅은 시도했다가 실측 회귀로 되돌렸다(모듈 docstring 참고).
        #
        # replay_batch까지 distillation에 포함하는 것도 시도했다가 되돌림:
        # SSF 원문(`ssf.py:296-330`)은 old+new를 합친 전체 배치에
        # distillation을 적용하므로 replay_batch에도 teacher와의 MSE를
        # 더하도록 확장해봤으나, A/B 실측(NSL-KDD, `dd=ssf/ss=ssf/mm=ssf/
        # af=lwf_ssf/as=cade_mad`) 결과 f1 0.6565→0.6128, roc_auc 0.7150→
        # 0.5981로 나빠졌다 — teacher는 매 라운드 그 시점 모델을 스냅샷한
        # 것이라 replay_batch(과거 라운드 데이터)에도 teacher와 거리를
        # 좁히라고 강제하면, teacher 자신이 이미 replay_batch로 학습된
        # 상태라 distillation 신호가 tautological해지면서 new_batch 쪽
        # 신호(이번 라운드 새로 배워야 할 것)의 gradient 용량을 깎아먹는
        # 것으로 보인다. new_batch만 distillation하는 원래 방식을 유지한다.
        if self._teacher is not None:
            with torch.no_grad():
                _, _, teacher_logit = self._teacher(data)
            # post-sigmoid MSE(SSF 원문 그대로)는 시그모이드 포화로
            # distillation 그래디언트가 소실돼 실측 회귀 — 모듈 docstring
            # 참고. raw logit MSE를 유지한다.
            distill = F.mse_loss(logit, teacher_logit)
            loss = loss + self.lwf_lambda * distill

        return loss

    def on_task_end(self, model: BaseCLModel) -> None:
        self._teacher = copy.deepcopy(model)
        self._teacher.eval()
        for p in self._teacher.parameters():
            p.requires_grad_(False)
