"""CADEDriftDetector — 독립 표현 소유자 (PRD 4절/12.1절/12.2절).

CADE 원 논문 근거: 클래스별 centroid=학습셋 latent 평균(CADE/cade/detect.py:62),
MAD=1.4826*median(|d-median(d)|)(detect.py:150-158), 샘플의 MAD 정규화 거리
A(x,i)=|‖z_x-centroid_i‖-median(dis_i)|/mad_i(detect.py:91), 최소값이
T_MAD(기본 3.5, utils.py:77-78)를 넘으면 drift로 판정(detect.py:97-104).

12.1절 — uses_shared_representation=False. CLClient는 메인 모델의 z가 아니라
원본 data를 그대로 넘긴다. fit()은 자기 소유의 ContrastiveAutoEncoder를
직접 학습시킨다(PRD 13절 step 3d — 라벨이 공개된 selected_data로만 호출).

**GPU 이식성**: 이 컴포넌트는 메인 모델(BaseCLModel)과 별개로 자기 소유의
`nn.Module`(ContrastiveAutoEncoder)을 갖는 유일한 컴포넌트다 — CLClient는
`self.model`만 `.to(device)`로 옮기므로, 이 사설 encoder는 CLClient가 명시적으로
`to(device)`를 호출해줘야 올바른 디바이스로 옮겨간다(`pipeline/cl_client.py`
참고, 다른 컴포넌트에는 없는 훅).

**2026-08-11 발견·수정 — 미니배치 학습 누락**: `fit()`이 `encoder_epochs`회
반복하며 매번 `train_step`에 selected_data 전체를 미니배치 분할 없이 한
번에 넘기고 있었다 — 즉 "5 epoch"이 아니라 총 5회의 그래디언트 업데이트에
불과했다. CADE 원문(`cade/main.py`, `run_drebin_cade.sh`/`run_ids_cade.sh`)을
재대조한 결과 두 데이터셋(Drebin/IDS2018) 모두 배치 크기(64/512)만 다를 뿐
미니배치 자체는 항상 쓴다 — 이미 문서화된 "250→5 epoch 축소"는 epoch 수
축소에 대한 근거였지 미니배치 자체를 없애는 근거가 아니었다. 표준 미니배치
학습(매 epoch마다 셔플 후 batch_size 단위 분할)을 추가했다. batch_size는
CADE 원문의 Drebin/IDS2018 값(64/512)이 이 테스트베드의 3개 데이터셋과
깔끔하게 대응되지 않으므로(NSL-KDD·UNSW-NB15는 CADE 원 논문이 평가하지
않은 데이터셋) 새로 추측해 채우지 않고, Track A가 이미 공유하는
`global_hparams.batch_size`를 그대로 재사용한다.

**2026-08-12 발견·수정 — class-aware pairing 누락**: 위 미니배치 수정은
"몇 스텝 학습하는가"만 고쳤을 뿐, 각 배치 안에서 비교 쌍이 어떻게 만들어지는지는
그대로 무작위 셔플 후 슬라이싱이었다. CADE 원문(`cade/data.py:268-345`)은
배치 구성 자체에서 similar_ratio 비율로 same/different-class 쌍을 강제한다
— 자세한 내용과 이식 방법은 `contrastive_ae.py`의 `build_paired_batches()`
참고. `fit()`이 이제 이 함수를 쓴다.

**2026-08-25 발견·수정 — 이진 정상/공격만으로 학습해 CADE의 family 구조가
빠져 있던 문제**: CADE 원문의 실제 단위는 이진이 아니라 "정상 + 각 공격
family"다 — centroid도 family별(`detect.py:62`), contrastive pairing의
same/diff도 family 기준(`data.py:268-345`)이다. 그런데 이 컴포넌트는
`fit(data, labels)`의 `labels`가 항상 이진(y=0/1)이라 "정상 vs 전체 공격"
2-클래스로만 대조학습하고 centroid도 2개(정상/공격 뭉뚱그림)만 만들고
있었다 — 서로 성격이 다른 DoS/Probe/R2L/U2R를 "공격"이라는 한 뭉치로
묶으면 그 안에서 대조학습 신호가 사라지고, CADE의 핵심인 "이 표본이 어떤
family와 가장 가까운가"라는 다중클래스 최근접 판정 자체가 성립하지 않는다.
`fit_with_category()`를 추가해 (있으면) 다중클래스 `category`로 pairing과
centroid를 만들도록 했다 — `pipeline/cl_client.py`가 dataset_loader가 노출한
`train_category`를 이걸로 넘긴다. category 문자열은 class-incremental
분할 때문에 라운드마다 등장 family가 달라지므로(`data/dataset_loader.py`의
`_class_incremental_split` 참고), `_category_to_code`로 문자열→정수 코드를
**라운드를 넘어 고정**한다 — 매 라운드 새로 인코딩하면 같은 정수 코드가
다른 라운드엔 다른 family를 가리키게 되어 centroid의 의미가 뒤섞인다.

**2026-08-25 1차 시도 실패·재설계 — centroid를 "그 라운드 raw 데이터로 딱
한 번" 계산했다가 f1 0.7713→0.0804(recall 0.66→0.04)로 붕괴**: 처음엔
`group.unique()`로 이번 라운드에 있는 category만 그 라운드의 `data`로
centroid를 계산하고, 없어진 category는 `.clear()` 없이 이전 값을 그대로
남기는 방식이었다. 그런데 이 encoder는(CADE 원문과 달리) 매 라운드 계속
미세조정된다 — 한 번 등장했다가 사라진 family(예: NSL-KDD의 DoS→exp0 이후
다시는 등장 안 함)의 centroid는 그 등장 라운드의 encoder 좌표에 박제된 채
남는데, encoder는 이후 라운드에도 계속 움직인다. 특히 NSL-KDD 마지막
experience(exp4)는 공격이 전혀 없어(class-incremental 분할 설계상 자연
발생, `data/dataset_loader.py` 참고) 그 라운드는 "정상끼리만 뭉치기"
대조학습만 수행 — 정상 centroid만 최신으로 갱신되고 DoS/Probe/R2L/U2R
centroid는 몇 라운드 전 좌표에 남아, 최종 판정 시 공격 표본이 자신의
(낡은) family centroid보다 (방금 갱신된) 정상 centroid에 우연히 더
가깝게 나와 정상으로 오판되는 것으로 확인했다 — "원문에 더 충실하지만
이 구조와 안 맞는" 문제가 아니라 순수한 구현 결함이었다. `category`별
raw 참조 표본을 인스턴스 수명 전체에 걸쳐 누적 보관했다가(`_update_
category_refs`, 최근 `max_category_ref`개 캡) 매 라운드 **알려진 모든
category**의 centroid를 **현재** encoder로 다시 계산(`_recompute_all_
centroids`)하도록 재설계해 이 어긋남을 없앴다 — 원 논문에 없는, 이
테스트베드의 "encoder가 계속 움직인다"는 구조적 차이를 보정하기 위한
전용 장치다. 재설계 후 A/B 결과는 `docs/metric_justification.md` 참고.
"""

from typing import Dict, Optional, Sequence

import torch

from testbed.base.drift_detector import BaseDriftDetector
from testbed.components.cade.contrastive_ae import (
    ContrastiveAutoEncoder, build_paired_batches, train_step)


class CADEDriftDetector(BaseDriftDetector):
    uses_shared_representation = False

    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 32,
                 t_mad: float = 3.5, contrastive_margin: float = 10.0,
                 contrastive_lambda: float = 0.1, encoder_epochs: int = 5,
                 encoder_lr: float = 1e-3, batch_size: int = 128,
                 similar_ratio: float = 0.25, max_category_ref: int = 500):
        self.t_mad = t_mad
        self.margin = contrastive_margin
        self.lam = contrastive_lambda
        self.epochs = encoder_epochs
        self.batch_size = batch_size
        self.similar_ratio = similar_ratio
        self.max_category_ref = max_category_ref
        self._device = torch.device("cpu")
        self._encoder = ContrastiveAutoEncoder(input_dim, hidden_dim, latent_dim)
        self._optimizer = torch.optim.Adam(self._encoder.parameters(), lr=encoder_lr)
        self._centroids: Dict[int, torch.Tensor] = {}
        self._median: Dict[int, torch.Tensor] = {}
        self._mad: Dict[int, torch.Tensor] = {}
        self._category_to_code: Dict[str, int] = {}
        self._category_refs: Dict[int, torch.Tensor] = {}

    def to(self, device) -> "CADEDriftDetector":
        """CLClient가 생성 직후 호출하는 선택적 훅 — 사설 encoder를 메인
        모델과 같은 디바이스로 옮긴다(`hasattr(component, 'to')` 패턴,
        `NoAnomalyScorer.set_model` 등과 같은 방식)."""
        self._device = torch.device(device)
        self._encoder.to(self._device)
        return self

    def fit(self, data: torch.Tensor, labels: torch.Tensor) -> None:
        self._fit_impl(data, labels)

    def _encode_category(self, category: Sequence, device: torch.device) -> torch.Tensor:
        """category(문자열 배열)를 정수 코드로 변환한다 — 새 문자열을 만날
        때마다 다음 정수를 배정하고 **절대 재사용하지 않는다**(인스턴스
        수명 전체에 걸쳐 고정, 위 모듈 docstring "2026-08-25" 절 참고)."""
        codes = []
        for cat in category:
            cat = str(cat)
            if cat not in self._category_to_code:
                self._category_to_code[cat] = len(self._category_to_code)
            codes.append(self._category_to_code[cat])
        return torch.tensor(codes, dtype=torch.long, device=device)

    def fit_with_category(self, data: torch.Tensor, labels: torch.Tensor,
                           category: Sequence) -> None:
        """CADE의 실제 grouping 단위(정상 + 공격 family)로 pairing/centroid를
        만든다 — `labels`(이진)는 이 경로에서 쓰지 않는다(위 모듈 docstring
        "2026-08-25" 절 참고)."""
        if len(data) < 2:
            return
        group = self._encode_category(category, data.device)
        self._fit_impl(data, group)

    def _fit_impl(self, data: torch.Tensor, group: torch.Tensor) -> None:
        if len(data) < 2:
            return
        train_data, train_group = self._replay_known_categories(data, group)
        for _ in range(self.epochs):
            batch_count, paired_data, paired_group = build_paired_batches(
                train_data, train_group, self.batch_size, self.similar_ratio)
            for b in range(batch_count):
                train_step(self._encoder, self._optimizer, paired_data[b], paired_group[b],
                           self.margin, self.lam)

        self._update_category_refs(data, group)
        self._recompute_all_centroids()

    def _replay_known_categories(self, data: torch.Tensor, group: torch.Tensor
                                  ) -> "tuple[torch.Tensor, torch.Tensor]":
        """이번 라운드 `data`/`group`에 과거 라운드에서 저장해 둔 category
        참조 표본(`self._category_refs`, 아직 이번 라운드분으로 갱신 전이라
        전부 "과거" 표본)을 섞어 대조학습 배치를 구성한다.

        2026-08-25 2차 재설계 배경 — centroid를 매 라운드 현재 encoder로 다시
        계산하도록 고쳐도(`_recompute_all_centroids`) 여전히 f1=0(recall=0)
        이었다: encoder 자체가 매 라운드 "이번 라운드의 2-클래스(정상 vs
        그 라운드의 공격 family 하나)"만으로 계속 미세조정되면서, 이번
        라운드에 없는 과거 family와 정상을 구분하는 능력을 잃어간다(대조학습
        손실 자체가 이번 라운드 구성에만 반응하므로) — 그 결과 라운드가
        진행될수록 오래된 family의 재인코딩 latent가 점점 "정상"과 가까워져
        `min_anomaly_score`가 전 표본에 대해 계속 작아지고, 마지막 라운드
        threshold가 그 압축된 스케일 위에서 계산돼 어떤 표본도 넘지 못했다
        (실측: NSL-KDD 5라운드 누적 min_anomaly_score 최댓값이 14.98→3.83→
        3.06→1.87→2.55로 단조 감소, 그 결과 f1=0.0). CADE 원문은 애초에
        전체 family를 한 번에 학습하므로 이 문제가 없다 — 이 테스트베드는
        family가 라운드마다 하나씩만 등장하므로, 과거 family를 계속
        "리허설"시켜야 encoder가 그 구분을 유지한다. 리플레이 버퍼
        (`BaseMemoryManager`)와 같은 개념이지만 메인 모델의 리플레이
        계약과는 별개로, 이 사설 encoder 전용으로 최소하게 구현했다."""
        extra_data, extra_group = [], []
        for code, ref in self._category_refs.items():
            extra_data.append(ref)
            extra_group.append(torch.full((len(ref),), code, dtype=torch.long, device=data.device))
        if not extra_data:
            return data, group
        combined_data = torch.cat([data] + extra_data, dim=0)
        combined_group = torch.cat([group] + extra_group, dim=0)
        return combined_data, combined_group

    def _update_category_refs(self, data: torch.Tensor, group: torch.Tensor) -> None:
        """category(또는 이진 label)별 raw 참조 표본을 인스턴스 수명 전체에
        걸쳐 누적 보관한다 — 최근 `max_category_ref`개만 유지(오래된 표본을
        버리는 게 아니라, 어차피 아래 `_recompute_all_centroids`가 매 라운드
        **현재** encoder로 다시 인코딩하므로 표본 자체의 신선도는 이 캡이
        결정하는 유일한 요소다).

        2026-08-25 도입 배경(중요) — CADE는 원래 정적 설계(한 번 학습한
        encoder로 전체 코퍼스를 인코딩해 family centroid를 딱 한 번 계산)라
        "raw 표본을 나중에 다시 인코딩"할 필요 자체가 없다. 이 테스트베드는
        encoder를 매 라운드 계속 미세조정하는 지속학습 구조라, 첫 시도(raw
        데이터로 그 라운드에 딱 한 번만 centroid를 계산해 저장)는 **그
        라운드 이후로 없어진 category의 centroid가 이후 라운드의(이미 이동한)
        encoder 좌표계와 어긋나는** 치명적 결함으로 이어졌다(A/B 실측:
        NSL-KDD 순정 CADE 콤보 f1 0.7713→0.0804, recall 0.66→0.04로 붕괴 —
        `docs/metric_justification.md` 참고). 특히 NSL-KDD의 마지막 experience는
        공격이 전혀 없어(`data/dataset_loader.py`의 class-incremental 분할
        설계상 자연 발생) 그 라운드의 대조학습이 "정상끼리만 뭉치기"만
        수행하면서 정상 centroid만 갱신되고 나머지 family는 몇 라운드 전
        좌표에 그대로 남아 기하가 완전히 어긋났다. 원문에 없는 이 참조 버퍼는
        그 어긋남을 막기 위한, 이 테스트베드 고유의 보정 장치다."""
        for c in group.unique():
            code = int(c.item())
            mask = group == c
            if mask.sum() == 0:
                continue
            new_samples = data[mask].detach()
            if code in self._category_refs:
                combined = torch.cat([self._category_refs[code], new_samples], dim=0)
            else:
                combined = new_samples
            if len(combined) > self.max_category_ref:
                combined = combined[-self.max_category_ref:]
            self._category_refs[code] = combined

    def _recompute_all_centroids(self) -> None:
        """지금까지 알려진 **모든** category(이번 라운드에 없던 것 포함)의
        centroid/median/MAD를 현재 encoder로 매번 다시 계산한다 — 위
        `_update_category_refs` docstring의 근거 참고."""
        self._encoder.eval()
        for code, ref_data in self._category_refs.items():
            with torch.no_grad():
                z, _ = self._encoder(ref_data)
            centroid = z.mean(dim=0)
            dist = torch.norm(z - centroid, dim=1)
            median = dist.median()
            mad = 1.4826 * (dist - median).abs().median()
            self._centroids[code] = centroid
            self._median[code] = median
            # 2026-08-26 발견·수정(재감사) — 절대 상수 1e-8 floor는 실제 거리
            # 스케일(보통 0.1~수십)에 비해 지나치게 작아서, 참조 표본이 서로
            # 거의 중복이라 MAD가 0에 가까워지는 희소 category(예: NIDS
            # 흐름 레코드의 근접 중복)에서 `(dist-median)/mad`가 물리적
            # 의미 없는 값(~1e8~1e9)까지 폭발할 수 있다(합성 데이터로 재현
            # 확인). 그 category 자신의 거리 스케일(median)에 비례한 floor
            # (1%)를 절대 floor와 함께 적용해, 스케일이 큰 category는 그
            # 스케일에 맞는 floor를, 전부 동일한 극단적 퇴화 상황(median도
            # 0)에서는 기존 절대 floor를 쓴다.
            mad_floor = torch.clamp(0.01 * median.abs(), min=1e-8)
            self._mad[code] = torch.clamp(mad, min=mad_floor)

    def min_anomaly_score(self, data: torch.Tensor) -> Optional[torch.Tensor]:
        if not self._centroids:
            return None
        self._encoder.eval()
        with torch.no_grad():
            z, _ = self._encoder(data)
        scores = []
        for c, centroid in self._centroids.items():
            dist = torch.norm(z - centroid, dim=1)
            anomaly = (dist - self._median[c]).abs() / self._mad[c]
            scores.append(anomaly)
        stacked = torch.stack(scores, dim=1)
        min_scores, _ = stacked.min(dim=1)
        return min_scores

    def detect(self, new_data: torch.Tensor, buf_ref: Optional[torch.Tensor]) -> bool:
        # CADE 원문(detect.py:97-104)은 샘플 단위 판정만 정의한다
        # (min_anomaly_score > t_mad ⇒ 그 샘플이 drift). 이 메서드가 요구하는
        # "라운드 전체가 drift인가"라는 bool 반환은 원문에 없는 개념이라,
        # 이 테스트베드가 "이번 라운드 표본의 과반이 판정 기준을 넘으면
        # drift"라는 다수결 집계를 자체적으로 추가했다(BaseDriftDetector
        # 계약을 만족시키기 위한 테스트베드 발명 — 원 논문 인용이 아님).
        if buf_ref is None:
            return False
        min_scores = self.min_anomaly_score(new_data)
        if min_scores is None:
            return False
        return bool((min_scores > self.t_mad).float().mean().item() > 0.5)

    def get_drift_score(self, new_data: torch.Tensor, buf_ref: Optional[torch.Tensor]) -> float:
        if buf_ref is None:
            return 0.0
        min_scores = self.min_anomaly_score(new_data)
        if min_scores is None:
            return 0.0
        return float(min_scores.mean().item())
