"""CADEDriftDetector — 독립 표현 소유자 (PRD 4절/12.1절/12.2절).

CADE 원 논문 근거: 클래스별 centroid=학습셋 latent 평균(CADE/cade/detect.py:62),
MAD=1.4826*median(|d-median(d)|)(detect.py:150-158), 샘플의 MAD 정규화 거리
A(x,i)=|‖z_x-centroid_i‖-median(dis_i)|/mad_i(detect.py:91), 최소값이
T_MAD(기본 3.5, utils.py:77-78)를 넘으면 drift로 판정(detect.py:97-104).

uses_shared_representation=False — CLClient는 메인 모델의 z가 아니라 원본
data를 넘긴다. fit()은 자기 소유의 ContrastiveAutoEncoder를 학습시킨다
(PRD 13절 step 3d, selected_data로만 호출).

이 컴포넌트만 메인 모델과 별개로 자기 소유 `nn.Module`(ContrastiveAutoEncoder)
을 가져, CLClient가 `to(device)`를 명시 호출해 옮긴다(`pipeline/cl_client.py`).

**미니배치 학습**: `fit()`이 미니배치 분할 없이 selected_data
전체를 `train_step`에 한 번에 넘겨 "5 epoch"이 실제로는 5회 그래디언트
업데이트였다. CADE 원문(`cade/main.py`, `run_drebin_cade.sh`/
`run_ids_cade.sh`)은 배치 크기(64/512)만 다를 뿐 항상 미니배치를 쓴다 —
표준 미니배치 학습을 추가. batch_size는 `global_hparams.batch_size`(Track A
공유값)를 재사용(원문의 64/512는 이 테스트베드 3개 데이터셋과 대응 안 됨).

**class-aware pairing**: 배치 내 비교 쌍이 무작위 슬라이싱이었다.
CADE 원문(`cade/data.py:268-345`)은 similar_ratio 비율로 same/different-class
쌍을 강제한다 — `contrastive_ae.py`의 `build_paired_batches()`로 이식,
`fit()`이 이 함수를 쓴다.

**다중클래스 family 연결**: CADE 원문 단위는 "정상 + 각 공격
family"(centroid도 family별 `detect.py:62`, pairing도 family 기준
`data.py:268-345`)인데, `fit(data, labels)`의 labels가 이진이라 "정상 vs
전체 공격" 2-클래스로만 학습하고 있었다. `fit_with_category()`를 추가해
다중클래스 `category`(`pipeline/cl_client.py`가 `train_category`를 전달)로
pairing/centroid를 만든다. `category` 문자열→정수 코드(`_category_to_code`)는
라운드를 넘어 고정 — 매 라운드 다시 인코딩하면 같은 코드가 다른 라운드엔
다른 family를 가리키게 된다(`_class_incremental_split` 참고).

**재설계 이력**: 1차 시도(그 라운드 raw 데이터로 centroid를 1회
계산, 이후 라운드에서 갱신 안 함)는 f1 0.7713→0.0804로 붕괴 — encoder가
매 라운드 계속 미세조정되는데(원문은 정적) 등장 안 하는 family의 centroid가
낡은 좌표에 남아 최종 판정에서 정상으로 오판됐다(NSL-KDD exp4는 공격이
없어 정상 centroid만 갱신됨, `data/dataset_loader.py` class-incremental
분할 설계). `category`별 raw 참조 표본을 누적 보관(`_update_category_refs`,
`max_category_ref` 캡)하고 매 라운드 알려진 모든 category의 centroid를
현재 encoder로 재계산(`_recompute_all_centroids`)하도록 수정. A/B 결과는
`docs/metric_justification.md` 참고.
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
        """category(문자열 배열)를 정수 코드로 변환한다 — 새 문자열마다 다음
        정수를 배정하고 인스턴스 수명 전체에 걸쳐 고정, 재사용하지 않는다."""
        codes = []
        for cat in category:
            cat = str(cat)
            if cat not in self._category_to_code:
                self._category_to_code[cat] = len(self._category_to_code)
            codes.append(self._category_to_code[cat])
        return torch.tensor(codes, dtype=torch.long, device=device)

    def fit_with_category(self, data: torch.Tensor, labels: torch.Tensor,
                           category: Sequence) -> None:
        """정상 + 공격 family 단위로 pairing/centroid를 만든다 — `labels`
        (이진)는 이 경로에서 쓰지 않는다."""
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
        """이번 라운드 `data`/`group`에 과거 category 참조 표본
        (`self._category_refs`, 갱신 전이라 전부 과거 표본)을 섞어 대조학습
        배치를 구성한다.

        centroid를 매 라운드 재계산해도(`_recompute_all_centroids`) f1=0
        이었다 — encoder가 매 라운드 그 라운드의 2-클래스(정상 vs 해당
        family)로만 미세조정되며 과거 family 구분 능력을 잃는다(실측: NSL-KDD
        5라운드 누적 min_anomaly_score 최댓값이 14.98→3.83→3.06→1.87→2.55로
        단조 감소, f1=0.0). CADE 원문은 전체 family를 한 번에 학습해 이
        문제가 없다 — 이 테스트베드는 family가 라운드마다 하나씩만 등장하므로
        과거 family를 리허설시켜야 encoder가 구분을 유지한다. 메인 모델의
        리플레이 계약과 별개로 이 사설 encoder 전용으로 구현."""
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
        """category별 raw 참조 표본을 인스턴스 수명 전체에 걸쳐 누적 보관,
        최근 `max_category_ref`개만 유지(신선도는 `_recompute_all_centroids`가
        매 라운드 현재 encoder로 재인코딩하므로 이 캡이 결정).

        CADE는 정적 설계(encoder 1회 학습 후 centroid 1회 계산)라 원문엔 이
        참조 버퍼가 없다. 이 테스트베드는 encoder를 매 라운드 미세조정하므로,
        raw 데이터로 그 라운드에 1회만 centroid를 계산하면 그 이후 안 나오는
        category의 centroid가 이후 라운드의 encoder 좌표계와 어긋난다(A/B
        실측: NSL-KDD 순정 CADE 콤보 f1 0.7713→0.0804 붕괴 —
        `docs/metric_justification.md`)."""
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
        """알려진 모든 category(이번 라운드에 없던 것 포함)의 centroid/
        median/MAD를 현재 encoder로 매번 다시 계산한다."""
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
            # 절대 floor(1e-8)는 거리 스케일에 비해 작아 MAD→0인 희소
            # category에서 점수가 폭발할 수 있다 — median 비례 floor(1%)를
            # 함께 적용.
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
        # CADE 원문(detect.py:97-104)은 샘플 단위 판정만 정의(min_anomaly_score
        # > t_mad). "라운드 전체가 drift인가"는 원문에 없어, 과반 표본이 기준을
        # 넘으면 drift로 보는 다수결 집계를 추가(BaseDriftDetector 계약용,
        # 테스트베드 자체 발명).
        #
        # 2026-09-03 수정 — 게이트를 buf_ref(memory_manager 버퍼)가 아니라
        # self._centroids(이 컴포넌트 자신의 상태)로 바꾼다. buf_ref는
        # SSFDriftDetector처럼 "비교할 과거 표본"이 버퍼에 있어야 판정되는
        # 공유 표현 소비자를 위한 게이트인데, CADEDriftDetector는
        # uses_shared_representation=False로 애초에 buf_ref를 쓰지 않고
        # 자기 소유의 centroid로 판정한다(min_anomaly_score 참고). 그런데
        # mm=none 조합(memory_manager.get_buffer()가 항상 (None, None))에서는
        # centroid가 몇 라운드째 실제로 학습되고 있어도 buf_ref가 항상 None이라
        # detect()/get_drift_score()가 무조건 False/0.0을 반환했다 — 정확히
        # component_registry.py가 "순정 CADE"로 부르는
        # dd=cade/ss=random/mm=none/af=none/as=cade_mad 조합이 여기 해당된다.
        # (sample_selector, memory_manager) 조건부 제약(common/compatibility.py
        # TRACK_A_DD_ACTIVE_SS_MM)상 mm=none은 이미 drift 신호가 다운스트림에
        # 소비되지 않는 조합이라 F1/PR-AUC/BWT 등 기존 그리드 결과에는 영향이
        # 없다 — 바뀌는 건 drift_detected_per_round/n_drift_detected 진단
        # 리포팅뿐이다(grid_runner.py 2026-09-03 절 참고).
        if not self._centroids:
            return False
        min_scores = self.min_anomaly_score(new_data)
        if min_scores is None:
            return False
        return bool((min_scores > self.t_mad).float().mean().item() > 0.5)

    def get_drift_score(self, new_data: torch.Tensor, buf_ref: Optional[torch.Tensor]) -> float:
        # 2026-09-03 수정 — detect()와 동일한 이유로 게이트를 self._centroids로
        # 바꾼다(위 detect() 주석 참고).
        if not self._centroids:
            return 0.0
        min_scores = self.min_anomaly_score(new_data)
        if min_scores is None:
            return 0.0
        return float(min_scores.mean().item())
