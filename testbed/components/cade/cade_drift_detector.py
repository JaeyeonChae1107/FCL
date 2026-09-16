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

**사설 encoder 차원이 공유 backbone에 조용히 덮어써지던 결함**:
`cl_client.py`가 `component_hparams`(cade.yaml 포함)를 먼저 병합한 뒤
전역 backbone 차원(`input_dim`/`hidden_dim`/`latent_dim`)으로 마지막에
`.update()` 덮어쓰기를 한다 — 생성자 파라미터 이름이 하필 `hidden_dim`/
`latent_dim`과 겹쳐 `cade.yaml`에 무슨 값을 넣어도 무시되고 공유
backbone(SSF 공식) 차원이 그대로 들어갔다. 전체 컴포넌트 생성자 전수
대조 결과 이 이름 충돌은 이 클래스가 유일했다. `encoder_hidden_dim`/
`encoder_latent_dim`으로 개명해 전역 병합 키와 더 이상 겹치지 않게
했다 — 값 근거는 `configs/component_hparams/cade.yaml` 참고. 영향은
`dd=cade`+`as=cade_mad` 조합에만 미친다(`as=cade_mad`가 `dd=cade` 없이
쓰이면 공유 backbone의 z를 그대로 쓰고 이 사설 encoder를 거치지 않음).
A/B(NSL-KDD, `dd=cade/ss=random/mm=none/af=none/as=cade_mad`): 이전엔
공유 backbone 차원(NSL-KDD 기준 hidden=64/latent=32, SSF 공식)이 그대로
들어가 있었는데(f1 0.5687, bwt 0.2352), 독립화 후 `encoder_hidden_dim=64/
encoder_latent_dim=16`으로는 f1 0.5362, bwt 0.2058로 소폭(3%p 내외) 낮다
— latent 차원 축소(32→16)로 사설 encoder의 표현력이 줄어든 영향으로
보이나, 이전 값이 "의도한 설정"이 아니라 우연히 덮어써진 값이었다는
점을 감안하면 이 정도 차이는 감내 가능한 트레이드오프로 판단했다(버그
자체 — yaml 값이 반영되지 않는 문제 — 를 고치는 것이 우선).

**라벨 예산 면제(consumes_full_round_data, 2026-09-12, docs/design_decisions.md 5절)**:
CADE 원 논문(`cade/utils.py` argparse 전체 확인)에는 라벨 예산 개념이
전혀 없다 — 학습셋이 이미 충분히 라벨링되어 있다고 전제한다. `fit_with_
category()`를 `selected_data`(라벨 예산 10%) 대신 `new_data`(그 라운드
전체)로 호출하도록 `pipeline/cl_client.py` Step 3을 바꿨다.

1차 시도(이 플래그만 추가, `_replay_known_categories`는 그대로)는 NSL-KDD
smoke A/B(`dd=cade/ss=random/mm=none/af=none/as=cade_mad`)로 뚜렷한 회귀를
냈다 — f1 0.2688→0.1345, bwt -0.0738→-0.4402. 원인: `new_data`가 최대
10배 커지는데 `_replay_known_categories`가 합치는 과거 category
replay(`max_category_ref=500`으로 캡, 고정 크기)는 그대로라, `build_paired_
batches`의 anchor 추출(`torch.randperm`, 풀 크기 비례)이 압도적으로
"이번 라운드"에서만 뽑혀 replay 비중이 희석됨 — 정확히 이 클래스의
"재설계 이력" 절이 replay를 도입한 이유와 같은 실패 모드가 재발한 것.

수정: 이번 라운드 `data`는 자르지 않고(카테고리당 500건으로 캡하는
대안은 라운드 데이터가 캡보다 훨씬 크면 오히려 라벨 예산보다도 적은
양만 학습하게 되어 되돌림 — round1/2 자체 recall 붕괴 실측), 대신
`_replay_known_categories`가 replay를 `target_replay_fraction`
(cade.yaml, 기본 0.15)만큼 복원추출로 오버샘플링해 결합 anchor 풀에서
최소 그 비율을 차지하게 보정한다. NSL-KDD smoke 스케일에서 {0.08, 0.15,
0.20, 0.30, 0.50} 스윕 결과 **0.15가 국소최적**(비단조): f1=0.4405/
bwt=-0.1306/pr_auc=0.7334/roc_auc=0.7058 — 0.08과 0.20~0.50 모두 이보다
나쁨(값 근거는 cade.yaml `target_replay_fraction` 참고). 면제 이전
기준선(f1=0.2688/bwt=-0.0738/pr_auc=0.6326/roc_auc=0.6471) 대비 f1/
pr_auc/roc_auc는 뚜렷이 개선, bwt만 다소 나쁘다.

더 복잡한 콤보(`dd=cade/ss=ssf/mm=ssf/af=lwf_ssf/as=cade_mad`)에서도
0.15로 재검증: f1 0.2901→0.4478(개선), pr_auc 0.6640→0.7219(개선),
roc_auc 0.6944→0.6852(거의 동일), bwt +0.1045→-0.1078(악화). 다만 면제
이전 이 콤보의 라운드별 자체 성능(perf_matrix 대각선)이 [0.424, 0.006,
0.131, 0.487]로 round1이 사실상 붕괴 상태였다 — bwt가 양수였던 건
"애초에 배울 게 별로 없어 잃을 것도 없었다"에 가깝다. 면제 후 대각선은
[0.431, 0.691, 0.556, 0.441]로 전 라운드가 건강하게 학습되고, 그 상태에서
측정되니 "잃을 게 생겨서" bwt가 상대적으로 나빠진 것으로 해석한다 — f1/
pr_auc가 뚜렷이 개선됐으므로 순net 개선으로 채택.

**한계**: 단일 시드·NSL-KDD 단일 데이터셋·smoke 스케일(epoch 10, 행 캡
20k/5k) 스윕으로 찾은 값이라 과적합 위험이 있다 — CADE 사설 encoder는
`encoder_epochs=5`로 고정이라 메인 모델 epoch 수(smoke 10 vs 실전 200)와
무관하지만(이 컴포넌트 자체는 스케일 영향이 적음), 6.8절 멀티시드
실행 시 재검증 대상.

**생성자 기본값과 yaml 불일치(2026-09-14 재검토로 발견, 수정)**: 위
"사설 encoder 차원이 조용히 덮어써지던 결함"을 고치면서 인자명을
`encoder_hidden_dim`/`encoder_latent_dim`으로 개명했는데, `__init__`
기본값은 개명 이전의 "우연히 덮어써지던" 값(128/32)에 그대로 남아있고
`cade.yaml`만 의도한 값(64/16)으로 갱신돼 있었다. 실행 경로
(`grid_runner.py`/`smoke_test.py`)는 항상 `cl_client.py`를 거쳐 yaml
값을 명시적으로 넘기므로 지금까지의 모든 결과에는 영향이 없었지만,
yaml 없이 이 클래스를 직접 생성하는 코드(신규 유닛테스트 등)가 생기면
경고 없이 문서화된 값과 다른 아키텍처로 조용히 실행됐을 것 — 기본값을
64/16으로 맞췄다.

**전역 RNG 오염(2026-09-14 재검토로 발견, 수정 — 이 벤치마크의 핵심
전제를 위협하는 문제)**: 이 테스트베드는 "슬롯 값만 바꿔 통제된 비교를
한다"는 게 논문의 근거인데, `torch.manual_seed(seed)`는 콤보당 1회만
(`grid_runner.py`) 호출되고 라운드마다 재시드하지 않는다. 그런데
`dd=cade`가 선택되면 이 클래스의 생성자가 사설 `ContrastiveAutoEncoder`
가중치를 초기화하고(전역 RNG 소비), `fit_with_category()`가 매 라운드
`build_paired_batches`의 `torch.randperm` 등으로 추가로 전역 RNG를
소비한다 — 그 결과 `dd=cade`가 켜져 있는지 여부에 따라 **그 뒤에 실행되는
메인 공유 모델의 셔플·표본선택 등이 보는 난수 시퀀스 자체가 달라진다**.
즉 `dd=cade` on/off를 비교할 때, 차이의 일부가 CADE 메커니즘 자체가
아니라 "우연히 전역 RNG를 얼마나 소비했는가"에서 올 수 있다 — 같은
문제가 SSF 마스크 최적화(`ssf_masks.py`)·GPM/SPIDER의 `randperm`
(`spider_gpm/`)에도 있어 국지적 버그가 아니라 설계 패턴 자체의 문제다.

`torch.random.fork_rng()`(진입 시 RNG 상태 저장, 블록을 빠져나오면
복원)로 이 클래스의 모든 사설 랜덤성 소비를 격리한다 — 블록 안에서
`seed`로부터 파생된 고정 시드를 걸어 이 컴포넌트 자신의 결과는 여전히
재현 가능하게 하되, 바깥의 전역 RNG 스트림은 이 컴포넌트가 있든 없든
동일하게 유지된다. `testbed/common/rng_utils.py`의 `derived_seed(seed,
tag, counter)`로 파생 시드를 만든다 — 생성자는 `tag="cade_encoder_init"`,
`fit_with_category()`는 호출마다 `tag="cade_fit", counter=self._fit_round`
(라운드마다 다른 파생 시드 — 매 라운드 다른 데이터에 대해 완전히 같은
초기 난수를 반복하지 않도록). **2026-09-14 재검토로 정수 오프셋 방식(예:
`seed+90100+round`)에서 해시 기반으로 교체** — GPM의 자기학습 카운터처럼
미니배치마다 증가해 수천까지 가는 카운터가 있으면 100 단위로 떼어둔
정수 오프셋 구간을 실제로 침범할 수 있음을 발견해, 모든 소비처를
`rng_utils.derived_seed`로 통일했다(각 파일 docstring 참고).
"""

from typing import Dict, Optional, Sequence

import torch

from testbed.base.drift_detector import BaseDriftDetector
from testbed.common.rng_utils import derived_seed
from testbed.components.cade.contrastive_ae import (
    ContrastiveAutoEncoder, build_paired_batches, train_step)


class CADEDriftDetector(BaseDriftDetector):
    uses_shared_representation = False
    # 2026-09-12 추가(docs/design_decisions.md 5절) — CADE 원 논문
    # (cade/utils.py argparse 전체 확인 완료)에는 "라벨 예산" 개념이 전혀 없다. CADE는 학습셋이 이미
    # 충분히 라벨링되어 있다고 전제하는 논문이고, 이 테스트베드의 10% 라벨
    # 예산은 SSF에서 빌려온 제약을 그리드 전체에 적용한 것일 뿐 CADE
    # 자신에게는 근거가 없다. `selected_data`(라벨 예산 통과분)가 아니라
    # `new_data`(그 라운드 전체, 라벨 포함 — 원래 있던 라벨을 안 숨기는 것뿐)로
    # `fit_with_category()`를 호출하도록 `pipeline/cl_client.py` Step 3이
    # 이 플래그를 확인한다. `SPIDERMemoryManager.consumes_full_round_data`와
    # 동일한 관례(클래스 속성 + getattr(.., False) 호출부 체크, base 클래스
    # 선언 없음).
    consumes_full_round_data = True

    def __init__(self, input_dim: int, encoder_hidden_dim: int = 64, encoder_latent_dim: int = 16,
                 t_mad: float = 3.5, contrastive_margin: float = 10.0,
                 contrastive_lambda: float = 0.1, encoder_epochs: int = 5,
                 encoder_lr: float = 1e-4, batch_size: int = 128,
                 similar_ratio: float = 0.25, max_category_ref: int = 500,
                 target_replay_fraction: float = 0.15, seed: int = 42):
        self.t_mad = t_mad
        self.margin = contrastive_margin
        self.lam = contrastive_lambda
        self.epochs = encoder_epochs
        self.batch_size = batch_size
        self.similar_ratio = similar_ratio
        self.max_category_ref = max_category_ref
        # 전역 RNG 오염 격리(클래스 docstring 참고) — 이 컴포넌트 전용
        # 파생 시드, fit_with_category() 호출마다 라운드 카운터로 다시 파생.
        self._seed = seed
        self._fit_round = 0
        # 2026-09-12 추가(docs/design_decisions.md 5절) — consumes_full_round_data로
        # new_data(전체 라운드)가 anchor 풀에 들어오면서, 과거 category
        # replay(max_category_ref로 캡, 항상 작음)가 풀 크기 비례 anchor
        # 추출(_replay_known_categories/build_paired_batches)에서 압도적으로
        # 희석되는 회귀를 실측 확인(아래 _replay_known_categories docstring
        # 참고). 이번 라운드 data는 자르지 않고, 대신 replay 쪽을 오버샘플링해
        # 합쳐진 풀에서 최소 이 비율은 차지하도록 보정한다.
        self.target_replay_fraction = target_replay_fraction
        self._device = torch.device("cpu")
        with torch.random.fork_rng():
            torch.manual_seed(derived_seed(seed, "cade_encoder_init"))
            self._encoder = ContrastiveAutoEncoder(input_dim, encoder_hidden_dim, encoder_latent_dim)
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
        # 2026-09-12(docs/design_decisions.md 5절) — 이번 라운드 data는 그대로 전부 쓴다
        # (CADE의 "완전 라벨링 전제"를 대조학습 anchor 풀에도 온전히 반영 —
        # 아래 _replay_known_categories가 replay 쪽을 오버샘플링해 균형을
        # 맞추므로 여기서 따로 자를 필요가 없다).
        # 전역 RNG 오염 격리(클래스 docstring 참고) — _replay_known_categories()의
        # 복원추출 오버샘플링과 build_paired_batches()의 anchor
        # torch.randperm이 둘 다 전역 RNG를 쓴다. 라운드마다(self._fit_round)
        # 다른 파생 시드를 써서, 이 컴포넌트가 있든 없든 바깥의 메인 모델
        # 학습 루프가 보는 RNG 시퀀스가 달라지지 않게 한다.
        with torch.random.fork_rng():
            torch.manual_seed(derived_seed(self._seed, "cade_fit", self._fit_round))
            train_data, train_group = self._replay_known_categories(data, group)
            for _ in range(self.epochs):
                batch_count, paired_data, paired_group = build_paired_batches(
                    train_data, train_group, self.batch_size, self.similar_ratio)
                for b in range(batch_count):
                    train_step(self._encoder, self._optimizer, paired_data[b], paired_group[b],
                               self.margin, self.lam)
        self._fit_round += 1

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
        리플레이 계약과 별개로 이 사설 encoder 전용으로 구현.

        **2026-09-12 오버샘플링 추가**: `consumes_full_round_data`
        도입 후 `data`(이번 라운드)가 라벨 예산 대비 최대 10배 커지는데,
        `_category_refs`(과거 category, `max_category_ref`로 캡)는 그대로라
        결합 풀에서 replay 비중이 raw 크기 비율만큼 그대로 희석된다.
        `build_paired_batches`의 anchor 추출이 `torch.randperm`으로 풀
        크기에 비례하므로, 희석되면 과거 category가 anchor로 거의 뽑히지
        않아 리허설 효과가 사실상 사라진다 — NSL-KDD smoke A/B로 실측
        확인(`dd=cade+as=cade_mad`: f1 0.2688→0.1345, bwt
        -0.0738→-0.4402). replay를 `target_replay_fraction`(기본 0.15,
        {0.08,0.15,0.20,0.30,0.50} 스윕 후 채택 — 아래 참고)만큼
        복원추출로 오버샘플링해 결합 풀에서 최소 그 비율은 차지하게
        보정한다 — `data`(이번 라운드)는 자르지 않는다. (카테고리당 500건
        캡을 이번 라운드 쪽에도 거는 대안을 먼저 시도했으나, 라운드 데이터가
        캡보다 훨씬 큰 경우 오히려 라벨 예산(10%)보다도 적은 양만 학습해
        이번 라운드 자체 학습이 무너짐 — round1/2 자체 recall이 거의 0으로
        붕괴하는 걸 실측 확인하고 되돌렸다. 현재의 오버샘플링 방식은 이번
        라운드 쪽을 전혀 줄이지 않으므로 그 실패 모드가 구조적으로 재현되지
        않는다.) 이 접근의 A/B 실측 결과는 클래스 docstring 참고."""
        extra_data, extra_group = [], []
        for code, ref in self._category_refs.items():
            extra_data.append(ref)
            extra_group.append(torch.full((len(ref),), code, dtype=torch.long, device=data.device))
        if not extra_data:
            return data, group
        replay_data = torch.cat(extra_data, dim=0)
        replay_group = torch.cat(extra_group, dim=0)
        frac = self.target_replay_fraction
        target_replay_n = int(len(data) * frac / (1 - frac))
        if len(replay_data) < target_replay_n:
            reps = torch.randint(len(replay_data), (target_replay_n,), device=replay_data.device)
            replay_data = replay_data[reps]
            replay_group = replay_group[reps]
        combined_data = torch.cat([data, replay_data], dim=0)
        combined_group = torch.cat([group, replay_group], dim=0)
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
        `docs/metric_justification.md`).

        **`combined[-self.max_category_ref:]` 꼬리 슬라이싱이 편향을 만들지
        않는 이유(2026-09-15 재확인 — `cndids_memory_manager.py`의 "FIFO가
        직전 라운드만 기억" 문제와 겉보기엔 같은 패턴이라 재검토했다)**:
        공격 category는 class-incremental 분할 설계상 정확히 한 라운드
        (`data/dataset_loader.py`의 "라운드 하나 = 새 공격 유형 하나")에만
        등장하므로, 이 메서드가 그 category로 호출되는 건 평생 단 한 번뿐
        이다 — `combined`가 그 category의 새 데이터로만 채워지고, "이전
        라운드 참조가 밀려난다"는 문제 자체가 성립하지 않는다. 문제가 될 수
        있는 건 매 라운드 반복 호출되는 정상(normal) category뿐인데, 정상
        행은 `_class_incremental_split`이 라운드 배정 **이전에** 이미 전역
        셔플해 고르게 나눈 것이라(카테고리/피처와 무관한 무작위 분배, 라운드
        간 분포 이동이 애초에 설계상 없음) "최근 라운드 것만 남는다"고 해도
        그 자체가 정상 트래픽 전체 분포의 편향된 부분집합이 되는 게 아니라
        — 단지 그 분포를 대표하는 **표본 크기가 줄어드는 것**(분산 증가)일
        뿐이다. `combined`가 이미 (그 라운드 자신의 최종 셔플 단계를 거친)
        무작위 순서로 들어오므로 "꼬리"를 자르는 것과 무작위 서브샘플을
        뽑는 것이 분포상 동일하다는 점도 같이 성립한다. 이건 CND-IDS의
        FIFO 버퍼(같은 category가 매 라운드 재등장하고, 라운드 자체가 서로
        다른 정보를 담아 소실이 실제 정보 손실인 경우)와 근본적으로 다른
        상황이라 같은 수정(무작위 교체)을 적용할 필요가 없다고 판단한다 —
        다만 "필요 없다"는 이 판단 자체가 실제 그리드 스케일로 A/B 검증된
        적은 없다(정성적 논증일 뿐)."""
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
