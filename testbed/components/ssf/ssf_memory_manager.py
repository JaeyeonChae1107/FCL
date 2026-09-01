"""SSF MemoryManager — Strategic Forgetting (PRD 4절/12.4절).

SSF 원 논문 근거: utils.py의 select_and_update_representative_samples()가
비대표(M_c_bin==0) 샘플을 우선 제거하고, 부족하면 대표 샘플 중 가장 낮은
M_c 점수부터 추가 제거한다(utils.py:192-257,259-388) — FIFO가 아니라
"대표성 낮은 순 우선 제거"가 핵심 성질이다. drift 시에는 더 공격적으로
교체한다(utils.py:259-388의 별도 drift 분기).

이 테스트베드의 목적은 원본 코드를 그대로 재현하는 게 아니라, "drift_detector가
만드는 신호(drift_detected)를 실제로 소비하는 컴포넌트가 있어야 drift_detector
슬롯(none/ssf/cade)이 조합 비교에서 의미 있는 축이 된다"는 것이다(사용자
지시 — 이 소비 관계가 없는 (sample_selector, memory_manager) 조합은
common/compatibility.py의 TRACK_A_DD_ACTIVE_SS_MM에서 애초에 제외했다).
원문의 마스크 점수(M_c)를 그대로 전달받지 않으므로, 매 update 시점에
"자기 클래스 centroid까지의 거리가 가까울수록 대표적"이라는 대표성 개념으로
점수를 재계산해 하위 점수부터 제거한다.

**2026-08-26 발견·수정 — 클래스 간 점수 스케일 차이로 소수 클래스가
버퍼에서 통째로 밀려나는 문제**: 4개 논문 컴포넌트 전수 재감사에서,
`_representativeness()`의 점수(자기 클래스 centroid까지의 거리)는 그
클래스 자신의 특징 분산에 스케일이 좌우되는데, `update()`가 이 점수를
**클래스 구분 없이 전역** `topk`로 랭킹해 버퍼를 채우고 있었다는 걸
재확인했다. 특징 분산이 더 큰 클래스(예: 서로 이질적인 공격 유형이 섞인
"공격" 클래스)는 모든 점수가 분산이 작은 클래스(예: 정상 트래픽)보다
구조적으로 더 낮게 나와, 아무리 그 클래스 내부에서 "대표적인" 표본이라도
전역 랭킹에서 밀린다. 실측(NSL-KDD 5라운드 전체, `dd=ssf/ss=ssf/mm=ssf/
af=lwf_ssf/as=cade_mad`)으로 확인: 라운드3(U2R)에서 선택된 attack 5건이
그 즉시 같은 `update()` 호출 안에서 버퍼 attack 슬롯(369개, 전부
DoS/Probe/R2L)에 밀려 **0/5 생존** — 한 번도 리플레이되지 못했다. U2R
라운드 자체의 학습 성능(diag-F1)이 0.048에 그친 것과 직접 연결된다.

클래스별로 버퍼 슬롯을 그 클래스가 combined 데이터에서 차지하는 비율만큼
(최소 1개) 미리 배정하고, **그 쿼터 안에서만** `topk`로 대표성을 매기도록
바꿨다(`_quota_topk_indices`) — 클래스 간 점수 스케일 차이와 무관하게,
어떤 클래스든 "자기 몫" 안에서는 항상 대표적인 표본이 생존한다. drift
시 과거 표본을 추려내는 단계(`old_keep`)에도 동일하게 적용한다.

**category 쿼터로 확장했다가 되돌림 — 재검증 후에도 기각 유지**: 이진
쿼터 수정 후에도 U2R 라운드 자체 학습 성능(diag-F1)이 0.048→0.048로
거의 그대로여서, `train_category`(다중클래스)로 버퍼 쿼터를 나누는
`update_with_category()`를 추가해봤다. 처음 A/B(NSL-KDD, `dd=ssf/ss=ssf/
mm=ssf/af=lwf_ssf/as=cade_mad`)는 U2R 0.048→0.074 개선, 전체 f1 0.7291→
0.6216 악화로 기각했는데, 이 비교가 그 사이 바뀐 MinMaxScaler 시간 유출
수정 이전 낡은 숫자와 비교된 것이었음이 재감사에서 드러났다(자세한 경위는
`ssf_sample_selector.py` 모듈 docstring "2026-08-26 재검증" 절 참고).
현재 코드로 4개 변형(이진만/선택기만 category/버퍼만 category/둘 다)을
다시 재본 결과, **선택기의 category 쿼터는 실제로 이진보다 낫다는 게
확인돼 채택**했지만, **버퍼의 category 쿼터는 재검증 후에도 여전히
이진보다 나빴다**(버퍼만: f1=0.5107/roc_auc=0.5101, 둘 다: f1=0.5879/
roc_auc=0.5276 — 이진만 f1=0.6565/roc_auc=0.7150, 선택기만 f1=0.7040/
roc_auc=0.6451보다 둘 다 못하다). 버퍼 쪽만 기각을 유지한다 — SSF의
"대표성" 대체 휴리스틱으로 버퍼 슬롯을 category 단위로 쪼개면(선택기의
"예산 배정"과 달리 "이미 담긴 표본들 사이에서 누구를 내보낼지"라는 문제라)
여전히 전체 대표성을 해치는 것으로 보인다.

**2026-08-12 정정 — drift 반응 방향**: 이전에는 drift_detected=True인
라운드에 유지 개수 자체를 `max_size*drift_retention_ratio`로 줄였다(버퍼
총량 축소). 그런데 SSF 원문(`utils.py:259-388`,
`select_and_update_representative_samples_when_drift`)을 다시 보면 drift
시에도 버퍼는 목표 크기를 유지한 채(비대표 표본을 더 적극적으로 제거하고
그만큼 새 대표 표본을 더 채워) "총량은 유지하되 회전율을 높이는" 방식이다
— "축소"가 아니라 "더 공격적인 교체"였다. 이제는 **기존 버퍼 쪽만** drift
시 대표성 상위 `max_size*drift_retention_ratio`개로 먼저 추려내고(비대표
과거 표본을 선제적으로 더 많이 버림), 거기에 이번 라운드 신규 표본
전체를 합친 뒤, 평시와 동일하게 대표성 상위 `max_size`개를 최종 유지한다
— 버퍼 총량은 drift 여부와 무관하게 항상 max_size로 수렴하되(초기 데이터
부족 라운드 제외), drift 라운드는 과거 표본이 새 표본에 자리를 더 많이
내주게 된다.

**2026-09-01 정정 — 위 "항상 max_size로 수렴" 서술은 drift 라운드에서
엄밀하지 않다(구조 감사에서 발견)**: `update()`의 마지막 캡 단계는
`len(combined_data) > self.max_size`일 때만 대표성 top-k를 적용한다.
drift 라운드에서 `old_keep(=max_size*drift_retention_ratio)`개로 먼저
줄인 뒤 이번 라운드 신규 표본(`selected_data`, 보통 라벨 예산의 극히
일부)을 더해도 그 합이 `max_size`를 넘지 못하면 이 캡이 아예 발동하지
않아, 버퍼가 `max_size`보다 작은 채로 그 라운드를 마친다 — drift 라운드에
한해 "총량 유지"가 깨질 수 있다(평시 라운드는 기존 버퍼 자체가 이미
`max_size`라 항상 캡이 발동해 문제없다). 위 A/B 실측 수치는 이 코드
그대로에서 측정된 것이라 결론 자체는 유효하지만, 이 코너케이스는 미반영
상태로 남아있다.
"""

from typing import Optional, Tuple

import torch

from testbed.base.memory_manager import BaseMemoryManager


class SSFMemoryManager(BaseMemoryManager):
    def __init__(self, max_size: int = 1000, drift_retention_ratio: float = 0.7):
        self.max_size = max_size
        # drift 감지 시 버퍼를 이 비율만큼만 유지해 더 공격적으로 교체한다.
        # PRD 6절 "테스트베드 기본값" 성격 — 특정 논문 수치가 아니라, drift_detector
        # 슬롯이 실제로 결과에 영향을 주도록 만드는 최소한의 원칙적 조정.
        self.drift_retention_ratio = drift_retention_ratio
        self._buf_data: Optional[torch.Tensor] = None
        self._buf_labels: Optional[torch.Tensor] = None

    def update(self, selected_data: torch.Tensor, selected_labels: torch.Tensor,
               drift_detected: bool = False) -> None:
        if self._buf_data is None:
            combined_data, combined_labels = selected_data.clone(), selected_labels.clone()
        else:
            old_data, old_labels = self._buf_data, self._buf_labels
            if drift_detected and len(old_data) > 0:
                # 과거 표본만 먼저 대표성 상위 일부로 추려내 새 표본에
                # 자리를 더 내준다 — 버퍼 "총량"은 아래에서 max_size로
                # 그대로 복원되므로 축소되지 않는다.
                old_keep = max(0, int(self.max_size * self.drift_retention_ratio))
                if old_keep < len(old_data):
                    if old_keep == 0:
                        old_data = old_data[:0]
                        old_labels = old_labels[:0]
                    else:
                        old_scores = self._representativeness(old_data, old_labels)
                        idx = self._quota_topk_indices(old_labels, old_scores, old_keep)
                        old_data, old_labels = old_data[idx], old_labels[idx]
            combined_data = torch.cat([old_data, selected_data], dim=0)
            combined_labels = torch.cat([old_labels, selected_labels], dim=0)

        if len(combined_data) <= self.max_size:
            self._buf_data, self._buf_labels = combined_data, combined_labels
            return

        scores = self._representativeness(combined_data, combined_labels)
        keep_idx = self._quota_topk_indices(combined_labels, scores, self.max_size)
        self._buf_data = combined_data[keep_idx]
        self._buf_labels = combined_labels[keep_idx]

    @staticmethod
    def _representativeness(data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        scores = torch.zeros(len(data), device=data.device)
        for c in labels.unique():
            mask = labels == c
            centroid = data[mask].mean(dim=0, keepdim=True)
            dist = torch.norm(data[mask] - centroid, dim=1)
            scores[mask] = -dist  # 가까울수록(비대표성 낮을수록) 높은 점수
        return scores

    @staticmethod
    def _quota_topk_indices(labels: torch.Tensor, scores: torch.Tensor, k: int) -> torch.Tensor:
        """전역 top-k 대신 클래스별 쿼터 안에서만 top-k를 뽑는다(위 모듈
        docstring "2026-08-26" 절 참고) — `_representativeness()`의 점수는
        클래스 자신의 특징 분산에 스케일이 좌우되므로, 클래스 구분 없이
        전역 랭킹하면 분산이 큰 클래스가 통째로 밀려날 수 있다. 각 클래스에
        combined 데이터에서 차지하는 비율만큼(최소 1개, 그 클래스 표본
        수를 넘지 않게) 슬롯을 배정하고, 반올림 오차는 표본 수가 많은
        클래스부터 순서대로 보정해 합이 정확히 k가 되게 한다."""
        n = len(labels)
        if k >= n:
            return torch.arange(n, device=labels.device)
        classes = labels.unique().tolist()
        idx_by_class = {c: (labels == c).nonzero(as_tuple=True)[0] for c in classes}
        counts = {c: len(idx_by_class[c]) for c in classes}
        total = sum(counts.values())

        quotas = {c: min(max(1, round(k * counts[c] / total)), counts[c]) for c in classes}
        diff = k - sum(quotas.values())
        order = sorted(classes, key=lambda c: counts[c], reverse=True)
        i = 0
        while diff != 0 and i < 10000:
            c = order[i % len(order)]
            if diff > 0 and quotas[c] < counts[c]:
                quotas[c] += 1
                diff -= 1
            elif diff < 0 and quotas[c] > 0:
                quotas[c] -= 1
                diff += 1
            i += 1

        keep = []
        for c in classes:
            if quotas[c] <= 0:
                continue
            class_idx = idx_by_class[c]
            top = torch.topk(scores[class_idx], quotas[c]).indices
            keep.append(class_idx[top])
        return torch.cat(keep) if keep else torch.empty(0, dtype=torch.long, device=labels.device)

    def get_replay_batch(self, batch_size: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self._buf_data is None:
            return None, None
        n = min(batch_size, len(self._buf_data))
        idx = torch.randperm(len(self._buf_data), device=self._buf_data.device)[:n]
        return self._buf_data[idx], self._buf_labels[idx]

    def get_buffer(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._buf_data, self._buf_labels

    def size(self) -> int:
        return 0 if self._buf_data is None else len(self._buf_data)
