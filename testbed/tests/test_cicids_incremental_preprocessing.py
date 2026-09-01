"""CICIDS2018 전용 라운드-누적 전처리(2026-08-26 도입) 회귀 테스트.

CICIDS2018은 로컬에서 전체 파이프라인을 끝까지 실행할 수 없으므로(원본
CSV 병합 시 메모리 부족 — docs/metric_justification.md 참고), 실제
데이터 없이도 검증 가능한 핵심 불변조건만 합성 데이터로 확인한다:
  1. `_class_incremental_split(extra_arrays=...)`이 port/protocol을
     X/y/category와 정확히 같은 행 순서로 슬라이싱하는가.
  2. `_IncrementalPortBucketer`가 항상 {0,1,2} 안의 값만 내는가.
  3. `_one_hot(categories=...)`로 고정한 원-핫 폭이 experience마다 어떤
     부분집합이 나타나든 항상 동일한가(공유 모델의 input_dim이 라운드마다
     안 바뀌어야 한다 — 이게 깨지면 두 번째 라운드부터 바로 shape 에러로
     크래시한다).
"""

import numpy as np

from testbed.data.dataset_loader import (
    _class_incremental_split, _IncrementalPortBucketer, _one_hot)


def test_extra_arrays_and_incremental_port_bucketing_keep_input_dim_stable():
    rng = np.random.RandomState(0)
    n = 5000
    n_features = 10
    X = rng.randn(n, n_features)
    y = (rng.rand(n) < 0.3).astype(int)
    categories_pool = np.array(["A", "B", "C", "D"])
    category = np.where(y == 0, "normal", rng.choice(categories_pool, size=n))
    port = rng.choice([80, 443, 22, 12345, 54321, 9999], size=n,
                       p=[0.4, 0.3, 0.2, 0.05, 0.03, 0.02])
    protocol = rng.choice([6, 17, 1], size=n, p=[0.7, 0.25, 0.05])

    n_experiences = 5
    exp_chunks, _ = _class_incremental_split(
        X, y, category, n_experiences, seed=42, extra_arrays=[port, protocol])

    assert len(exp_chunks) == n_experiences

    bucketer = _IncrementalPortBucketer()
    protocol_categories = np.unique(protocol)
    input_dims = set()
    total_rows = 0
    for X_exp, y_exp, cat_exp, port_exp, protocol_exp in exp_chunks:
        assert len(X_exp) == len(y_exp) == len(cat_exp) == len(port_exp) == len(protocol_exp)
        total_rows += len(X_exp)

        bucket = bucketer.bucket(port_exp)
        assert set(np.unique(bucket).tolist()) <= {0, 1, 2}

        port_onehot = _one_hot(bucket, n_categories_cap=3, categories=np.array([0, 1, 2]))
        protocol_onehot = _one_hot(protocol_exp, categories=protocol_categories)
        assert port_onehot.shape == (len(X_exp), 3)
        assert protocol_onehot.shape == (len(X_exp), len(protocol_categories))

        combined = np.concatenate([port_onehot, protocol_onehot, X_exp], axis=1)
        input_dims.add(combined.shape[1])

    assert total_rows == n
    assert len(input_dims) == 1, f"input_dim이 experience마다 달라짐: {input_dims}"


def test_incremental_port_bucketer_never_sees_future_data():
    """라운드 i의 버킷 경계는 0..i까지 누적된 빈도만 반영해야 한다 —
    아직 등장하지 않은 미래 라운드의 포트 분포에 영향받으면 안 된다."""
    bucketer = _IncrementalPortBucketer()

    round0_ports = np.array([80] * 100 + [443] * 100)
    bucket0 = bucketer.bucket(round0_ports)
    # 이 시점엔 80/443만 알려져 있고 둘 다 빈도가 같다.
    assert set(bucket0.tolist()) <= {0, 1, 2}

    # 미래에 압도적으로 등장할 포트를 미리 "관측"시키지 않고, 라운드1에서
    # 등장했을 때 라운드0의 버킷 결과가 뒤늦게 안 바뀌는지 확인한다
    # (누적 상태는 다음 호출에만 영향을 줘야 하며, 이전 반환값을 소급
    # 수정하지 않는다 — 함수가 순수하게 새 배열을 리턴할 뿐 과거 반환값을
    # 변형하지 않는다는 걸 확인).
    bucket0_snapshot = bucket0.copy()
    round1_ports = np.array([9999] * 10000)
    bucketer.bucket(round1_ports)
    assert np.array_equal(bucket0, bucket0_snapshot)
