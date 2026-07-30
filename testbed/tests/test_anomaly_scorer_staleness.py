"""PRD 12.6절/6절 — CLClient.run_experience()가 매 experience 종료 시
anomaly_scorer.refit_on_update()를 실제로 호출하는지 검증하는 회귀 테스트.
검증 대상은 개별 컴포넌트가 아니라 CLClient다(기본 refit_on_update() 구현은
이미 요구사항을 만족하므로)."""

from unittest.mock import MagicMock

import torch

from testbed.base import FCLAutoEncoder
from testbed.pipeline import CLClient


def _make_client(combo):
    global_hparams = {
        "optimizer": "adam", "learning_rate": 1e-3, "batch_size": 16,
        "epochs_per_experience": 1, "hidden_dim": 16, "latent_dim": 4,
        "_input_dim": 8,
    }
    model = FCLAutoEncoder(input_dim=8, hidden_dim=16, latent_dim=4)
    client = CLClient(model, combo, global_hparams, component_hparams={})
    return client


def _run_n_experiences(client, n=3, batch_n=30, dim=8):
    all_test_splits = []
    exps = []
    for _ in range(n):
        X = torch.randn(batch_n, dim)
        y = (torch.rand(batch_n) < 0.3).long()
        train_n = int(batch_n * 0.8)
        exps.append((X[:train_n], y[:train_n]))
        all_test_splits.append((X[train_n:], y[train_n:]))

    labeling_budget = {"mode": "fixed_ratio", "value": 0.5}
    outs = []
    for i, (train_X, train_y) in enumerate(exps):
        out = client.run_experience(i, train_X, train_y, all_test_splits, labeling_budget)
        outs.append(out)
    return outs


def test_refit_on_update_called_every_experience_track_a():
    combo = {"track": "A", "drift_detector": "none", "sample_selector": "random",
             "memory_manager": "spider", "anti_forgetting": "none",
             "anomaly_scorer": "cade_mad"}
    client = _make_client(combo)
    client.anomaly_scorer.refit_on_update = MagicMock(
        wraps=client.anomaly_scorer.refit_on_update)

    n = 3
    _run_n_experiences(client, n=n)

    assert client.anomaly_scorer.refit_on_update.call_count == n


def test_refit_on_update_called_every_experience_track_b():
    combo = {"track": "B", "drift_detector": "none", "sample_selector": "random",
             "memory_manager": "cndids", "anti_forgetting": "cndids",
             "anomaly_scorer": "pca"}
    client = _make_client(combo)
    client.anomaly_scorer.refit_on_update = MagicMock(
        wraps=client.anomaly_scorer.refit_on_update)

    n = 3
    _run_n_experiences(client, n=n)

    assert client.anomaly_scorer.refit_on_update.call_count == n


def test_refit_receives_current_encoder_output_not_stale_cache():
    """refit_on_update에 넘어가는 인코딩 값이 매 라운드 실제로 달라지는지
    (즉, 캐시된 옛 encoder 출력이 재사용되지 않는지) 확인한다."""
    combo = {"track": "A", "drift_detector": "none", "sample_selector": "random",
             "memory_manager": "spider", "anti_forgetting": "none",
             "anomaly_scorer": "cade_mad"}
    client = _make_client(combo)

    seen_inputs = []
    original = client.anomaly_scorer.refit_on_update

    def spy(normal_data):
        seen_inputs.append(normal_data.clone())
        return original(normal_data)

    client.anomaly_scorer.refit_on_update = spy
    _run_n_experiences(client, n=3)

    assert len(seen_inputs) == 3
    # 2026-07-30 재설계: normal_subset이 매 라운드 새로 선택된 데이터에서
    # 뽑히므로 이제 크기 자체가 라운드마다 다를 수 있다(이전엔 고정 크기
    # 참조 표본을 재인코딩만 했음) — 크기가 다르면 그 자체로 "캐시된 옛 값을
    # 재사용하지 않는다"는 증거이므로 shape부터 비교한다. 크기가 우연히
    # 같더라도(모델이 매 라운드 갱신되므로) 인코딩 결과 값 자체는 달라야
    # 한다.
    shapes = [tuple(t.shape) for t in seen_inputs]
    all_identical = len(set(shapes)) == 1 and all(
        torch.allclose(seen_inputs[0], seen_inputs[i]) for i in range(1, len(seen_inputs)))
    assert not all_identical
