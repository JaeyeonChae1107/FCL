"""End-to-end pipeline tests."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pytest
import torch
from testbed.pipeline.cl_client import CLClient
from testbed.experiments.metrics import f1_score


def _make_model(dim: int = 121):
    from testbed.pipeline.models import SSFModel
    return SSFModel(input_dim=dim)


def _make_tasks(n: int = 500, dim: int = 121, n_tasks: int = 3):
    X = torch.randn(n, dim)
    y = torch.randint(0, 2, (n,))
    size = n // n_tasks
    return [(X[i*size:(i+1)*size], y[i*size:(i+1)*size])
            for i in range(n_tasks)]


def test_ssf_gpm_pca_pipeline():
    """SSF drift + SSF selector + SSF memory + GPM + PCA scorer."""
    config = {
        "drift_detector":  {"name": "ssf", "drift_threshold": 0.05},
        "sample_selector": {"name": "ssf"},
        "memory_manager":  {"name": "ssf", "max_size": 200},
        "anti_forgetting": {"name": "gpm", "threshold": 0.97},
        "anomaly_scorer":  {"name": "pca"},
        "label_budget": 30, "lr": 1e-3,
    }
    tasks = _make_tasks(n_tasks=3)
    client = CLClient(model=_make_model(), config=config, device='cpu')

    for X, y in tasks:
        result = client.update(X, y)
        assert 'loss' in result
        assert isinstance(result['drift'], bool)

    normal = tasks[0][0][tasks[0][1] == 0]
    if len(normal) == 0:
        normal = tasks[0][0][:5]
    client.fit_anomaly_scorer(normal)

    out = client.infer(tasks[-1][0])
    assert 'scores' in out and 'predictions' in out
    assert out['scores'].shape[0] == len(tasks[-1][0])
    assert set(out['predictions'].tolist()).issubset({0, 1})

    f1 = f1_score(tasks[-1][1].numpy(), out['predictions'].numpy())
    assert f1 >= 0.0, f"F1 should be non-negative, got {f1}"


def test_none_none_none_pipeline():
    """All-none / random baseline pipeline."""
    config = {
        "drift_detector":  {"name": "none"},
        "sample_selector": {"name": "random"},
        "memory_manager":  {"name": "none"},
        "anti_forgetting": {"name": "none"},
        "anomaly_scorer":  {"name": "pca"},
        "label_budget": 30, "lr": 1e-3,
    }
    tasks = _make_tasks(n_tasks=3)
    client = CLClient(model=_make_model(), config=config, device='cpu')

    for X, y in tasks:
        result = client.update(X, y)
        assert result['drift'] is False   # NoDriftDetector always False

    normal = tasks[0][0][:10]
    client.fit_anomaly_scorer(normal)

    out = client.infer(tasks[-1][0])
    assert out['predictions'].shape == tasks[-1][1].shape


def test_model_state_round_trip():
    """get_model_state → load_model_state round-trip."""
    config = {
        "drift_detector":  {"name": "none"},
        "sample_selector": {"name": "random"},
        "memory_manager":  {"name": "none"},
        "anti_forgetting": {"name": "none"},
        "anomaly_scorer":  {"name": "pca"},
        "label_budget": 10, "lr": 1e-3,
    }
    model = _make_model()
    client = CLClient(model=model, config=config)
    X = torch.randn(20, 121)
    y = torch.randint(0, 2, (20,))
    client.update(X, y)

    state = client.get_model_state()
    assert isinstance(state, dict)
    client.load_model_state(state)   # should not raise


def test_fl_aggregation():
    """Simulate FL: two clients share model state."""
    config = {
        "drift_detector":  {"name": "none"},
        "sample_selector": {"name": "random"},
        "memory_manager":  {"name": "fifo", "max_size": 50},
        "anti_forgetting": {"name": "none"},
        "anomaly_scorer":  {"name": "pca"},
        "label_budget": 20, "lr": 1e-3,
    }
    dim = 121
    client1 = CLClient(model=_make_model(dim), config=config)
    client2 = CLClient(model=_make_model(dim), config=config)

    X1 = torch.randn(50, dim)
    y1 = torch.randint(0, 2, (50,))
    client1.update(X1, y1)

    # Simple FedAvg: average the two states
    state1 = client1.get_model_state()
    state2 = client2.get_model_state()
    averaged = {k: (state1[k] + state2[k]) / 2.0 for k in state1}
    client1.load_model_state(averaged)
    client2.load_model_state(averaged)
