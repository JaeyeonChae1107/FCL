"""Unit tests for all pluggable components."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pytest
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Base imports ───────────────────────────────────────────────────────────
from testbed.base import (BaseDriftDetector, BaseSampleSelector,
                           BaseMemoryManager, BaseAntiForgetting,
                           BaseAnomalyScorer)

# ── Model ──────────────────────────────────────────────────────────────────
from testbed.pipeline.models import SSFModel

# ── Drift detectors ────────────────────────────────────────────────────────
from testbed.components.ssf import SSFDriftDetector
from testbed.components.cade import CADEDriftDetector
from testbed.components.cndids import DDMDriftDetector
from testbed.pipeline.component_registry import NoDriftDetector

# ── Sample selectors ───────────────────────────────────────────────────────
from testbed.components.ssf import SSFSampleSelector
from testbed.pipeline.component_registry import AllSampleSelector, RandomSelector

# ── Memory managers ────────────────────────────────────────────────────────
from testbed.components.ssf import SSFMemoryManager
from testbed.components.cndids import CNDIDSMemoryManager
from testbed.pipeline.component_registry import NoMemoryManager, FIFOMemoryManager

# ── Anti-forgetting ────────────────────────────────────────────────────────
from testbed.components.ssf import SSFAntiForgetting
from testbed.components.cndids import CNDIDSAntiForgetting
from testbed.components.gpm import GPMAntiForgetting
from testbed.pipeline.component_registry import ReplayOnlyLoss

# ── Anomaly scorers ────────────────────────────────────────────────────────
from testbed.components.cndids import PCAAnomalyScorer
from testbed.components.cade import CADEAnomalyScorer


# ── Fixtures ───────────────────────────────────────────────────────────────
DIM = 16
N = 64
BUDGET = 10


@pytest.fixture
def data():
    return torch.randn(N, DIM)


@pytest.fixture
def labels():
    return torch.randint(0, 2, (N,))


@pytest.fixture
def simple_model():
    return SSFModel(input_dim=DIM)


# ── Drift Detector tests ───────────────────────────────────────────────────
DRIFT_DETECTORS = [
    NoDriftDetector,
    SSFDriftDetector,
    DDMDriftDetector,
]


@pytest.mark.parametrize("cls", DRIFT_DETECTORS)
def test_drift_detector_detect_returns_bool(cls, data):
    det = cls()
    result = det.detect(data, None)
    assert isinstance(result, bool)


@pytest.mark.parametrize("cls", DRIFT_DETECTORS)
def test_drift_detector_score_returns_float(cls, data):
    det = cls()
    score = det.get_drift_score(data, None)
    assert isinstance(score, float)


def test_cade_drift_detector_after_fit(data):
    det = CADEDriftDetector(threshold=3.5)
    y = torch.randint(0, 2, (N,))
    det.fit(data, y)
    result = det.detect(data, None)
    assert isinstance(result, bool)
    score = det.get_drift_score(data, None)
    assert isinstance(score, float) and score >= 0.0


def test_no_drift_detector_always_false(data):
    det = NoDriftDetector()
    assert det.detect(data, None) is False
    assert det.get_drift_score(data, None) == 0.0


# ── Sample Selector tests ──────────────────────────────────────────────────
SAMPLE_SELECTORS = [AllSampleSelector, RandomSelector, SSFSampleSelector]


@pytest.mark.parametrize("cls", SAMPLE_SELECTORS)
def test_selector_returns_list(cls, data, labels):
    sel = cls()
    idx = sel.select(data, labels, BUDGET)
    assert isinstance(idx, list)


@pytest.mark.parametrize("cls", SAMPLE_SELECTORS)
def test_selector_respects_budget(cls, data, labels):
    sel = cls()
    idx = sel.select(data, labels, BUDGET)
    assert len(idx) <= BUDGET


@pytest.mark.parametrize("cls", SAMPLE_SELECTORS)
def test_selector_valid_indices(cls, data, labels):
    sel = cls()
    idx = sel.select(data, labels, BUDGET)
    assert all(0 <= i < N for i in idx)


# ── Memory Manager tests ───────────────────────────────────────────────────
MEMORY_MANAGERS = [
    (NoMemoryManager, {}),
    (FIFOMemoryManager, {"max_size": 50}),
    (SSFMemoryManager, {"max_size": 50, "num_labeled_sample": 10}),
    (CNDIDSMemoryManager, {"capacity": 50}),
]


@pytest.mark.parametrize("cls,kwargs", MEMORY_MANAGERS)
def test_memory_size_zero_before_update(cls, kwargs):
    mgr = cls(**kwargs)
    assert mgr.size() == 0


@pytest.mark.parametrize("cls,kwargs", MEMORY_MANAGERS)
def test_memory_update_increases_size(cls, kwargs, data, labels):
    mgr = cls(**kwargs)
    mgr.update(data[:10], labels[:10], drift_detected=False)
    if cls is NoMemoryManager:
        assert mgr.size() == 0
    else:
        assert mgr.size() > 0


@pytest.mark.parametrize("cls,kwargs", MEMORY_MANAGERS)
def test_memory_replay_batch_shape(cls, kwargs, data, labels):
    mgr = cls(**kwargs)
    mgr.update(data[:20], labels[:20], drift_detected=False)
    r_data, r_labels = mgr.get_replay_batch(8)
    if cls is NoMemoryManager:
        assert r_data is None
    else:
        assert r_data is not None
        assert r_data.shape[0] <= 8
        assert r_data.shape[1] == DIM


# ── Anti-Forgetting tests ──────────────────────────────────────────────────
ANTI_FORGETTING = [
    (ReplayOnlyLoss, {}),
    (SSFAntiForgetting, {"lwf_lambda": 0.5}),
    (CNDIDSAntiForgetting, {}),
    (GPMAntiForgetting, {"threshold": 0.97}),
]


@pytest.mark.parametrize("cls,kwargs", ANTI_FORGETTING)
def test_af_compute_loss_returns_tensor(cls, kwargs, data, labels, simple_model):
    af = cls(**kwargs)
    new_batch = (data[:16], labels[:16])
    loss = af.compute_loss(simple_model, new_batch, None)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar


@pytest.mark.parametrize("cls,kwargs", ANTI_FORGETTING)
def test_af_loss_requires_grad(cls, kwargs, data, labels, simple_model):
    af = cls(**kwargs)
    new_batch = (data[:16], labels[:16])
    loss = af.compute_loss(simple_model, new_batch, None)
    assert loss.requires_grad or loss.item() == 0.0


@pytest.mark.parametrize("cls,kwargs", ANTI_FORGETTING)
def test_af_on_task_end_no_exception(cls, kwargs, simple_model):
    af = cls(**kwargs)
    af.on_task_end(simple_model)


# ── Anomaly Scorer tests ───────────────────────────────────────────────────
ANOMALY_SCORERS = [
    (PCAAnomalyScorer, {}),
    (CADEAnomalyScorer, {}),
]


@pytest.mark.parametrize("cls,kwargs", ANOMALY_SCORERS)
def test_scorer_fit_then_score_shape(cls, kwargs, data):
    scorer = cls(**kwargs)
    scorer.fit(data)
    scores = scorer.score(data)
    assert scores.shape == (N,), f"{cls.__name__}: scores shape {scores.shape}"


@pytest.mark.parametrize("cls,kwargs", ANOMALY_SCORERS)
def test_scorer_predict_binary(cls, kwargs, data):
    scorer = cls(**kwargs)
    scorer.fit(data)
    threshold = scorer.score(data).median().item()
    preds = scorer.predict(data, threshold)
    assert set(preds.tolist()).issubset({0, 1})
