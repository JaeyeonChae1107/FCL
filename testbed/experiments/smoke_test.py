"""Smoke test — validate 5 representative pipeline combinations.

Runs each config with dummy data (N=300, dim=121, n_tasks=3).
PASS = no exception raised + f1 >= 0.0.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import json
import torch
import numpy as np

from testbed.pipeline.cl_client import CLClient
from testbed.experiments.metrics import f1_score


def make_dummy_tasks(n: int = 300, dim: int = 121, n_tasks: int = 3):
    X = torch.randn(n, dim)
    y = torch.randint(0, 2, (n,))
    size = n // n_tasks
    return [(X[i*size:(i+1)*size], y[i*size:(i+1)*size])
            for i in range(n_tasks)]


SMOKE_CONFIGS = [
    {   # 1. All none — baseline
        "name": "all_none",
        "drift_detector":  {"name": "none"},
        "sample_selector": {"name": "random"},
        "memory_manager":  {"name": "none"},
        "anti_forgetting": {"name": "none"},
        "anomaly_scorer":  {"name": "pca"},
        "label_budget": 50, "lr": 1e-3,
    },
    {   # 2. SSF full combination
        "name": "ssf_full",
        "drift_detector":  {"name": "ssf", "drift_threshold": 0.05},
        "sample_selector": {"name": "ssf"},
        "memory_manager":  {"name": "ssf", "max_size": 200},
        "anti_forgetting": {"name": "lwf_ssf", "lwf_lambda": 0.5},
        "anomaly_scorer":  {"name": "pca"},
        "label_budget": 50, "lr": 1e-3,
    },
    {   # 3. CND-IDS full combination (no drift detector, no memory buffer — uses teacher LwF)
        "name": "cndids_full",
        "drift_detector":  {"name": "none"},
        "sample_selector": {"name": "all"},
        "memory_manager":  {"name": "none"},
        "anti_forgetting": {"name": "cndids"},
        "anomaly_scorer":  {"name": "pca"},
        "label_budget": 50, "lr": 1e-3,
    },
    {   # 4. CADE drift + GPM forgetting + FIFO buffer (SPIDER-style)
        "name": "cade_gpm",
        "drift_detector":  {"name": "cade"},
        "sample_selector": {"name": "random"},
        "memory_manager":  {"name": "fifo", "max_size": 200},
        "anti_forgetting": {"name": "gpm", "threshold": 0.97},
        "anomaly_scorer":  {"name": "cade_mad"},
        "label_budget": 50, "lr": 1e-3,
    },
    {   # 5. SSF drift + CND-IDS forgetting + CADE scorer (cross combination)
        "name": "ssf_cndids_cade_cross",
        "drift_detector":  {"name": "ssf"},
        "sample_selector": {"name": "ssf"},
        "memory_manager":  {"name": "ssf", "max_size": 200},
        "anti_forgetting": {"name": "cndids"},
        "anomaly_scorer":  {"name": "cade_mad"},
        "label_budget": 50, "lr": 1e-3,
    },
]


def _default_model(dim: int = 121):
    from testbed.pipeline.models import FCLAutoEncoder
    return FCLAutoEncoder(input_dim=dim, hidden_dim=64, latent_dim=32)


def run_smoke_tests():
    results = []
    for cfg in SMOKE_CONFIGS:
        name = cfg["name"]
        try:
            tasks = make_dummy_tasks()
            model = _default_model()
            client = CLClient(model=model, config=cfg, device='cpu')

            for X, y in tasks:
                client.update(X, y)

            # Fit anomaly scorer on first task normal data
            X0, y0 = tasks[0]
            normal = X0[y0 == 0]
            if len(normal) == 0:
                normal = X0[:5]
            client.fit_anomaly_scorer(normal)

            out = client.infer(tasks[-1][0])
            f1 = f1_score(tasks[-1][1].numpy(), out["predictions"].numpy())
            status = "PASS" if f1 >= 0.0 else "FAIL"
        except Exception as e:
            status = f"FAIL: {e}"
            f1 = -1.0

        results.append({"name": name, "status": status, "f1": round(f1, 4)})
        print(f"  [{status}] {name}  f1={f1:.4f}")

    os.makedirs('./testbed/results', exist_ok=True)
    with open('./testbed/results/smoke_test.json', 'w') as fp:
        json.dump(results, fp, indent=2)

    all_pass = all("PASS" in r["status"] for r in results)
    print()
    if all_pass:
        print("[ALL PASS] Smoke test PASSED")
    else:
        print("[FAIL] Some combinations failed - check logs above")
    return all_pass


if __name__ == '__main__':
    run_smoke_tests()
