"""Smoke test — validate 5 representative pipeline combinations.

Runs each config with dummy data (N=1500, dim=121, n_tasks=3).
Uses paper-canonical epoch settings for each combination.
PASS = no exception raised + f1 >= 0.0.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import json
import torch
import numpy as np

from testbed.pipeline.cl_client import CLClient
from testbed.experiments.metrics import f1_score


def make_dummy_tasks(n: int = 1500, dim: int = 121, n_tasks: int = 3):
    X = torch.randn(n, dim)
    y = torch.randint(0, 2, (n,))
    size = n // n_tasks
    return [(X[i*size:(i+1)*size], y[i*size:(i+1)*size])
            for i in range(n_tasks)]


# Paper-faithful epoch settings per canonical combination
# FROM: original paper code (see PAPER_EPOCH_CONFIGS in grid_runner.py for references)
SMOKE_CONFIGS = [
    {   # 1. CADE canonical — CADEModel, Adam, pretrain=250 / task=50
        "name": "cade_canonical",
        "drift_detector":  {"name": "cade"},
        "sample_selector": {"name": "random"},
        "memory_manager":  {"name": "none"},
        "anti_forgetting": {"name": "cade"},
        "anomaly_scorer":  {"name": "cade_mad"},
        "label_budget": 50, "optimizer": "adam", "lr": 1e-3,
        "pretrain_epochs": 250, "task_epochs": 50, "batch_size": 64,
    },
    {   # 2. SSF canonical — SSFModel, SGD, pretrain=200 / task=1
        "name": "ssf_canonical",
        "drift_detector":  {"name": "ssf", "drift_threshold": 0.05},
        "sample_selector": {"name": "ssf"},
        "memory_manager":  {"name": "ssf", "max_size": 200},
        "anti_forgetting": {"name": "lwf_ssf", "lwf_lambda": 0.5},
        "anomaly_scorer":  {"name": "pca"},
        "label_budget": 50, "optimizer": "sgd", "lr": 1e-3,
        "pretrain_epochs": 200, "task_epochs": 1, "batch_size": 128,
    },
    {   # 3. CND-IDS canonical — CNDIDSModel, Adam, pretrain=10 / task=20
        "name": "cndids_canonical",
        "drift_detector":  {"name": "ddm"},
        "sample_selector": {"name": "random"},
        "memory_manager":  {"name": "cndids", "capacity": 200},
        "anti_forgetting": {"name": "cndids"},
        "anomaly_scorer":  {"name": "pca"},
        "label_budget": 50, "optimizer": "adam", "lr": 1e-3,
        "pretrain_epochs": 10, "task_epochs": 20, "batch_size": 64,
    },
    {   # 4. SPIDER canonical — SSFModel, Adam, pretrain=5 / task=5 (no paper default)
        "name": "spider_canonical",
        "drift_detector":  {"name": "none"},
        "sample_selector": {"name": "random"},
        "memory_manager":  {"name": "fifo", "max_size": 200},
        "anti_forgetting": {"name": "gpm", "threshold": 0.97},
        "anomaly_scorer":  {"name": "pca"},
        "label_budget": 50, "optimizer": "adam", "lr": 1e-3,
        "pretrain_epochs": 5, "task_epochs": 5, "batch_size": 64,
    },
    {   # 5. Cross — SSF drift + CND-IDS memory + GPM anti-forgetting → CNDIDSModel
        "name": "cross_ssf_cnd_gpm",
        "drift_detector":  {"name": "ssf"},
        "sample_selector": {"name": "random"},
        "memory_manager":  {"name": "cndids", "capacity": 200},
        "anti_forgetting": {"name": "gpm"},
        "anomaly_scorer":  {"name": "cade_mad"},
        "label_budget": 50, "optimizer": "adam", "lr": 1e-3,
        "pretrain_epochs": 5, "task_epochs": 5, "batch_size": 64,
    },
]


def _make_model(cfg: dict, dim: int = 121):
    """Build paper-appropriate model for the config."""
    from testbed.pipeline.models import select_paper, build_model
    combo = {
        'drift_detector':  cfg.get('drift_detector', {}).get('name', 'none'),
        'sample_selector': cfg.get('sample_selector', {}).get('name', 'random'),
        'memory_manager':  cfg.get('memory_manager',  {}).get('name', 'none'),
        'anti_forgetting': cfg.get('anti_forgetting', {}).get('name', 'none'),
        'anomaly_scorer':  cfg.get('anomaly_scorer',  {}).get('name', 'pca'),
    }
    paper = select_paper(combo)
    return build_model(paper, dim)


def run_smoke_tests():
    results = []
    for cfg in SMOKE_CONFIGS:
        name = cfg["name"]
        try:
            tasks  = make_dummy_tasks()
            model  = _make_model(cfg)
            client = CLClient(model=model, config=cfg, device='cpu')

            for X, y in tasks:
                client.update(X, y)
                normal_i = X[y == 0]
                if len(normal_i) == 0:
                    normal_i = X[:5]
                client.fit_anomaly_scorer(normal_i)

            out = client.infer(tasks[-1][0])
            f1  = f1_score(tasks[-1][1].numpy(), out["predictions"].numpy())
            status = "PASS" if f1 >= 0.0 else "FAIL"
        except Exception as e:
            import traceback
            status = f"FAIL: {e}"
            print(traceback.format_exc())
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
