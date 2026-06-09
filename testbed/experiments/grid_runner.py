"""Grid search runner over component combinations."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import json
import logging
import traceback
import itertools
from typing import Optional, Dict, List, Any

import torch
import numpy as np
import pandas as pd

from testbed.pipeline.cl_client import CLClient
from testbed.experiments.metrics import (f1_score, precision_score,
                                          detection_rate, false_alarm_rate)

logger = logging.getLogger(__name__)

# grid_runner.py 기준으로 FCL 루트를 절대 경로로 계산
_FCL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COMPONENT_GRID = {
    # 4 × 2 × 4 × 5 × 2 = 320 combinations per dataset
    # Paper mapping (drift / sample / memory / anti / anomaly):
    #   CND-IDS : ddm  / random / cndids / cndids  / pca
    #   SSF     : ssf  / ssf    / ssf    / lwf_ssf / pca
    #   CADE    : cade / random / none   / cade    / cade_mad
    #   SPIDER  : none / random / fifo   / gpm     / pca
    "drift_detector":  ["none", "ssf", "cade", "ddm"],
    "sample_selector": ["random", "ssf"],
    "memory_manager":  ["none", "fifo", "ssf", "cndids"],
    "anti_forgetting": ["none", "cade", "cndids", "gpm", "lwf_ssf"],
    "anomaly_scorer":  ["pca", "cade_mad"],
}

# ---------------------------------------------------------------------------
# Paper-canonical training hyperparameters
# ---------------------------------------------------------------------------
# FROM original paper code:
#   SSF     (utils.py / ssf.py)   : SGD lr=0.001, bs=128, epochs=200 init / 1 per task
#   CND-IDS (AE_Exactor.py)       : Adam lr=0.001, bs=64,  epochs=10 init / 50 per task
#   CADE    (autoencoder.py)      : Adam lr=0.001, bs=64,  epochs=250 init / 50 per task
#   SPIDER/GPM                    : Adam lr=0.001, bs=64,  no fixed epoch count
PAPER_EPOCH_CONFIGS: Dict[str, Dict[str, Any]] = {
    'ssf': {
        'pretrain_epochs': 200,   # --epochs 200 (NSL-KDD README)
        'task_epochs':     1,     # --epoch_1 1 (code default; UNSW README uses 180)
        'batch_size':      128,
        'optimizer':       'sgd',
        'lr':              0.001,
    },
    'cndids': {
        'pretrain_epochs': 10,    # AE_Exactor train_epochs=10
        'task_epochs':     20,    # CND_IDS.fit() train_epochs=20
        'batch_size':      64,
        'optimizer':       'adam',
        'lr':              0.001,
    },
    'cade': {
        'pretrain_epochs': 250,   # --cae-epochs 250
        'task_epochs':     50,    # --mlp-epochs 50
        'batch_size':      64,
        'optimizer':       'adam',
        'lr':              0.001,
    },
    'gpm': {                      # SPIDER/GPM — no paper-specified epoch count
        'pretrain_epochs': 5,
        'task_epochs':     5,
        'batch_size':      64,
        'optimizer':       'adam',
        'lr':              0.001,
    },
    'default': {
        'pretrain_epochs': 5,
        'task_epochs':     5,
        'batch_size':      64,
        'optimizer':       'adam',
        'lr':              0.001,
    },
}

# 조합 약어 (파일명 단축용)
_SHORT = {
    "none": "none", "ssf": "ssf", "cade": "cade", "ddm": "ddm",
    "random": "rand",
    "fifo": "fifo", "cndids": "cnd",
    "gpm": "gpm", "lwf_ssf": "lwf",
    "pca": "pca", "cade_mad": "mad",
    "cade": "cade",
}


def _make_exp_name(dataset: str, combo: dict) -> str:
    short = lambda k, v: _SHORT.get(v, v[:4])
    return (
        f"{dataset}"
        f"__dr={short('drift_detector',  combo['drift_detector'])}"
        f"__sl={short('sample_selector', combo['sample_selector'])}"
        f"__mm={short('memory_manager',  combo['memory_manager'])}"
        f"__af={short('anti_forgetting', combo['anti_forgetting'])}"
        f"__sc={short('anomaly_scorer',  combo['anomaly_scorer'])}"
    )


def _make_dummy_tasks(n: int = 2000, dim: int = 121, n_tasks: int = 5):
    X = torch.randn(n, dim)
    y = torch.randint(0, 2, (n,))
    size = n // n_tasks
    return [(X[i*size:(i+1)*size], y[i*size:(i+1)*size])
            for i in range(n_tasks)]


def _select_paper(combo_dict: dict) -> str:
    """Determine which paper's epoch/hyperparameter settings to use."""
    from testbed.pipeline.models import select_paper
    anti = combo_dict.get('anti_forgetting', 'none')
    if anti == 'gpm':
        return 'gpm'
    return select_paper(combo_dict)


def _build_model(combo_dict: dict, dim: int, model_fn=None):
    """Build paper-appropriate model for the combination."""
    if model_fn:
        return model_fn(dim)
    from testbed.pipeline.models import select_paper, build_model
    paper = select_paper(combo_dict)
    return build_model(paper, dim)


def run_grid(dataset: str = 'dummy',
             model_fn=None,
             label_budget: int = 50,
             subset: Optional[Dict[str, List[str]]] = None,
             n_tasks: int = 5,
             device: str = 'cpu',
             dim: int = 121,
             max_samples_per_task: Optional[int] = None,
             n_epochs: int = 5,
             use_paper_epochs: bool = False):
    """Component 조합 그리드 서치 실행.

    Args:
        dataset:              'dummy', 'nslkdd', 'unswnb15'.
        model_fn:             Callable(dim) → nn.Module. None=논문 적합 모델 자동 선택.
        label_budget:         라운드당 레이블 최대 샘플 수.
        subset:               특정 슬롯만 비교할 때 사용.
        n_tasks:              연속학습 태스크 수.
        device:               Torch device.
        dim:                  dummy 데이터 특성 차원.
        max_samples_per_task: 태스크당 최대 샘플 수.
        n_epochs:             use_paper_epochs=False일 때 flat epoch 수.
        use_paper_epochs:     True면 각 조합의 논문 원본 epoch/optimizer 설정 사용.

    Returns:
        pd.DataFrame — summary 결과.
    """
    os.makedirs('./testbed/results', exist_ok=True)
    os.makedirs('./testbed/results/plots', exist_ok=True)
    errors_path  = './testbed/results/errors.log'
    summary_path = './testbed/results/summary.csv'

    grid = dict(COMPONENT_GRID)
    if subset:
        for slot, choices in subset.items():
            grid[slot] = choices

    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    logger.info(f"총 조합 수: {len(combos)}  (dataset={dataset})")

    if dataset == 'dummy':
        tasks      = _make_dummy_tasks(n=2000, dim=dim, n_tasks=n_tasks)
        actual_dim = dim
    else:
        tasks, actual_dim = _load_real_tasks(dataset, n_tasks, max_samples_per_task)

    records = []
    for combo in combos:
        combo_dict = dict(zip(keys, combo))
        exp_name   = _make_exp_name(dataset, combo_dict)
        out_path   = f'./testbed/results/{exp_name}.json'

        if os.path.exists(out_path):
            logger.info(f"[SKIP] {exp_name} (already exists)")
            with open(out_path) as f:
                records.append(json.load(f))
            continue

        config: Dict[str, Any] = {
            "drift_detector":  {"name": combo_dict["drift_detector"]},
            "sample_selector": {"name": combo_dict["sample_selector"]},
            "memory_manager":  {"name": combo_dict["memory_manager"], "max_size": 500},
            "anti_forgetting": {"name": combo_dict["anti_forgetting"]},
            "anomaly_scorer":  {"name": combo_dict["anomaly_scorer"]},
            "label_budget": label_budget,
        }

        if use_paper_epochs:
            paper     = _select_paper(combo_dict)
            epoch_cfg = PAPER_EPOCH_CONFIGS.get(paper, PAPER_EPOCH_CONFIGS['default'])
            config.update({
                'pretrain_epochs': epoch_cfg['pretrain_epochs'],
                'task_epochs':     epoch_cfg['task_epochs'],
                'batch_size':      epoch_cfg['batch_size'],
                'optimizer':       epoch_cfg['optimizer'],
                'lr':              epoch_cfg['lr'],
            })
            logger.info(f"  paper={paper} pretrain={epoch_cfg['pretrain_epochs']} "
                        f"task={epoch_cfg['task_epochs']} bs={epoch_cfg['batch_size']}")
        else:
            config.update({
                'n_epochs':   n_epochs,
                'batch_size': 64,
                'optimizer':  'adam',
                'lr':         1e-3,
            })

        try:
            model  = _build_model(combo_dict, actual_dim, model_fn)
            client = CLClient(model=model, config=config, device=device)

            perf_matrix = []

            for i, task in enumerate(tasks):
                X_tr, y_tr = task[0], task[1]
                client.update(X_tr, y_tr)

                # 학습 후 현재 인코더 기준으로 anomaly scorer 재학습
                normal_i = X_tr[y_tr == 0]
                if len(normal_i) == 0:
                    normal_i = X_tr[:max(1, len(X_tr)//2)]
                client.fit_anomaly_scorer(normal_i)

                row = []
                for j, t in enumerate(tasks):
                    X_te = t[2] if len(t) == 4 else t[0]
                    y_te = t[3] if len(t) == 4 else t[1]
                    out  = client.infer(X_te)
                    row.append(f1_score(y_te.numpy(), out['predictions'].numpy()))
                perf_matrix.append(row)

            final_f1 = float(np.mean(perf_matrix[-1]))

            _last     = tasks[-1]
            X_last_te = _last[2] if len(_last) == 4 else _last[0]
            y_last_te = _last[3] if len(_last) == 4 else _last[1]
            _last_out = client.infer(X_last_te)
            _y_true   = y_last_te.numpy()
            _y_pred   = _last_out['predictions'].numpy()
            final_precision = precision_score(_y_true, _y_pred)
            final_recall    = detection_rate(_y_true, _y_pred)
            final_fpr       = false_alarm_rate(_y_true, _y_pred)

            record = {
                'exp_name':    exp_name,
                'dataset':     dataset,
                **combo_dict,
                'f1':          round(final_f1, 4),
                'precision':   round(final_precision, 4),
                'recall':      round(final_recall, 4),
                'fpr':         round(final_fpr, 4),
                'perf_matrix': perf_matrix,
            }
            records.append(record)

            with open(out_path, 'w') as f:
                json.dump(record, f, indent=2)

            logger.info(
                f"[OK] {exp_name} → "
                f"F1={final_f1:.3f}  Prec={final_precision:.3f}  "
                f"Rec={final_recall:.3f}  FPR={final_fpr:.3f}"
            )

        except Exception as e:
            err_msg = f"[FAIL] {exp_name}\n{traceback.format_exc()}\n"
            logger.warning(err_msg)
            with open(errors_path, 'a') as f:
                f.write(err_msg)

    _new_cols = ['precision', 'recall', 'fpr']
    for r in records:
        for k in _new_cols:
            r.setdefault(k, float('nan'))

    df_records = [{k: v for k, v in r.items() if k != 'perf_matrix'} for r in records]
    df = pd.DataFrame(df_records)
    df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved → {summary_path}")
    return df


def _load_real_tasks(dataset: str, n_tasks: int,
                     max_samples_per_task: Optional[int] = None):
    sys.path.insert(0, './testbed')
    from testbed.data.dataset_loader import (load_nslkdd, load_unswnb15,
                                              split_into_tasks)
    loaders = {'nslkdd': load_nslkdd, 'unswnb15': load_unswnb15}
    if dataset not in loaders:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose 'nslkdd' or 'unswnb15'.")

    ds  = loaders[dataset](_FCL_ROOT)
    dim = ds['X'].shape[1]

    if max_samples_per_task is not None:
        total_max = max_samples_per_task * n_tasks
        N = len(ds['X'])
        if N > total_max:
            idx_normal = (ds['y'] == 0).nonzero(as_tuple=True)[0]
            idx_attack = (ds['y'] == 1).nonzero(as_tuple=True)[0]
            ratio  = len(idx_normal) / N
            n_norm = int(total_max * ratio)
            n_atk  = total_max - n_norm
            sel_n  = idx_normal[torch.randperm(len(idx_normal))[:n_norm]]
            sel_a  = idx_attack[torch.randperm(len(idx_attack))[:n_atk]]
            sel    = torch.cat([sel_n, sel_a])
            sel    = sel[torch.randperm(len(sel))]
            ds     = {'X': ds['X'][sel], 'y': ds['y'][sel], 'scaler': ds['scaler']}

    tasks = split_into_tasks(ds, n_tasks)
    return tasks, dim
