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
# testbed/experiments/grid_runner.py → FCL/
_FCL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COMPONENT_GRID = {
    # 4 × 2 × 4 × 4 × 2 = 256 combinations per dataset
    # Paper mapping (drift / sample / memory / anti / anomaly):
    #   CND-IDS : ddm  / random / cndids / cndids  / pca
    #   SSF     : ssf  / ssf    / ssf    / lwf_ssf / pca
    #   CADE    : cade / random / none   / none    / cade_mad
    #   SPIDER  : none / random / fifo   / gpm     / pca
    "drift_detector":  ["none", "ssf", "cade", "ddm"],
    "sample_selector": ["random", "ssf"],
    "memory_manager":  ["none", "fifo", "ssf", "cndids"],
    "anti_forgetting": ["none", "cndids", "gpm", "lwf_ssf"],
    "anomaly_scorer":  ["pca", "cade_mad"],
}

# 조합 약어 (파일명 단축용)
_SHORT = {
    # drift_detector
    "none": "none", "ssf": "ssf", "cade": "cade", "ddm": "ddm",
    # sample_selector
    "random": "rand",
    # memory_manager
    "fifo": "fifo", "cndids": "cnd",
    # anti_forgetting
    "gpm": "gpm", "lwf_ssf": "lwf",
    # anomaly_scorer
    "pca": "pca", "cade_mad": "mad",
}


def _make_exp_name(dataset: str, combo: dict) -> str:
    """조합 정보를 담은 실험 파일명 생성.

    형식: {dataset}__dr={drift}__sl={sel}__mm={mem}__af={af}__sc={scorer}
    예시: nslkdd__dr=ssf__sl=rand__mm=ssf__af=lwf__sc=pca
    """
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


def run_grid(dataset: str = 'dummy',
             model_fn=None,
             label_budget: int = 50,
             subset: Optional[Dict[str, List[str]]] = None,
             n_tasks: int = 5,
             device: str = 'cpu',
             dim: int = 121,
             max_samples_per_task: Optional[int] = None,
             n_epochs: int = 5):
    """Component 조합 그리드 서치 실행.

    Args:
        dataset: 'dummy', 'nslkdd', 'unswnb15'.
        model_fn: Callable(dim) → nn.Module. None이면 기본 MLP 사용.
        label_budget: 라운드당 레이블 최대 샘플 수.
        subset: 특정 슬롯만 비교할 때 사용. e.g. {"anti_forgetting": ["gpm","lwf_ssf"]}.
        n_tasks: 연속학습 태스크 수.
        device: Torch device.
        dim: dummy 데이터 특성 차원 (실제 데이터는 자동 결정).
        max_samples_per_task: 태스크당 최대 샘플 수. 실제 데이터 실험 시간 단축용.

    Returns:
        pd.DataFrame — summary 결과.
    """
    os.makedirs('./testbed/results', exist_ok=True)
    os.makedirs('./testbed/results/plots', exist_ok=True)
    errors_path = './testbed/results/errors.log'
    summary_path = './testbed/results/summary.csv'

    # 그리드 구성
    grid = dict(COMPONENT_GRID)
    if subset:
        for slot, choices in subset.items():
            grid[slot] = choices

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    logger.info(f"총 조합 수: {len(combos)}  (dataset={dataset})")

    # 데이터 로드
    if dataset == 'dummy':
        tasks = _make_dummy_tasks(n=500, dim=dim, n_tasks=n_tasks)
        actual_dim = dim
    else:
        tasks, actual_dim = _load_real_tasks(dataset, n_tasks,
                                             max_samples_per_task)

    records = []
    for combo in combos:
        combo_dict = dict(zip(keys, combo))
        exp_name = _make_exp_name(dataset, combo_dict)
        out_path = f'./testbed/results/{exp_name}.json'

        # 이미 완료된 실험은 스킵
        if os.path.exists(out_path):
            logger.info(f"[SKIP] {exp_name} (already exists)")
            with open(out_path) as f:
                records.append(json.load(f))
            continue

        config = {
            "drift_detector":  {"name": combo_dict["drift_detector"]},
            "sample_selector": {"name": combo_dict["sample_selector"]},
            "memory_manager":  {"name": combo_dict["memory_manager"],
                                "max_size": 500},
            "anti_forgetting": {"name": combo_dict["anti_forgetting"]},
            "anomaly_scorer":  {"name": combo_dict["anomaly_scorer"]},
            "label_budget": label_budget,
            "batch_size": 64,
            "n_epochs": n_epochs,
            "lr": 1e-3,
        }

        try:
            model = model_fn(actual_dim) if model_fn else _default_model(actual_dim)
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
                    out = client.infer(X_te)
                    row.append(f1_score(y_te.numpy(), out['predictions'].numpy()))
                perf_matrix.append(row)

            final_f1 = float(np.mean(perf_matrix[-1]))

            # 마지막 태스크 기준 precision / recall / FPR
            _last = tasks[-1]
            X_last_te = _last[2] if len(_last) == 4 else _last[0]
            y_last_te = _last[3] if len(_last) == 4 else _last[1]
            _last_out = client.infer(X_last_te)
            _y_true = y_last_te.numpy()
            _y_pred = _last_out['predictions'].numpy()
            final_precision = precision_score(_y_true, _y_pred)
            final_recall    = detection_rate(_y_true, _y_pred)
            final_fpr       = false_alarm_rate(_y_true, _y_pred)

            record = {
                'exp_name': exp_name,
                'dataset': dataset,
                **combo_dict,
                'f1':        round(final_f1, 4),
                'precision': round(final_precision, 4),
                'recall':    round(final_recall, 4),
                'fpr':       round(final_fpr, 4),
                'perf_matrix': perf_matrix,
            }
            records.append(record)

            # 파일명에 조합 정보 포함하여 저장
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

    # 기존 JSON 파일 재로드 시 지표 컬럼이 없을 수 있으므로 NaN으로 채움
    _new_cols = ['precision', 'recall', 'fpr']
    for r in records:
        for k in _new_cols:
            r.setdefault(k, float('nan'))

    # perf_matrix는 summary에서 제외
    df_records = [{k: v for k, v in r.items() if k != 'perf_matrix'}
                  for r in records]
    df = pd.DataFrame(df_records)
    df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved → {summary_path}")
    return df


def _default_model(dim: int) -> torch.nn.Module:
    from testbed.pipeline.models import FCLAutoEncoder
    hidden = max(128, dim)
    latent = max(64, dim // 2)
    return FCLAutoEncoder(input_dim=dim, hidden_dim=hidden, latent_dim=latent)


def _load_real_tasks(dataset: str, n_tasks: int,
                     max_samples_per_task: Optional[int] = None):
    """실제 데이터셋 로드 및 태스크 분할.

    Returns:
        (tasks, dim) — tasks는 (X_tr, y_tr, X_te, y_te) 리스트, dim은 특성 차원.
    """
    sys.path.insert(0, './testbed')
    from testbed.data.dataset_loader import (load_nslkdd, load_unswnb15,
                                              split_into_tasks)
    loaders = {'nslkdd': load_nslkdd, 'unswnb15': load_unswnb15}
    if dataset not in loaders:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose 'nslkdd' or 'unswnb15'.")

    ds = loaders[dataset](_FCL_ROOT)
    dim = ds['X'].shape[1]

    # 메모리/시간 절약: 전체 데이터를 서브샘플
    if max_samples_per_task is not None:
        total_max = max_samples_per_task * n_tasks
        N = len(ds['X'])
        if N > total_max:
            # 정상/이상 비율 유지하며 샘플링
            idx_normal = (ds['y'] == 0).nonzero(as_tuple=True)[0]
            idx_attack = (ds['y'] == 1).nonzero(as_tuple=True)[0]
            ratio = len(idx_normal) / N
            n_norm = int(total_max * ratio)
            n_atk  = total_max - n_norm
            sel_n = idx_normal[torch.randperm(len(idx_normal))[:n_norm]]
            sel_a = idx_attack[torch.randperm(len(idx_attack))[:n_atk]]
            sel = torch.cat([sel_n, sel_a])
            sel = sel[torch.randperm(len(sel))]
            ds = {'X': ds['X'][sel], 'y': ds['y'][sel], 'scaler': ds['scaler']}
            logger.info(f"Sampled {len(ds['X'])} rows from {N} "
                        f"(max_samples_per_task={max_samples_per_task})")

    tasks = split_into_tasks(ds, n_tasks)
    return tasks, dim
