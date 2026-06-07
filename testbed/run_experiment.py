"""CLI entry point for FCL testbed experiments.

Usage examples:
  # Single config experiment
  python run_experiment.py --config configs/exp_ssf_full.yaml

  # Full grid search
  python run_experiment.py --grid

  # Subset grid search
  python run_experiment.py --grid --subset anti_forgetting=gpm,lwf_ssf

  # Dummy data grid search
  python run_experiment.py --grid --dataset dummy --n_tasks 3

  # Visualise existing results
  python run_experiment.py --visualize

  # Run tests
  python run_experiment.py --test
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import logging
import subprocess
from typing import Dict, Any

import torch
import yaml
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

def _parse_subset(subset_str: str) -> Dict[str, list]:
    """Parse 'key=val1,val2 key2=val3' into a dict."""
    result = {}
    for part in subset_str.split():
        if '=' not in part:
            continue
        key, vals = part.split('=', 1)
        result[key.strip()] = [v.strip() for v in vals.split(',')]
    return result


def _default_model(dim: int = 121) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
    )


def _make_dummy_tasks(n: int = 500, dim: int = 121, n_tasks: int = 5):
    X = torch.randn(n, dim)
    y = torch.randint(0, 2, (n,))
    size = n // n_tasks
    return [(X[i*size:(i+1)*size], y[i*size:(i+1)*size])
            for i in range(n_tasks)]


# ── Single config experiment ───────────────────────────────────────────────

def run_single_config(config_path: str, data_dir: str = './data/',
                      device: str = 'cpu') -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    logger.info(f"Running experiment from config: {config_path}")
    dataset = cfg.pop('dataset', 'dummy')
    n_tasks = cfg.pop('n_tasks', 5)
    dim = 121

    if dataset == 'dummy':
        tasks = _make_dummy_tasks(dim=dim, n_tasks=n_tasks)
    else:
        from testbed.data.dataset_loader import (load_nslkdd, load_unswnb15,
                                                  split_into_tasks)
        loader = {'nslkdd': load_nslkdd, 'unswnb15': load_unswnb15}[dataset]
        ds = loader(data_dir)
        dim = ds['X'].shape[1]
        tasks = split_into_tasks(ds, n_tasks)

    from testbed.pipeline.cl_client import CLClient
    from testbed.experiments.metrics import f1_score, backward_transfer

    model = _default_model(dim)
    client = CLClient(model=model, config=cfg, device=device)

    perf_matrix = []
    for i, (X_tr, y_tr, *_) in enumerate(tasks):
        client.update(X_tr, y_tr)
        row = []
        for j, task in enumerate(tasks):
            X_te = task[2] if len(task) == 4 else task[0]
            y_te = task[3] if len(task) == 4 else task[1]
            out = client.infer(X_te)
            row.append(f1_score(y_te.numpy(), out['predictions'].numpy()))
        perf_matrix.append(row)
        logger.info(f"Task {i+1}/{n_tasks} — current F1: {row[i]:.3f}")

    bwt = backward_transfer(perf_matrix)
    final_f1 = sum(perf_matrix[-1]) / len(perf_matrix[-1])
    print(f"\n=== Results ===")
    print(f"Final F1:  {final_f1:.4f}")
    print(f"BWT:       {bwt:.4f}")


# ── Grid search ────────────────────────────────────────────────────────────

def run_grid_search(args) -> pd.DataFrame:
    from testbed.experiments.grid_runner import run_grid

    subset = None
    if args.subset:
        subset = _parse_subset(args.subset)
        logger.info(f"Subset grid: {subset}")

    df = run_grid(
        dataset=args.dataset,
        label_budget=args.label_budget,
        subset=subset,
        n_tasks=args.n_tasks,
        device=args.device,
        dim=args.dim,
        max_samples_per_task=args.max_samples,
    )
    return df


# ── Summary printer ────────────────────────────────────────────────────────

def print_summary(results_dir: str = './testbed/results') -> None:
    summary_path = os.path.join(results_dir, 'summary.csv')
    if not os.path.exists(summary_path):
        print("No summary.csv found. Run experiments first.")
        return

    df = pd.read_csv(summary_path)
    n_total = len(df)
    print(f"\n=== 실험 결과 요약 ===")
    print(f"총 실험 수: {n_total}개")

    slot_cols = ['drift_detector', 'sample_selector', 'memory_manager',
                 'anti_forgetting', 'anomaly_scorer']
    slot_cols = [c for c in slot_cols if c in df.columns]

    def _top5(df, sort_col, ascending=False):
        top = df.nlargest(5, sort_col) if not ascending else df.nsmallest(5, sort_col)
        print(f"\n[{sort_col} 기준 Top 5]")
        print(f"{'순위':<5} {'exp_id':<10} {' '.join(f'{c[:8]:<10}' for c in slot_cols)}"
              f" {'F1':<8} {'BWT':<8} {'FWT':<8}")
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            vals = ' '.join(f"{str(row.get(c,'?'))[:8]:<10}" for c in slot_cols)
            print(f"{rank:<5} {str(row.get('exp_id','?')):<10} {vals}"
                  f" {row.get('f1', 0):<8.4f} {row.get('bwt', 0):<8.4f}"
                  f" {row.get('fwt', 0):<8.4f}")

    _top5(df, 'f1')
    _top5(df, 'bwt')
    _top5(df, 'label_efficiency')

    # Save best config
    best = df.nlargest(1, 'f1').iloc[0]
    best_cfg: Dict[str, Any] = {}
    for c in slot_cols:
        best_cfg[c] = {'name': best.get(c, 'none')}
    best_cfg['label_budget'] = 50
    best_cfg['lr'] = 0.001
    best_cfg['n_tasks'] = 5

    best_path = os.path.join(results_dir, '..', 'configs', 'best_config.yaml')
    os.makedirs(os.path.dirname(best_path), exist_ok=True)
    with open(best_path, 'w') as f:
        yaml.dump(best_cfg, f, default_flow_style=False)
    print(f"\n최적 조합 config 저장: {os.path.abspath(best_path)}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='FCL Testbed CLI')
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--grid', action='store_true', help='Run grid search')
    parser.add_argument('--subset', type=str, default=None,
                        help='Subset grid: "anti_forgetting=gpm,lwf_ssf"')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate plots from existing results')
    parser.add_argument('--test', action='store_true', help='Run pytest')
    parser.add_argument('--dataset', type=str, default='dummy',
                        help='Dataset: dummy / nslkdd / unswnb15')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--n_tasks', type=int, default=5)
    parser.add_argument('--label_budget', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dim', type=int, default=121,
                        help='Feature dimension for dummy data')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='실제 데이터 태스크당 최대 샘플 수 (시간 단축용)')
    args = parser.parse_args()

    if args.test:
        logger.info("Running pytest...")
        ret = subprocess.run(
            [sys.executable, '-m', 'pytest', './testbed/tests/', '-v', '--tb=short'],
            cwd=os.path.dirname(os.path.abspath(__file__)) + '/..')
        sys.exit(ret.returncode)

    if args.visualize:
        from testbed.experiments.visualizer import run_all_plots
        run_all_plots('./testbed/results')
        return

    if args.config:
        run_single_config(args.config, args.data_dir, args.device)

    if args.grid:
        df = run_grid_search(args)
        print_summary('./testbed/results')
        from testbed.experiments.visualizer import run_all_plots
        run_all_plots('./testbed/results')
        _print_completion_message()
        return

    if not any([args.config, args.grid, args.visualize, args.test]):
        parser.print_help()


def _print_completion_message():
    msg = "\n" + "="*48 + "\n"
    msg += "[DONE] FCL Testbed Build & Experiment Complete\n"
    msg += "="*48 + "\n"
    msg += "- Testbed location: ./testbed/\n"
    msg += "- Experiment results: ./testbed/results/summary.csv\n"
    msg += "- Visualisations: ./testbed/results/plots/\n"
    msg += "- Best config: ./testbed/configs/best_config.yaml\n"
    msg += "\nNext steps:\n"
    msg += "  Real data grid search:\n"
    msg += "    python run_experiment.py --grid --dataset nslkdd --data_dir ./data/\n"
    msg += "  Single experiment:\n"
    msg += "    python run_experiment.py --config configs/best_config.yaml\n"
    msg += "  Visualise:\n"
    msg += "    python run_experiment.py --visualize\n"
    msg += "="*48 + "\n"
    print(msg)


if __name__ == '__main__':
    main()
