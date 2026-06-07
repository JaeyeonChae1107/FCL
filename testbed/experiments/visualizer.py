"""Result visualisation utilities."""

import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')


def _load_results(results_dir: str) -> pd.DataFrame:
    summary = os.path.join(results_dir, 'summary.csv')
    if not os.path.exists(summary):
        raise FileNotFoundError(f"summary.csv not found in {results_dir}")
    return pd.read_csv(summary)


def plot_heatmap(results_dir: str, x_axis: str, y_axis: str,
                 metric: str = 'f1'):
    """Save x_axis × y_axis heatmap of mean metric values.

    Args:
        results_dir: Directory containing summary.csv.
        x_axis: Column name for the x axis (e.g. 'anti_forgetting').
        y_axis: Column name for the y axis (e.g. 'drift_detector').
        metric: Metric column to aggregate (default 'f1').
    """
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except ImportError:
        print("seaborn / matplotlib not installed — skipping heatmap")
        return

    df = _load_results(results_dir)
    if x_axis not in df.columns or y_axis not in df.columns:
        print(f"Columns {x_axis!r} or {y_axis!r} not found in results")
        return

    pivot = df.pivot_table(values=metric, index=y_axis, columns=x_axis,
                           aggfunc='mean')
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns)),
                                    max(4, len(pivot))))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax)
    ax.set_title(f'{metric} — {y_axis} × {x_axis}')
    out = os.path.join(results_dir, 'plots',
                       f'heatmap_{y_axis}_x_{x_axis}_{metric}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap → {out}")


def plot_bwt_ranking(results_dir: str, top_n: int = 10):
    """Save bar chart of top_n combinations by BWT (least forgetting).

    Args:
        results_dir: Directory containing summary.csv.
        top_n: Number of top combinations to show (default 10).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping BWT ranking")
        return

    df = _load_results(results_dir)
    if 'bwt' not in df.columns:
        print("'bwt' column missing — skipping ranking")
        return

    top = df.nlargest(top_n, 'bwt').copy()
    slot_cols = ['drift_detector', 'sample_selector', 'memory_manager',
                 'anti_forgetting', 'anomaly_scorer']
    existing = [c for c in slot_cols if c in top.columns]
    top['label'] = top[existing].apply(
        lambda r: '/'.join(r.astype(str)), axis=1)

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.4)))
    colors = ['#2196F3' if v >= 0 else '#F44336' for v in top['bwt']]
    ax.barh(top['label'], top['bwt'], color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('BWT (higher = less forgetting)')
    ax.set_title(f'Top {top_n} combinations by BWT')
    out = os.path.join(results_dir, 'plots', 'bwt_ranking.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved BWT ranking → {out}")


def plot_pareto(results_dir: str):
    """Save label_efficiency vs f1 scatter with Pareto front.

    Args:
        results_dir: Directory containing summary.csv.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping Pareto plot")
        return

    df = _load_results(results_dir)
    if 'f1' not in df.columns or 'label_efficiency' not in df.columns:
        print("Required columns missing — skipping Pareto plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['label_efficiency'], df['f1'],
               alpha=0.6, s=40, label='All combinations')

    # Pareto front
    pts = df[['label_efficiency', 'f1']].values
    pareto_mask = _pareto_front(pts)
    pareto = pts[pareto_mask]
    pareto_sorted = pareto[pareto[:, 0].argsort()]
    ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1],
            'r-o', linewidth=2, markersize=5, label='Pareto front')

    ax.set_xlabel('Label Efficiency (higher = fewer labels)')
    ax.set_ylabel('F1 Score')
    ax.set_title('Pareto Front — Label Efficiency vs F1')
    ax.legend()
    out = os.path.join(results_dir, 'plots', 'pareto.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved Pareto plot → {out}")


def _pareto_front(pts: np.ndarray) -> np.ndarray:
    """Return boolean mask for non-dominated points (maximise both axes)."""
    n = len(pts)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if (pts[j, 0] >= pts[i, 0] and pts[j, 1] >= pts[i, 1]
                    and (pts[j, 0] > pts[i, 0] or pts[j, 1] > pts[i, 1])):
                is_pareto[i] = False
                break
    return is_pareto


def run_all_plots(results_dir: str = './testbed/results'):
    """Generate all standard plots from results in results_dir.

    Args:
        results_dir: Directory containing summary.csv.
    """
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    plot_heatmap(results_dir, 'anti_forgetting', 'drift_detector', 'f1')
    plot_heatmap(results_dir, 'memory_manager', 'anti_forgetting', 'bwt')
    plot_bwt_ranking(results_dir, top_n=10)
    plot_pareto(results_dir)
