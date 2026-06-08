"""Result visualisation utilities.

저장 지표: F1, Precision, Recall, FPR (4개만)
- 모든 플롯 함수를 DataFrame 직접 수신 방식으로 리팩토링
- run_all_plots()가 dataset 컬럼으로 데이터셋별 서브디렉토리 분리 생성
- precision / recall / FPR 히트맵 + Recall vs FPR 트레이드오프 산점도 생성
- plot_guide.md 자동 생성 (각 플롯의 의미·해석 방법 설명)
- 기존 공개 API (plot_heatmap, plot_bwt_ranking, plot_pareto) 유지 (backward compat.)
"""

import os
import warnings
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# 내부 헬퍼 — DataFrame 직접 수신
# ---------------------------------------------------------------------------

def _heatmap_from_df(df: pd.DataFrame, x_col: str, y_col: str,
                     metric: str, out_path: str,
                     title_suffix: str = '') -> None:
    """DataFrame을 직접 받아 피벗 히트맵 생성·저장."""
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except ImportError:
        print("seaborn / matplotlib not installed — skipping heatmap")
        return

    if x_col not in df.columns or y_col not in df.columns:
        print(f"Columns {x_col!r} or {y_col!r} not found — skipping heatmap")
        return
    if metric not in df.columns:
        print(f"Metric {metric!r} not found — skipping heatmap")
        return

    pivot = df.pivot_table(values=metric, index=y_col, columns=x_col,
                           aggfunc='mean')
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.4),
                                    max(4, len(pivot) * 1.2)))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax,
                linewidths=0.5, cbar_kws={'label': metric})
    title = f"{metric.upper()} — {y_col} × {x_col}"
    if title_suffix:
        title = f"[{title_suffix}] {title}"
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(x_col, fontsize=10)
    ax.set_ylabel(y_col, fontsize=10)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap → {out_path}")


def _precision_recall_heatmap_from_df(df: pd.DataFrame, x_col: str, y_col: str,
                                       out_path: str,
                                       title_suffix: str = '') -> None:
    """Precision과 Recall을 나란히 보여주는 2-패널 히트맵."""
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except ImportError:
        print("seaborn / matplotlib not installed — skipping precision-recall heatmap")
        return

    if 'precision' not in df.columns or 'recall' not in df.columns:
        return
    if x_col not in df.columns or y_col not in df.columns:
        return

    pivot_prec = df.pivot_table(values='precision', index=y_col, columns=x_col,
                                aggfunc='mean')
    pivot_rec  = df.pivot_table(values='recall',    index=y_col, columns=x_col,
                                aggfunc='mean')
    if pivot_prec.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(pivot_prec.columns) * 2.8),
                                             max(4, len(pivot_prec) * 1.2)))
    for ax, pivot, label in zip(axes,
                                 [pivot_prec, pivot_rec],
                                 ['PRECISION (알람 정확도)', 'RECALL / Detection Rate (공격 탐지율)']):
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax,
                    linewidths=0.5, vmin=0, vmax=1,
                    cbar_kws={'label': label})
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xlabel(x_col, fontsize=9)
        ax.set_ylabel(y_col, fontsize=9)

    sup = f"Precision & Recall — {y_col} × {x_col}"
    if title_suffix:
        sup = f"[{title_suffix}] {sup}"
    fig.suptitle(sup, fontsize=12, fontweight='bold', y=1.02)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved precision-recall heatmap → {out_path}")


def _recall_ranking_from_df(df: pd.DataFrame, top_n: int, out_path: str,
                             title_suffix: str = '') -> None:
    """Recall(Detection Rate) 상위 N개 조합 수평 막대 그래프 생성·저장."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping Recall ranking")
        return

    if 'recall' not in df.columns:
        print("'recall' column missing — skipping ranking")
        return

    top = df.nlargest(top_n, 'recall').copy()
    slot_cols = ['drift_detector', 'sample_selector', 'memory_manager',
                 'anti_forgetting', 'anomaly_scorer']
    existing = [c for c in slot_cols if c in top.columns]
    top['label'] = top[existing].apply(
        lambda r: '/'.join(r.astype(str)), axis=1)

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.5)))
    colors = ['#2196F3' if v >= 0.7 else '#FF9800' for v in top['recall']]
    ax.barh(top['label'], top['recall'], color=colors)
    ax.axvline(0.7, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Recall (Detection Rate) — 높을수록 공격 탐지율이 높음')
    title = f'Top {top_n} combinations by Recall (Detection Rate)'
    if title_suffix:
        title = f'[{title_suffix}] {title}'
    ax.set_title(title, fontweight='bold')
    ax.text(0.01, 0.02, '파란색: Recall≥0.7 (권장)  /  주황색: Recall<0.7 (낮은 탐지율)',
            transform=ax.transAxes, fontsize=8, color='gray')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved Recall ranking → {out_path}")


def _recall_fpr_tradeoff_from_df(df: pd.DataFrame, out_path: str,
                                  title_suffix: str = '') -> None:
    """Recall vs FPR 트레이드오프 산점도 (Pareto 경계선 포함) 생성·저장."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping tradeoff plot")
        return

    if 'recall' not in df.columns or 'fpr' not in df.columns:
        print("Required columns (recall, fpr) missing — skipping tradeoff plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df['fpr'], df['recall'],
                    alpha=0.5, s=40, c=df.get('f1', 0.5),
                    cmap='YlGnBu', vmin=0, vmax=1, label='모든 조합')
    plt.colorbar(sc, ax=ax, label='F1 Score')

    # Pareto front: 낮은 FPR + 높은 Recall이 이상적 → 좌상단 점들
    pts = df[['fpr', 'recall']].dropna().values
    if len(pts) > 1:
        # 최소 FPR + 최대 Recall: x를 -FPR로 변환하여 두 축 모두 최대화
        pts_inv = np.column_stack([-pts[:, 0], pts[:, 1]])
        pareto_mask = _pareto_front(pts_inv)
        pareto = pts[pareto_mask]
        pareto_sorted = pareto[pareto[:, 0].argsort()]
        ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1],
                'r-o', linewidth=2, markersize=5, label='Pareto 최적 경계선')

    ax.set_xlabel('FPR (False Positive Rate) — 낮을수록 오탐 적음', fontsize=10)
    ax.set_ylabel('Recall (Detection Rate) — 높을수록 공격 탐지율 높음', fontsize=10)
    title = 'Recall vs FPR Tradeoff (좌상단 = 이상적)'
    if title_suffix:
        title = f'[{title_suffix}] {title}'
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.text(0.01, 0.02,
            'Pareto 선 위: 같은 탐지율로 오탐 적음, 또는 같은 오탐률로 탐지율 높음',
            transform=ax.transAxes, fontsize=8, color='gray')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved Recall-FPR tradeoff → {out_path}")


def _compare_datasets_plot(df_all: pd.DataFrame, out_dir: str) -> None:
    """두 데이터셋의 F1 점수를 나란히 비교하는 히트맵 (anti_forgetting × dataset)."""
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if 'dataset' not in df_all.columns or 'anti_forgetting' not in df_all.columns:
        return

    pivot = df_all.pivot_table(values='f1', index='anti_forgetting',
                               columns='dataset', aggfunc='mean')
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 2.5),
                                    max(4, len(pivot) * 1.2)))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, cbar_kws={'label': 'F1 Score'})
    ax.set_title('F1 Score — NSL-KDD vs UNSW-NB15 (Anti-Forgetting별 비교)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=10)
    ax.set_ylabel('Anti-Forgetting Strategy', fontsize=10)

    out_path = os.path.join(out_dir, 'compare_datasets_f1.png')
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved dataset comparison → {out_path}")


def _generate_plot_guide(plots_dir: str, datasets: list) -> None:
    """각 플롯의 의미와 해석 방법을 정리한 plot_guide.md 생성."""
    content = f"""\
# FCL Testbed — 시각화 가이드 (Plot Guide)

생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
포함 데이터셋: {', '.join(datasets) if datasets else '(없음)'}

## 디렉토리 구조

```
results/plots/
├── nslkdd/                    # NSL-KDD 데이터셋 단독 결과
├── unswnb15/                  # UNSW-NB15 데이터셋 단독 결과
├── all/                       # 두 데이터셋 합산 (비교 기준선)
├── compare_datasets_f1.png    # 두 데이터셋 나란히 비교
└── plot_guide.md              # 이 파일
```

---

## 각 플롯 설명

### 1. `heatmap_drift_x_af_f1.png` — F1 히트맵 (드리프트 탐지기 × 망각방지 전략)

| 항목 | 내용 |
|------|------|
| **X축** | Anti-Forgetting 전략 (`none` / `lwf_ssf` / `cndids` / `gpm`) |
| **Y축** | Drift Detector (`none` / `ssf` / `cade`) |
| **색상** | 평균 F1 점수 (진한 파란색 = 높은 F1 = 좋음) |
| **수치** | 각 셀 = 해당 조합의 모든 나머지 컴포넌트 평균 F1 |

**해석 방법**:
- 특정 행 전체가 밝다 → 해당 Drift Detector가 F1에 부정적 영향
- 특정 열 전체가 진하다 → 해당 Anti-Forgetting 전략이 탐지 성능에 유리
- 하나의 셀만 유독 진하다 → 해당 조합 쌍에 시너지 효과 존재

**IDS 문맥**: F1은 공격 탐지율(Recall)과 알람 정확도(Precision)의 조화평균.
NSL-KDD/UNSW-NB15처럼 클래스 불균형이 심한 데이터에서 단순 accuracy보다 신뢰할 수 있는 지표.

---

### 2. `heatmap_precision_recall.png` — Precision & Recall 2-패널 히트맵

| 항목 | 내용 |
|------|------|
| **왼쪽 패널** | Precision (정밀도): 이상 예측 중 실제 공격 비율 |
| **오른쪽 패널** | Recall (재현율, Detection Rate): 실제 공격 중 탐지한 비율 |
| **X축** | Anti-Forgetting 전략 |
| **Y축** | Drift Detector |

**해석 방법**:
- Precision 높고 Recall 낮음 → 알람은 정확하나 공격을 많이 놓침 (보수적 탐지)
- Precision 낮고 Recall 높음 → 공격 잘 잡지만 오탐 많음 (공격적 탐지)
- 두 패널 모두 진한 셀 → 오탐도 적고 탐지율도 높은 최선의 조합

**IDS 문맥**:
- **Precision**: 보안 운영자의 알람 피로도(alert fatigue)에 직결.
  낮은 precision → 오탐 알람 폭주 → 실제 위협 무시 위험
- **Recall**: 공격 미탐지(FN) 비율의 역수.
  낮은 recall → 실제 침입을 탐지 못함 → 보안 사고 발생

---

### 3. `heatmap_drift_x_af_fpr.png` — FPR 히트맵 (드리프트 탐지기 × 망각방지 전략)

| 항목 | 내용 |
|------|------|
| **X축** | Anti-Forgetting 전략 |
| **Y축** | Drift Detector |
| **색상** | 평균 FPR (밝을수록 낮은 FPR = 좋음) |

**해석 방법**:
- FPR = FP / (FP + TN): 정상 트래픽 중 공격으로 잘못 예측한 비율
- 셀 값이 낮을수록 오탐이 적은 조합
- Recall 히트맵과 함께 보면 탐지율-오탐률 트레이드오프 파악 가능

**IDS 문맥**: 높은 FPR은 SOC(보안 운영 센터) 분석가의 알람 피로를 유발하여
실제 공격 경보가 묻힐 위험이 있다.

---

### 4. `recall_ranking.png` — Recall 상위 10개 조합 막대 그래프

| 항목 | 내용 |
|------|------|
| **X축** | Recall (Detection Rate) 값 (오른쪽 길수록 좋음) |
| **Y축** | 조합 라벨 (`drift/selector/memory/forgetting/scorer` 순) |
| **파란색** | Recall ≥ 0.7 (권장 탐지율 이상) |
| **주황색** | Recall < 0.7 (낮은 탐지율 경고) |

**해석 방법**:
- 상단에 위치한 조합이 공격 탐지율이 가장 높은 최선의 조합
- 회색 점선(0.7 기준선)을 넘는 조합이 실용적 배포 후보
- FPR 히트맵과 함께 봐야 오탐 여부도 확인 가능

---

### 5. `recall_fpr_tradeoff.png` — Recall vs FPR 트레이드오프

| 항목 | 내용 |
|------|------|
| **X축** | FPR (낮을수록 좋음, 좌측) |
| **Y축** | Recall (높을수록 좋음, 상단) |
| **점 색상** | F1 점수 (진할수록 높음) |
| **빨간선** | Pareto 최적 경계선 (비지배 해집합) |

**해석 방법**:
- 좌상단에 위치한 점 = 오탐 적으면서 탐지율도 높은 이상적 조합
- Pareto 선상의 점들이 실용적 배포 후보
- 선이 좌상단 모서리에 가까울수록 전체 조합의 품질이 우수함

**IDS 문맥**: 보안 정책에 따라 Recall 우선(공격 놓치지 않기) 또는 FPR 우선(오탐 최소화)
중 하나를 선택할 수 있으며, 이 플롯이 그 트레이드오프를 한눈에 보여준다.

---

### 6. `compare_datasets_f1.png` — NSL-KDD vs UNSW-NB15 F1 비교

| 항목 | 내용 |
|------|------|
| **X축** | Dataset (nslkdd / unswnb15) |
| **Y축** | Anti-Forgetting 전략 |
| **색상** | 평균 F1 (주황-빨강 계열, 진할수록 높음) |

**해석 방법**:
- 같은 Anti-Forgetting 전략이 두 데이터셋에서 일관되게 좋으면 강건한 전략
- 특정 데이터셋에서만 좋으면 데이터 특성 의존성이 있음
- NSL-KDD(121 features)와 UNSW-NB15(196 features)는 특성 차원과 공격 유형이 다름

---

## 평가 지표 요약표

| 지표 | 수식 | 이상값 | IDS 의미 |
|------|------|--------|---------|
| **F1** | 2·P·R/(P+R) | 1.0 | 탐지 성능 균형 요약 지표 |
| **Precision** | TP/(TP+FP) | 1.0 | 알람 정확도, 오탐(FP) 비용 |
| **Recall** | TP/(TP+FN) | 1.0 | 공격 탐지율, 미탐(FN) 비용 |
| **FPR** | FP/(FP+TN) | 0.0 | 정상 트래픽 오탐률 |
"""
    guide_path = os.path.join(plots_dir, 'plot_guide.md')
    os.makedirs(plots_dir, exist_ok=True)
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Saved plot guide → {guide_path}")


# ---------------------------------------------------------------------------
# 내부 유틸리티
# ---------------------------------------------------------------------------

def _load_results(results_dir: str) -> pd.DataFrame:
    summary = os.path.join(results_dir, 'summary.csv')
    if not os.path.exists(summary):
        raise FileNotFoundError(f"summary.csv not found in {results_dir}")
    return pd.read_csv(summary)


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


# ---------------------------------------------------------------------------
# 공개 API (backward compatibility 유지)
# ---------------------------------------------------------------------------

def plot_heatmap(results_dir: str, x_axis: str, y_axis: str,
                 metric: str = 'f1') -> None:
    """Save x_axis × y_axis heatmap of mean metric values (legacy API).

    데이터셋 구분 없이 summary.csv 전체를 사용한다.
    데이터셋별 분리된 플롯은 run_all_plots()를 사용한다.

    Args:
        results_dir: Directory containing summary.csv.
        x_axis: Column name for the x axis (e.g. 'anti_forgetting').
        y_axis: Column name for the y axis (e.g. 'drift_detector').
        metric: Metric column to aggregate (default 'f1').
    """
    df = _load_results(results_dir)
    out = os.path.join(results_dir, 'plots',
                       f'heatmap_{y_axis}_x_{x_axis}_{metric}.png')
    _heatmap_from_df(df, x_axis, y_axis, metric, out, title_suffix='ALL')


def plot_bwt_ranking(results_dir: str, top_n: int = 10) -> None:
    """Save bar chart of top_n combinations by Recall (legacy name kept for compat).

    Args:
        results_dir: Directory containing summary.csv.
        top_n: Number of top combinations to show (default 10).
    """
    df = _load_results(results_dir)
    out = os.path.join(results_dir, 'plots', 'recall_ranking.png')
    _recall_ranking_from_df(df, top_n, out, title_suffix='ALL')


def plot_pareto(results_dir: str) -> None:
    """Save Recall vs FPR tradeoff scatter with Pareto front (legacy name kept).

    Args:
        results_dir: Directory containing summary.csv.
    """
    df = _load_results(results_dir)
    out = os.path.join(results_dir, 'plots', 'recall_fpr_tradeoff.png')
    _recall_fpr_tradeoff_from_df(df, out, title_suffix='ALL')


# ---------------------------------------------------------------------------
# 메인 — 데이터셋별 분리 플롯 생성
# ---------------------------------------------------------------------------

def run_all_plots(results_dir: str = './testbed/results') -> None:
    """Dataset별로 분리된 서브디렉토리에 모든 표준 플롯 생성.

    플롯 구조:
        results_dir/plots/
            nslkdd/              ← NSL-KDD 단독 결과
            unswnb15/            ← UNSW-NB15 단독 결과
            all/                 ← 두 데이터셋 합산 (기준선)
            compare_datasets_f1.png
            plot_guide.md        ← 각 플롯 설명 가이드

    Args:
        results_dir: summary.csv가 위치한 디렉토리.
    """
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    df_all = _load_results(results_dir)

    # 데이터셋별 그룹 구성 + 합산 그룹
    has_ds_col = 'dataset' in df_all.columns
    groups = []
    if has_ds_col:
        for ds in sorted(df_all['dataset'].dropna().unique()):
            groups.append((ds, df_all[df_all['dataset'] == ds]))
    groups.append(('all', df_all))

    for ds_name, df in groups:
        out_dir = os.path.join(results_dir, 'plots', ds_name)
        os.makedirs(out_dir, exist_ok=True)

        # 1. F1 히트맵: drift_detector × anti_forgetting
        _heatmap_from_df(
            df, 'anti_forgetting', 'drift_detector', 'f1',
            os.path.join(out_dir, 'heatmap_drift_x_af_f1.png'),
            title_suffix=ds_name,
        )

        # 2. Precision & Recall 2-패널 히트맵 (컬럼 존재 시)
        if 'precision' in df.columns and 'recall' in df.columns:
            _precision_recall_heatmap_from_df(
                df, 'anti_forgetting', 'drift_detector',
                os.path.join(out_dir, 'heatmap_precision_recall.png'),
                title_suffix=ds_name,
            )

        # 3. FPR 히트맵 (컬럼 존재 시)
        if 'fpr' in df.columns and df['fpr'].notna().any():
            _heatmap_from_df(
                df, 'anti_forgetting', 'drift_detector', 'fpr',
                os.path.join(out_dir, 'heatmap_drift_x_af_fpr.png'),
                title_suffix=ds_name,
            )

        # 4. Recall 상위 10개 조합 랭킹
        _recall_ranking_from_df(
            df, 10,
            os.path.join(out_dir, 'recall_ranking.png'),
            title_suffix=ds_name,
        )

        # 5. Recall vs FPR 트레이드오프 산점도
        _recall_fpr_tradeoff_from_df(
            df,
            os.path.join(out_dir, 'recall_fpr_tradeoff.png'),
            title_suffix=ds_name,
        )

    # 두 데이터셋 나란히 비교 플롯
    if has_ds_col and df_all['dataset'].nunique() > 1:
        _compare_datasets_plot(df_all, os.path.join(results_dir, 'plots'))

    # 플롯 가이드 생성
    ds_list = [ds for ds, _ in groups if ds != 'all']
    _generate_plot_guide(os.path.join(results_dir, 'plots'), ds_list)
