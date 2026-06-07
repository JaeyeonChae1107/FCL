"""Result visualisation utilities.

변경 내역 (B3 버그 수정):
- 모든 플롯 함수를 DataFrame 직접 수신 방식으로 리팩토링
- run_all_plots()가 dataset 컬럼으로 데이터셋별 서브디렉토리 분리 생성
- precision / recall / FPR / balanced_accuracy 히트맵 추가 (컬럼 존재 시)
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


def _bwt_ranking_from_df(df: pd.DataFrame, top_n: int, out_path: str,
                          title_suffix: str = '') -> None:
    """BWT 상위 N개 조합 수평 막대 그래프 생성·저장."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping BWT ranking")
        return

    if 'bwt' not in df.columns:
        print("'bwt' column missing — skipping ranking")
        return

    top = df.nlargest(top_n, 'bwt').copy()
    slot_cols = ['drift_detector', 'sample_selector', 'memory_manager',
                 'anti_forgetting', 'anomaly_scorer']
    existing = [c for c in slot_cols if c in top.columns]
    top['label'] = top[existing].apply(
        lambda r: '/'.join(r.astype(str)), axis=1)

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.5)))
    colors = ['#2196F3' if v >= 0 else '#F44336' for v in top['bwt']]
    ax.barh(top['label'], top['bwt'], color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('BWT — 높을수록 이전 태스크 성능 보존 (0=변화 없음, 음수=망각)')
    title = f'Top {top_n} combinations by BWT (Backward Transfer)'
    if title_suffix:
        title = f'[{title_suffix}] {title}'
    ax.set_title(title, fontweight='bold')
    ax.text(0.01, 0.02, '파란색: BWT≥0 (보존 또는 향상)  /  빨간색: BWT<0 (망각)',
            transform=ax.transAxes, fontsize=8, color='gray')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved BWT ranking → {out_path}")


def _pareto_from_df(df: pd.DataFrame, out_path: str,
                    title_suffix: str = '') -> None:
    """Label Efficiency vs F1 Pareto front 산점도 생성·저장."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping Pareto plot")
        return

    if 'f1' not in df.columns or 'label_efficiency' not in df.columns:
        print("Required columns missing — skipping Pareto plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['label_efficiency'], df['f1'],
               alpha=0.4, s=30, c='steelblue', label='모든 조합')

    pts = df[['label_efficiency', 'f1']].dropna().values
    if len(pts) > 1:
        pareto_mask = _pareto_front(pts)
        pareto = pts[pareto_mask]
        pareto_sorted = pareto[pareto[:, 0].argsort()]
        ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1],
                'r-o', linewidth=2, markersize=5, label='Pareto 최적 경계선')

    ax.set_xlabel('Label Efficiency — 높을수록 레이블 수 적음 (비용 절감)', fontsize=10)
    ax.set_ylabel('F1 Score (공격 탐지 성능)', fontsize=10)
    title = 'Pareto Front — Label Efficiency vs F1'
    if title_suffix:
        title = f'[{title_suffix}] {title}'
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.text(0.01, 0.02,
            'Pareto 선 위의 점: 같은 레이블 비용으로 더 높은 F1 또는 더 적은 비용으로 같은 F1',
            transform=ax.transAxes, fontsize=8, color='gray')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved Pareto plot → {out_path}")


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
├── nslkdd/          # NSL-KDD 데이터셋 단독 결과
├── unswnb15/        # UNSW-NB15 데이터셋 단독 결과
├── all/             # 두 데이터셋 합산 (비교 기준선)
├── compare_datasets_f1.png   # 두 데이터셋 나란히 비교
└── plot_guide.md             # 이 파일
```

---

## 각 플롯 설명

### 1. `heatmap_drift_x_af_f1.png` — F1 히트맵 (드리프트 탐지기 × 망각방지 전략)

| 항목 | 내용 |
|------|------|
| **X축** | Anti-Forgetting 전략 (`none` / `lwf_ssf` / `cfe` / `cndids` / `gpm`) |
| **Y축** | Drift Detector (`none` / `ssf` / `cade` / `ddm`) |
| **색상** | 평균 F1 점수 (진한 파란색 = 높은 F1 = 좋음) |
| **수치** | 각 셀 = 해당 조합의 모든 나머지 컴포넌트 평균 F1 |

**해석 방법**:
- 특정 행 전체가 밝다 → 해당 Drift Detector가 F1에 부정적 영향
- 특정 열 전체가 진하다 → 해당 Anti-Forgetting 전략이 탐지 성능에 유리
- 하나의 셀만 유독 진하다 → 해당 조합 쌍에 시너지 효과 존재

**IDS 문맥**: F1은 공격 탐지율(Recall)과 오탐율의 역수(Precision)의 조화평균.
NSL-KDD/UNSW-NB15처럼 클래스 불균형이 심한 데이터에서 단순 accuracy보다 신뢰할 수 있는 지표.

---

### 2. `heatmap_af_x_mm_bwt.png` — BWT 히트맵 (망각방지 전략 × 메모리 관리)

| 항목 | 내용 |
|------|------|
| **X축** | Memory Manager (`none` / `fixed` / `ssf` / `cndids`) |
| **Y축** | Anti-Forgetting 전략 |
| **색상** | 평균 BWT (진한 파란색 = 높은 BWT = 망각 적음) |

**해석 방법**:
- BWT = 마지막 태스크 학습 후 **이전 태스크들의 F1 변화량 평균**
  - BWT < 0 : 이전 공격 탐지 능력 저하 (catastrophic forgetting) → 빨간색 계열
  - BWT ≈ 0 : 이전 성능 유지 (이상적)
  - BWT > 0 : 새 학습이 이전 태스크도 개선 (backward plasticity) → 진한 파란색
- 특정 행(Anti-Forgetting)이 전반적으로 진하다 → 해당 전략이 망각 방지에 효과적

**IDS 문맥**: 새로운 공격 유형(e.g. 랜섬웨어)을 학습한 후 기존 공격(e.g. DDoS) 탐지 능력이
유지되는지를 측정한다. BWT가 크게 음수이면 모델 재학습마다 이전 공격에 무방비가 된다.

---

### 3. `heatmap_precision_recall.png` — Precision & Recall 2-패널 히트맵

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

### 4. `bwt_ranking.png` — BWT 상위 10개 조합 막대 그래프

| 항목 | 내용 |
|------|------|
| **X축** | BWT 값 (오른쪽 길수록 좋음) |
| **Y축** | 조합 라벨 (`drift/selector/memory/forgetting/scorer` 순) |
| **파란색** | BWT ≥ 0 (망각 없음 또는 개선) |
| **빨간색** | BWT < 0 (이전 성능 저하, 망각 발생) |

**해석 방법**:
- 상단에 위치한 조합이 이전 태스크 성능 보존에 가장 우수
- 대부분 빨간색이면 전반적으로 망각 문제가 심각함을 의미
- 검은 수직선(BWT=0)이 망각/보존의 기준점

---

### 5. `pareto.png` — Label Efficiency vs F1 Pareto Front

| 항목 | 내용 |
|------|------|
| **X축** | Label Efficiency = 1 - (레이블 수/전체 샘플 수). 높을수록 레이블 절약 |
| **Y축** | F1 Score |
| **회색 점** | 모든 384개 조합의 (efficiency, F1) 좌표 |
| **빨간선** | Pareto 최적 경계선 (비지배 해집합) |

**해석 방법**:
- Pareto 선 위/오른쪽 점: 같은 F1이면 더 적은 레이블 사용, 또는 같은 비용으로 더 높은 F1
- 실용적 배포 후보는 Pareto 선상의 점들
- 선이 오른쪽 위로 뻗어있을수록 레이블 효율과 탐지 성능이 모두 우수한 조합 존재

**IDS 문맥**: 실제 환경에서 레이블링은 보안 전문가의 수작업 분석을 요구한다.
레이블 예산이 제한된 상황에서 최적의 탐지 성능을 달성하는 조합 선택에 활용한다.

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
| **Recall (DR)** | TP/(TP+FN) | 1.0 | 공격 탐지율, 미탐(FN) 비용 |
| **FPR** | FP/(FP+TN) | 0.0 | 정상 트래픽 오탐률 |
| **Balanced Acc** | (TPR+TNR)/2 | 1.0 | 클래스 불균형 보정 정확도 |
| **BWT** | Σ(R[T][j]−R[j][j])/(T−1) | 0.0 | 연속 학습 망각 정도 |
| **FWT** | Σ(R[i−1][i]−R_rand)/(T−1) | 양수 | 사전 지식 전이 능력 |
| **Label Efficiency** | 1−labeled/total | 1.0 | 레이블링 비용 절감률 |
| **Avg Inference ms** | mean(t)×1000 | 최소 | 실시간 탐지 지연 |

- R[i][j] = i번째 태스크까지 훈련 후 j번째 태스크의 F1
- Balanced Acc = (TPR + TNR) / 2, 클래스 불균형이 심할 때 단순 accuracy 대체
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
    """Save bar chart of top_n combinations by BWT (legacy API).

    Args:
        results_dir: Directory containing summary.csv.
        top_n: Number of top combinations to show (default 10).
    """
    df = _load_results(results_dir)
    out = os.path.join(results_dir, 'plots', 'bwt_ranking.png')
    _bwt_ranking_from_df(df, top_n, out, title_suffix='ALL')


def plot_pareto(results_dir: str) -> None:
    """Save label_efficiency vs f1 scatter with Pareto front (legacy API).

    Args:
        results_dir: Directory containing summary.csv.
    """
    df = _load_results(results_dir)
    out = os.path.join(results_dir, 'plots', 'pareto.png')
    _pareto_from_df(df, out, title_suffix='ALL')


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

        # 2. BWT 히트맵: anti_forgetting × memory_manager
        _heatmap_from_df(
            df, 'memory_manager', 'anti_forgetting', 'bwt',
            os.path.join(out_dir, 'heatmap_af_x_mm_bwt.png'),
            title_suffix=ds_name,
        )

        # 3. Precision & Recall 2-패널 히트맵 (컬럼 존재 시)
        if 'precision' in df.columns and 'recall' in df.columns:
            _precision_recall_heatmap_from_df(
                df, 'anti_forgetting', 'drift_detector',
                os.path.join(out_dir, 'heatmap_precision_recall.png'),
                title_suffix=ds_name,
            )

        # 4. 개별 신규 지표 히트맵 (컬럼 존재 시)
        for metric in ['fpr', 'balanced_accuracy']:
            if metric in df.columns and df[metric].notna().any():
                _heatmap_from_df(
                    df, 'anti_forgetting', 'drift_detector', metric,
                    os.path.join(out_dir, f'heatmap_drift_x_af_{metric}.png'),
                    title_suffix=ds_name,
                )

        # 5. BWT 랭킹
        _bwt_ranking_from_df(
            df, 10,
            os.path.join(out_dir, 'bwt_ranking.png'),
            title_suffix=ds_name,
        )

        # 6. Pareto front
        _pareto_from_df(
            df,
            os.path.join(out_dir, 'pareto.png'),
            title_suffix=ds_name,
        )

    # 두 데이터셋 나란히 비교 플롯
    if has_ds_col and df_all['dataset'].nunique() > 1:
        _compare_datasets_plot(df_all, os.path.join(results_dir, 'plots'))

    # 플롯 가이드 생성
    ds_list = [ds for ds, _ in groups if ds != 'all']
    _generate_plot_guide(os.path.join(results_dir, 'plots'), ds_list)
