"""Dataset loaders for NSL-KDD and UNSW-NB15.

실제 데이터 경로:
  NSL-KDD  → ./SSF-Strategic-Selection-and-Forgetting/NSL_pre_data/
  UNSW-NB15 → ./SSF-Strategic-Selection-and-Forgetting/UNSW_pre_data/
"""

import os
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import torch


def load_nslkdd(data_dir: str) -> Dict:
    """Load NSL-KDD dataset (SSF 전처리 버전 우선 탐색).

    레이블 컬럼 'labels2': 'normal' → 0, 'anomaly' → 1
    특성 차원: 121 (모두 수치형, MinMaxScaler 정규화)

    Args:
        data_dir: PKDDTrain+.csv / PKDDTest+.csv 가 있는 디렉토리.
                  지정하지 않으면 SSF 기본 경로를 자동 탐색.

    Returns:
        {'X': FloatTensor(N,121), 'y': LongTensor(N,), 'scaler': MinMaxScaler}
    """
    candidates_train = ['PKDDTrain+.csv', 'KDDTrain+.csv', 'nsl_train.csv']
    candidates_test  = ['PKDDTest+.csv',  'KDDTest+.csv',  'nsl_test.csv']

    # 사용자 지정 경로 → SSF 기본 경로 순서로 탐색
    search_dirs = [data_dir,
                   'SSF-Strategic-Selection-and-Forgetting/NSL_pre_data',
                   './SSF-Strategic-Selection-and-Forgetting/NSL_pre_data']

    train_path = _find_file_multi(search_dirs, candidates_train)
    test_path  = _find_file_multi(search_dirs, candidates_test)

    if train_path is None or test_path is None:
        raise FileNotFoundError(
            "NSL-KDD 파일을 찾을 수 없습니다.\n"
            "다음 경로에 PKDDTrain+.csv / PKDDTest+.csv 를 두거나\n"
            "--data_dir 로 경로를 지정하세요:\n"
            "  SSF-Strategic-Selection-and-Forgetting/NSL_pre_data/")

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)
    df = pd.concat([df_train, df_test], ignore_index=True)

    # labels2: 'normal' / 'anomaly' (문자열)
    label_col = _find_label_col(df, ['labels2', 'label', 'class', 'Label'])
    y = (df[label_col].astype(str) != 'normal').astype(int).values

    # 레이블 컬럼 전부 제거 (labels2, labels5 등)
    drop_cols = [c for c in df.columns if 'label' in c.lower()]
    X_df = df.drop(columns=drop_cols, errors='ignore')

    # 범주형 → One-hot (있다면)
    cat_cols = X_df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols):
        X_df = pd.get_dummies(X_df, columns=cat_cols)

    X_np = X_df.values.astype(np.float32)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_np = scaler.fit_transform(X_np)

    return {
        'X': torch.FloatTensor(X_np),
        'y': torch.LongTensor(y),
        'scaler': scaler,
    }


def load_unswnb15(data_dir: str) -> Dict:
    """Load UNSW-NB15 dataset (SSF 전처리 버전 우선 탐색).

    레이블 컬럼 'label': 0=정상, 1=공격 (정수)
    특성 차원: 196 (모두 수치형, MinMaxScaler 정규화)

    Args:
        data_dir: UNSWTrain.csv / UNSWTest.csv 가 있는 디렉토리.

    Returns:
        {'X': FloatTensor(N,196), 'y': LongTensor(N,), 'scaler': MinMaxScaler}
    """
    candidates_train = ['UNSWTrain.csv', 'UNSW_Train.csv', 'unsw_train.csv']
    candidates_test  = ['UNSWTest.csv',  'UNSW_Test.csv',  'unsw_test.csv']

    search_dirs = [data_dir,
                   'SSF-Strategic-Selection-and-Forgetting/UNSW_pre_data',
                   './SSF-Strategic-Selection-and-Forgetting/UNSW_pre_data']

    train_path = _find_file_multi(search_dirs, candidates_train)
    test_path  = _find_file_multi(search_dirs, candidates_test)

    if train_path is None or test_path is None:
        raise FileNotFoundError(
            "UNSW-NB15 파일을 찾을 수 없습니다.\n"
            "다음 경로에 UNSWTrain.csv / UNSWTest.csv 를 두거나\n"
            "--data_dir 로 경로를 지정하세요:\n"
            "  SSF-Strategic-Selection-and-Forgetting/UNSW_pre_data/")

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)
    df = pd.concat([df_train, df_test], ignore_index=True)

    # label: 0 / 1 (정수 — 'normal'과 비교하면 안 됨)
    label_col = _find_label_col(df, ['label', 'Label', 'class'])
    y = df[label_col].astype(int).values

    X_df = df.drop(columns=[label_col], errors='ignore')

    cat_cols = X_df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols):
        X_df = pd.get_dummies(X_df, columns=cat_cols)

    X_np = X_df.values.astype(np.float32)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_np = scaler.fit_transform(X_np)

    return {
        'X': torch.FloatTensor(X_np),
        'y': torch.LongTensor(y),
        'scaler': scaler,
    }


def split_into_tasks(
        dataset: Dict,
        n_tasks: int,
        mode: str = 'temporal',
) -> List[Tuple[torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor]]:
    """데이터를 n_tasks개 연속학습 태스크로 분할.

    Args:
        dataset: load_nslkdd() / load_unswnb15() 반환값.
        n_tasks: 태스크 수.
        mode: 'temporal' — 순서대로 균등 분할.
              'attack_type' — 레이블 유형별 분할.

    Returns:
        [(X_train, y_train, X_test, y_test), ...] 길이 n_tasks.
        각 태스크는 80% 학습 / 20% 테스트 분할.
    """
    X = dataset['X']
    y = dataset['y']
    N = len(X)

    if mode == 'temporal':
        size = N // n_tasks
        tasks = []
        for i in range(n_tasks):
            start = i * size
            end   = (i + 1) * size if i < n_tasks - 1 else N
            Xi, yi = X[start:end], y[start:end]
            split = max(1, int(len(Xi) * 0.8))
            tasks.append((Xi[:split], yi[:split], Xi[split:], yi[split:]))
        return tasks

    elif mode == 'attack_type':
        classes = y.unique().tolist()
        if len(classes) < n_tasks:
            raise ValueError(
                f"공격 유형 수({len(classes)})가 태스크 수({n_tasks})보다 적습니다.")
        tasks = []
        for i in range(n_tasks):
            mask = (y == classes[i % len(classes)])
            Xi, yi = X[mask], y[mask]
            split = max(1, int(len(Xi) * 0.8))
            tasks.append((Xi[:split], yi[:split], Xi[split:], yi[split:]))
        return tasks

    else:
        raise ValueError(f"알 수 없는 mode: {mode!r}. 'temporal' 또는 'attack_type'.")


# ── Helpers ────────────────────────────────────────────────────────────────

def _find_file_multi(directories: List[str],
                     candidates: List[str]) -> Optional[str]:
    """여러 디렉토리를 순서대로 탐색하여 첫 번째 매칭 파일 반환."""
    for directory in directories:
        result = _find_file(directory, candidates)
        if result is not None:
            return result
    return None


def _find_file(directory: str, candidates: List[str]) -> Optional[str]:
    """directory를 재귀 탐색하여 candidates 중 첫 번째 매칭 파일 반환."""
    if not os.path.isdir(directory):
        return None
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname in candidates:
                return os.path.join(root, fname)
    return None


def _find_label_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"레이블 컬럼을 찾을 수 없음. 시도한 이름: {candidates}\n"
                   f"실제 컬럼: {list(df.columns)}")
