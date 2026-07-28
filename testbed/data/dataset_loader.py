"""NSL-KDD / UNSW-NB15 / CICIDS2018 데이터셋 로더 — PRD 9.1/9.2절.

원본 위치:
  NSL-KDD    : SSF-Strategic-Selection-and-Forgetting/NSL_pre_data/{PKDDTrain+.csv, PKDDTest+.csv}
               (SSF 저장소 전처리 버전)
  UNSW-NB15  : SSF-Strategic-Selection-and-Forgetting/UNSW_pre_data/{UNSWTrain.csv, UNSWTest.csv}
               (SSF 저장소 전처리 버전)
  CICIDS2018 : CICIDS2018/*.csv에 CSV를 직접 넣어두면 그걸 쓰고, 없으면
               공식 AWS Open Data 버킷(s3://cse-cic-ids2018/, 계정/자격증명
               불필요)의 "Processed Traffic Data for ML Algorithms/" 폴더에서
               10일치 CSV 전부(합계 약 6.9GB)를 자동으로 받아 그 폴더에
               저장한다(`_load_cicids2018_raw`/`_download_cicids2018_from_s3`
               참고). Kaggle 미러(`primus11/cic-ids-2018-dataset`)를 먼저
               시도했으나 실제로는 10일 중 하루치만, 그마저 잘려 있음을
               실측으로 확인해(아래 함수 docstring 참고) 공식 소스로
               교체했다(사용자 지시). 공식 train/test 분리 파일이 없어 이
               데이터셋만 preserve_official_split을 강제로 False로 둔다
               (아래 프로토콜 설명 참고).

라벨 극성: 0=정상, 1=공격으로 프로젝트 전체에서 통일한다(9.1절). 이 변환은
이 모듈에서 한 번만 수행한다.

이 로더는 두 가지 프로토콜을 지원한다(`preserve_official_split` 인자):

- **False** — PRD 9.1절 "병합 규칙": Train 파일 다음 Test 파일 순서로
  pd.concat한 뒤, 병합한 풀을 고정 seed로 한 번 섞고(원본 파일이 라벨/
  시간대/공격 유형 등으로 정렬돼 있으면 셔플 없이 그대로 등분했을 때
  experience별 라벨 분포가 완전히 깨질 수 있어 — CICIDS2018 실측으로 확인됨)
  n_experiences개로 균등 분할, 각 experience 내부에서 라벨 비율을 유지하는
  층화 무작위 80/20 train/test 분할을 수행한다. NSL-KDD/UNSW-NB15는 test가
  train과 같은 분포에서 나오므로 원 논문보다 쉬운 문제가 되어(더 이상
  기본으로 쓰지 않음), 공식 train/test 분리 파일이 없는 CICIDS2018 전용으로
  쓴다.

- **True(기본, 원 논문과 동일한 문제)** — SSF 원본(`ssf.py:104,155-159`)이
  실제로 하는 것과 동일하게, KDDTrain+/KDDTest+(UNSW도 동일)를 **합치지 않고**
  끝까지 분리해서 쓰되, 각 파일 내부는 **고정 seed로 한 번 무작위로 섞은 뒤**
  n_experiences개로 나눈다. SSF 원본도 `train_test_split(x_train, ...)`(기본
  shuffle=True)로 train을 섞고, `torch.randperm`으로 test도 섞은 뒤에야
  스트리밍 윈도우로 처리한다 — 원본 파일의 행 순서(예: UNSW-NB15가 라벨
  기준으로 사실상 정렬되어 있음, 실측 확인)를 그대로 쓰지 않는다. train↔test
  데이터가 섞이는 일은 없다(핵심 원칙 유지) — 각 파일 내부만 섞는다.
  MinMaxScaler도 train 파일에만 fit하고 test 파일은 transform만 한다(test
  통계가 정규화에 새어 들어가지 않도록). 이 방식을 NSL-KDD/UNSW-NB15 양쪽에
  동일하게 적용해 두 데이터셋이 같은 프로토콜을 쓰도록 통일했다(사용자 지시).

Experience 분할의 test split은 로딩 시점에 한 번만 확정되고 실험 내내 다시
나누지 않는다.
"""

import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def _read_nslkdd_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    y = (df["labels2"].astype(str) != "normal").astype(int).to_numpy()
    X_df = df.drop(columns=["labels2", "labels5"])
    if X_df.shape[1] != 121:
        raise ValueError(f"NSL-KDD expected 121 features, got {X_df.shape[1]}")
    return X_df.to_numpy(dtype=np.float64), y


def _read_unsw_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    y = df["label"].astype(int).to_numpy()
    X_df = df.drop(columns=["label"])
    if X_df.shape[1] != 196:
        raise ValueError(f"UNSW-NB15 expected 196 features, got {X_df.shape[1]}")
    return X_df.to_numpy(dtype=np.float64), y


def _load_nslkdd_raw(base_dir: str):
    train_path = os.path.join(
        base_dir, "SSF-Strategic-Selection-and-Forgetting", "NSL_pre_data", "PKDDTrain+.csv")
    test_path = os.path.join(
        base_dir, "SSF-Strategic-Selection-and-Forgetting", "NSL_pre_data", "PKDDTest+.csv")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    X_train, y_train = _read_nslkdd_features(df_train)
    X_test, y_test = _read_nslkdd_features(df_test)
    return X_train, y_train, X_test, y_test


def _read_cicids2018_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """CICIDS2018(CSE-CIC-IDS2018) CSV 관례에 맞춘 피처/라벨 추출.

    - 라벨: 'Label' 컬럼(대소문자/공백 무관하게 탐색), 'Benign'이 아니면
      전부 공격(1)으로 이진화한다(9.1절 라벨 극성: 0=정상, 1=공격).
    - Flow ID/IP/Timestamp 등 식별자 컬럼은 일반화 가능한 피처가 아니므로
      제외한다(모델이 특정 IP/시간대를 외우는 걸 방지 — 표준 관행).
    - CICIDS2018 원본 CSV에는 알려진 결측치/Infinity/헤더 중복 행 문제가
      있어(공개적으로 보고된 이슈), 숫자로 변환 안 되는 값과 inf는 NaN
      처리 후 해당 행을 통째로 제거한다.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None:
        raise ValueError(
            "CICIDS2018 CSV에서 'Label' 컬럼을 찾지 못했습니다. "
            f"실제 컬럼: {list(df.columns)[:10]}...")
    y = (df[label_col].astype(str).str.strip().str.lower() != "benign").astype(int).to_numpy()

    id_like = {"flow id", "src ip", "source ip", "dst ip", "destination ip", "timestamp"}
    drop_cols = [c for c in df.columns if c.lower() in id_like] + [label_col]
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    valid_mask = ~X_df.isna().any(axis=1)
    X_df = X_df[valid_mask]
    y = y[valid_mask.to_numpy()]

    # 10일치 전부를 이어붙이면 수천만 행 규모라(실측: 10일 합쳐 약 1600만
    # 행) float64는 메모리를 불필요하게 두 배 쓴다 — 어차피 CLClient가 최종
    # 단계에서 torch.float32로 변환하므로 float32로 바로 저장한다(NSL-KDD/
    # UNSW-NB15는 행 수가 훨씬 적어 float64를 유지, 이 함수만 다르게 적용).
    return X_df.to_numpy(dtype=np.float32), y


# 공식 배포처(AWS Open Data Registry, 계정/자격증명 없이 익명 접근 가능 —
# registry.opendata.aws/cse-cic-ids2018 참고). "Original Network Traffic and
# Log data/"는 원본 pcap(하루치가 수십 GB)이라 대상이 아니고, 여기 "Processed
# Traffic Data for ML Algorithms/"가 CICFlowMeter로 이미 피처 추출된 하루별
# CSV 10개다(합계 약 6.9GB, 직접 `list_objects_v2`로 실측 확인).
_CICIDS2018_S3_BUCKET = "cse-cic-ids2018"
_CICIDS2018_S3_PREFIX = "Processed Traffic Data for ML Algorithms/"


def _download_cicids2018_from_s3(dest_dir: str) -> List[str]:
    """`s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/`의
    CSV 10개를 전부 `dest_dir`로 받는다. 이미 같은 크기의 파일이 있으면
    다시 받지 않는다(중단 후 재실행 시 이어받기 아님 — 파일 단위 스킵).
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    os.makedirs(dest_dir, exist_ok=True)
    s3 = boto3.client("s3", region_name="ca-central-1",
                       config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")

    csv_paths = []
    for page in paginator.paginate(Bucket=_CICIDS2018_S3_BUCKET, Prefix=_CICIDS2018_S3_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.lower().endswith(".csv"):
                continue
            fname = os.path.basename(key)
            local_path = os.path.join(dest_dir, fname)
            if os.path.exists(local_path) and os.path.getsize(local_path) == obj["Size"]:
                print(f"CICIDS2018: {fname} 이미 있음 (스킵)")
            else:
                print(f"CICIDS2018 다운로드 중: {fname} ({obj['Size'] / 1e6:.0f}MB)...")
                s3.download_file(_CICIDS2018_S3_BUCKET, key, local_path)
            csv_paths.append(local_path)

    if not csv_paths:
        raise FileNotFoundError(
            f"s3://{_CICIDS2018_S3_BUCKET}/{_CICIDS2018_S3_PREFIX}에서 CSV를 "
            f"찾지 못했습니다 — 버킷 구조가 바뀌었을 수 있습니다.")
    return sorted(csv_paths)


def _load_cicids2018_raw(base_dir: str):
    """`<base_dir>/CICIDS2018/*.csv`에 수동으로 넣어둔 CSV가 있으면 그걸 쓰고,
    없으면 공식 AWS Open Data 버킷에서 10일치 전부를 자동으로 받아 그
    폴더에 저장한다(다음 실행부터는 이미 받은 파일로 인식해 재다운로드하지
    않는다).

    **경위(사용자 지시로 kagglehub 자동 다운로드에서 이걸로 교체)**:
    처음에는 kagglehub의 `primus11/cic-ids-2018-dataset`을 자동 다운로드
    소스로 썼는데, 실제로 받아서 대조해보니 공식 10일 중 **하루(2018-02-14,
    FTP/SSH 브루트포스일)뿐**이었고, 그 하루 공식 파일(341.6MB)보다도 작은
    76.4MB — 행 수가 정확히 1,048,575개로 엑셀 시트 최대 행 수(2^20)와
    일치해 추가로 잘려나간 것으로 보였다(라벨도 Benign/FTP-BruteForce/
    SSH-Bruteforce 3종류뿐, DoS/DDoS/Botnet/Web Attack/Infiltration 등
    나머지 공격 유형은 전혀 없음). "데이터를 전부 사용"하려는 목적에 안
    맞아 공식 소스로 교체했다.

    공식 train/test 분리 파일이 없는 데이터셋이라(원본 배포가 일(day)별
    캡처 CSV들로만 구성됨), load_dataset()이 이 데이터셋에 대해서는
    preserve_official_split을 강제로 False로 두고(병합+재셔플 프로토콜)
    처리한다 — 그래서 여기서는 test 쪽에 빈 배열을 반환해 4-튜플 인터페이스
    (`_load_nslkdd_raw`/`_load_unsw_raw`와 동일)만 맞춰준다.
    """
    manual_dir = os.path.join(base_dir, "CICIDS2018")
    csv_paths = sorted(glob.glob(os.path.join(manual_dir, "*.csv")))

    if not csv_paths:
        try:
            import boto3  # noqa: F401  (설치 여부 확인용)
        except ImportError as e:
            raise ImportError(
                "CICIDS2018 CSV가 로컬에 없고 boto3도 설치되어 있지 않습니다. "
                "`pip install boto3`로 설치해 공식 AWS 데이터를 자동으로 "
                f"받거나, CSV 파일을 직접 {manual_dir}에 넣어주세요.") from e
        csv_paths = _download_cicids2018_from_s3(manual_dir)

    # 공식 10일치 CSV가 전부 같은 컬럼 구성은 아니다 — 실측으로 확인됨:
    # Thuesday-20-02-2018 파일이 다른 9개보다 컬럼이 1개 더 많다(78 vs 79,
    # CICFlowMeter를 날짜마다 조금씩 다른 스크립트로 돌려서 생긴, 이 데이터셋
    # 자체에 알려진 스키마 불일치). 헤더만 먼저 읽어(nrows=0, 수 GB 파일도
    # 즉시 끝남) 전체 파일 공통 컬럼만 쓰도록 맞춘 뒤 본 로딩을 한다 — 어느
    # 파일이 "옳은" 스키마인지 임의로 정하지 않고, 공통으로 존재하는 컬럼만
    # 신뢰한다.
    header_sets = []
    for path in csv_paths:
        cols = {c.strip() for c in pd.read_csv(path, nrows=0).columns}
        header_sets.append((path, cols))
    common_cols = set.intersection(*(cols for _, cols in header_sets))
    if not any(c.lower() == "label" for c in common_cols):
        raise ValueError(
            "CICIDS2018 CSV 10개의 공통 컬럼에 'Label'이 없습니다 — 파일 "
            "구성이 예상보다 크게 다릅니다. 파일별 컬럼을 직접 확인해야 합니다.")
    for path, cols in header_sets:
        extra = cols - common_cols
        if extra:
            print(f"CICIDS2018: {os.path.basename(path)}에만 있는 컬럼 "
                  f"{sorted(extra)} - 다른 파일엔 없어 전체 공통 컬럼에서 제외.")

    X_parts, y_parts = [], []
    for path in csv_paths:
        print(f"CICIDS2018 로딩 중: {os.path.basename(path)}...")
        df = pd.read_csv(path, usecols=lambda c: c.strip() in common_cols, low_memory=False)
        X, y = _read_cicids2018_features(df)
        X_parts.append(X)
        y_parts.append(y)
    X_full = np.concatenate(X_parts, axis=0)
    y_full = np.concatenate(y_parts, axis=0)
    return X_full, y_full, X_full[:0], y_full[:0]


def _load_unsw_raw(base_dir: str):
    train_path = os.path.join(
        base_dir, "SSF-Strategic-Selection-and-Forgetting", "UNSW_pre_data", "UNSWTrain.csv")
    test_path = os.path.join(
        base_dir, "SSF-Strategic-Selection-and-Forgetting", "UNSW_pre_data", "UNSWTest.csv")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    X_train, y_train = _read_unsw_features(df_train)
    X_test, y_test = _read_unsw_features(df_test)
    return X_train, y_train, X_test, y_test


def _chunk_by_row_order(X: np.ndarray, y: np.ndarray, n_experiences: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    n = len(X)
    boundaries = np.linspace(0, n, n_experiences + 1).astype(int)
    return [(X[boundaries[i]:boundaries[i + 1]], y[boundaries[i]:boundaries[i + 1]])
            for i in range(n_experiences)]


def _shuffle(X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """SSF 원본(ssf.py:104의 train_test_split, ssf.py:155-159의 torch.randperm)과
    동일하게, 파일 내부를 고정 seed로 한 번 무작위로 섞는다."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def _chunk_shuffled(X: np.ndarray, y: np.ndarray, n_experiences: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    X_shuf, y_shuf = _shuffle(X, y, seed)
    return _chunk_by_row_order(X_shuf, y_shuf, n_experiences)


def load_dataset(name: str, base_dir: str, n_experiences: int = 5,
                  seed: int = 42, preserve_official_split: bool = True) -> Dict:
    """PRD 9.1/9.2절 로드. `preserve_official_split=True`면 모듈 docstring의
    "원 논문과 동일한 문제" 프로토콜을 쓴다.

    Args:
        name: 'nsl-kdd' | 'unsw-nb15' | 'cicids2018'
        base_dir: FCL 저장소 루트 경로(SSF 원본 데이터 폴더, CICIDS2018 폴더를
              포함하는 위치).
        n_experiences: 10.1절 기본값 5.
        seed: 10.1절 기본값 42 — preserve_official_split=False일 때 experience
              내부 stratified split에 사용.
        preserve_official_split: True면 원본 train/test 파일을 합치지 않고
              끝까지 분리해서 쓴다(모듈 docstring 참고). 'cicids2018'은 공식
              train/test 분리 파일이 없어 이 값과 무관하게 항상 False로
              강제된다(병합+재셔플 프로토콜만 적용 가능).

    Returns:
        {'input_dim': int, 'experiences': [{'train_X','train_y','test_X','test_y'}] * n_experiences}
    """
    if name == "nsl-kdd":
        X_train, y_train, X_test, y_test = _load_nslkdd_raw(base_dir)
    elif name == "unsw-nb15":
        X_train, y_train, X_test, y_test = _load_unsw_raw(base_dir)
    elif name == "cicids2018":
        X_train, y_train, X_test, y_test = _load_cicids2018_raw(base_dir)
        preserve_official_split = False
    else:
        raise ValueError(
            f"Unknown dataset: {name!r} (expected 'nsl-kdd', 'unsw-nb15', or 'cicids2018')")

    experiences = []

    if preserve_official_split:
        # train 파일에만 scaler를 fit — test 통계가 정규화에 새어 들어가지 않게.
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 각 파일 내부를 고정 seed로 섞은 뒤 나눈다(SSF 원본과 동일한 절차,
        # 모듈 docstring 참고). train/test는 서로 다른 seed로 섞어 SSF의
        # train_test_split과 torch.randperm이 독립적인 난수였던 것과 맞춘다.
        train_chunks = _chunk_shuffled(X_train, y_train, n_experiences, seed)
        test_chunks = _chunk_shuffled(X_test, y_test, n_experiences, seed + 1)

        for (X_tr, y_tr), (X_te, y_te) in zip(train_chunks, test_chunks):
            experiences.append({
                "train_X": torch.tensor(X_tr, dtype=torch.float32),
                "train_y": torch.tensor(y_tr, dtype=torch.long),
                "test_X": torch.tensor(X_te, dtype=torch.float32),
                "test_y": torch.tensor(y_te, dtype=torch.long),
            })
        input_dim = X_train.shape[1]
    else:
        df_X = np.concatenate([X_train, X_test], axis=0)
        df_y = np.concatenate([y_train, y_test], axis=0)

        scaler = MinMaxScaler()
        df_X = scaler.fit_transform(df_X)

        # 병합한 풀을 원본 행 순서 그대로 청크로 나누면, 원본 파일이 특정
        # 기준(라벨/시간대/공격 유형 등)으로 정렬돼 있을 때 experience별
        # 라벨 분포가 완전히 깨질 수 있다 — 실측으로 확인됨: CICIDS2018
        # 원본은 공격 유형별로 뭉쳐 있어서, 셔플 없이 그대로 5등분하면
        # 앞쪽 experience는 거의 100% 공격, 뒤쪽은 거의 100% 정상이 된다
        # (UNSW-NB15의 원본 파일 정렬 문제와 동일한 종류 — 9.1절 docstring
        # 참고). 그래서 청크로 나누기 전에 고정 seed로 한 번 섞는다.
        df_X, df_y = _shuffle(df_X, df_y, seed)

        chunks = _chunk_by_row_order(df_X, df_y, n_experiences)
        for X_exp, y_exp in chunks:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_exp, y_exp, test_size=0.2, stratify=y_exp, random_state=seed)
            experiences.append({
                "train_X": torch.tensor(X_tr, dtype=torch.float32),
                "train_y": torch.tensor(y_tr, dtype=torch.long),
                "test_X": torch.tensor(X_te, dtype=torch.float32),
                "test_y": torch.tensor(y_te, dtype=torch.long),
            })
        input_dim = df_X.shape[1]

    return {"input_dim": input_dim, "experiences": experiences}


def extract_normal_reference(experiences: List[Dict], size: int = 500,
                              seed: int = 42) -> torch.Tensor:
    """PRD 12.6절 — experience 0의 train split에서 label=0인 샘플을 고정
    크기로 한 번만 추출한다. 이 원본은 실험 내내 바뀌지 않는다(test split에서는
    뽑지 않는다 — test split은 평가 전용으로만 쓰여야 한다)."""
    train_X = experiences[0]["train_X"]
    train_y = experiences[0]["train_y"]
    normal_idx = (train_y == 0).nonzero(as_tuple=True)[0]
    g = torch.Generator().manual_seed(seed)
    perm = normal_idx[torch.randperm(len(normal_idx), generator=g)]
    k = min(size, len(perm))
    return train_X[perm[:k]]
