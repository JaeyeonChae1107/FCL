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
               (아래 프로토콜 설명 참고). 2026-07-30 재검토: CADE 원 논문의
               IDS2018 전처리(`CADE/IDS_data_preprocess/clean_data.py`,
               `gen_IDS_data.py`)를 근거로 완전 중복 행 제거, `Dst Port`
               빈도 기반 범주화+원핫, `Protocol` 원핫을 추가했다
               (`_dedup_rows`/`_bucket_port_frequency`/`_one_hot` 참고).
               중복 제거 후 남는 전체 데이터(약 1200만 행)를 서브샘플링 없이
               그대로 쓴다(같은 날 사용자 지시로 서브샘플링 단계를 제거함 —
               그리드 소요 시간이 훨씬 길어짐, docs/metric_justification.md
               참고).

라벨 극성: 0=정상, 1=공격으로 프로젝트 전체에서 통일한다(9.1절). 이 변환은
이 모듈에서 한 번만 수행한다.

이 로더는 두 가지 프로토콜을 지원한다(`preserve_official_split` 인자):

- **False** — PRD 9.1절 "병합 규칙": Train 파일 다음 Test 파일 순서로
  pd.concat한 뒤, 병합한 풀을 아래 class-incremental 분할로 나눈다. NSL-KDD/
  UNSW-NB15는 test가 train과 같은 분포에서 나오므로 원 논문보다 쉬운 문제가
  되어(더 이상 기본으로 쓰지 않음), 공식 train/test 분리 파일이 없는
  CICIDS2018 전용으로 쓴다.

- **True(기본, 원 논문과 동일한 문제)** — KDDTrain+/KDDTest+(UNSW도 동일)를
  **합치지 않고** 끝까지 분리해서 쓰되, 각 파일을 아래 class-incremental
  분할로 각각 나눈다. train↔test 데이터가 섞이는 일은 없다(핵심 원칙 유지).
  MinMaxScaler도 train 파일에만 fit하고 test 파일은 transform만 한다(test
  통계가 정규화에 새어 들어가지 않도록).

**2026-08-12 재검토 — experience 분할을 class-incremental 구조로 전환**:
기존에는 "고정 seed로 전체를 무작위로 섞은 뒤 n_experiences개로 균등분할"
했다(과거 `_chunk_shuffled`/`_chunk_by_row_order`) — 그 결과 experience
사이에 의도된 분포 차이가 전혀 없어(모두 같은 분포의 무작위 표본), 지속학습이
방어해야 할 진짜 "새로운 것이 등장"하는 상황 자체가 시나리오에 없었다.
`n_experiences=5`는 이미 CND-IDS 원 논문 근거로 채택돼 있었는데(아래 참고),
정작 그 숫자를 의미 있게 만드는 CND-IDS 원문의 실제 분할 메커니즘
(`CND-IDS/utils.py:275-299`, `create_split_experiences`)은 가져오지
않고 있었다 — 이제 그 메커니즘을 그대로 이식한다(`_class_incremental_split`
참고): 정상 트래픽은 무작위로 고르게 5개 experience에 나누고, 공격은
세부 카테고리별로 묶어 라운드로빈으로 experience에 배정해 **한 experience는
자신에게 배정된 공격 유형만** 보게 된다. SSF 원 논문 방식(가변 길이 스트리밍
+ drift-조건부 로직)은 검토 후 기각했다 — SSF는 실제로는 drift 감지 여부와
무관하게 매 라운드 재학습하고(코드 재확인으로 확정), 라운드 수가 데이터
크기에 따라 달라져 이 테스트베드의 고정 n_experiences 구조와 안 맞으며,
SSF 고유의 "대표 표본 선택+망각" 알고리즘을 공유 시나리오로 채택하면 SSF
방법론을 전체 그리드에 강제하는 셈이 되어 0절에 어긋난다. class_order
방식은 데이터를 "언제 등장시킬지"의 시나리오 설계일 뿐이라 96개 조합
전부에 공평하다.

Experience 분할의 test split은 로딩 시점에 한 번만 확정되고 실험 내내 다시
나누지 않는다.
"""

import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def _read_nslkdd_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = (df["labels2"].astype(str) != "normal").astype(int).to_numpy()
    # class-incremental 분할용 세부 카테고리(정상/DoS/Probe/R2L/U2R) — 예전엔
    # 그냥 버려졌다(2026-08-12부터 보존, 아래 _class_incremental_split 참고).
    # 정상(y=0) 행의 값은 분할 알고리즘이 아예 안 쓰므로 상관없다.
    category = df["labels5"].astype(str).to_numpy()
    X_df = df.drop(columns=["labels2", "labels5"])
    if X_df.shape[1] != 121:
        raise ValueError(f"NSL-KDD expected 121 features, got {X_df.shape[1]}")
    return X_df.to_numpy(dtype=np.float64), y, category


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
    X_train, y_train, category_train = _read_nslkdd_features(df_train)
    X_test, y_test, category_test = _read_nslkdd_features(df_test)
    return X_train, y_train, X_test, y_test, category_train, category_test


def _read_cicids2018_features(
        df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CICIDS2018(CSE-CIC-IDS2018) CSV 관례에 맞춘 피처/라벨 추출.

    - 라벨: 'Label' 컬럼(대소문자/공백 무관하게 탐색), 'Benign'이 아니면
      전부 공격(1)으로 이진화한다(9.1절 라벨 극성: 0=정상, 1=공격). 이진화
      전의 원본 문자열(예: 'DDoS attacks-LOIC-HTTP', 'Infilteration')도
      class-incremental 분할용 카테고리로 함께 보존한다(2026-08-12 추가 —
      예전엔 이진화만 하고 원본 문자열은 버려졌다).
    - Flow ID/IP/Timestamp 등 식별자 컬럼은 일반화 가능한 피처가 아니므로
      제외한다(모델이 특정 IP/시간대를 외우는 걸 방지 — 표준 관행).
    - 'Dst Port'/'Protocol'은 숫자로 보이지만 실제로는 범주형이라(포트
      번호에 순서 의미가 없음) 나머지 연속형 피처와 분리해 반환한다 — CADE
      원 논문의 IDS2018 전처리(`CADE/IDS_data_preprocess/gen_IDS_data.py:
      184-215`, Dst Port 빈도 기반 범주화+원핫, Protocol 원핫)와 같은 이유다.
      호출부(`_load_cicids2018_raw`)에서 병합된 전체 풀 기준으로 인코딩한다.
    - CICIDS2018 원본 CSV에는 알려진 결측치/Infinity/헤더 중복 행 문제가
      있어(공개적으로 보고된 이슈), 숫자로 변환 안 되는 값과 inf는 NaN
      처리 후 해당 행을 통째로 제거한다.

    Returns:
        (X_numeric, port_raw, protocol_raw, y, category) — 전부 같은 행 순서/길이.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None:
        raise ValueError(
            "CICIDS2018 CSV에서 'Label' 컬럼을 찾지 못했습니다. "
            f"실제 컬럼: {list(df.columns)[:10]}...")
    label_str = df[label_col].astype(str).str.strip()
    y = (label_str.str.lower() != "benign").astype(int).to_numpy()
    category = label_str.to_numpy()

    port_col = next((c for c in df.columns if c.lower() == "dst port"), None)
    protocol_col = next((c for c in df.columns if c.lower() == "protocol"), None)

    id_like = {"flow id", "src ip", "source ip", "dst ip", "destination ip", "timestamp"}
    drop_cols = [c for c in df.columns if c.lower() in id_like] + [label_col]
    if port_col:
        drop_cols.append(port_col)
    if protocol_col:
        drop_cols.append(protocol_col)
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    valid_mask = (~X_df.isna().any(axis=1)).to_numpy()

    port_raw = pd.to_numeric(df[port_col], errors="coerce").fillna(-1).to_numpy() \
        if port_col else np.full(len(df), -1.0)
    protocol_raw = pd.to_numeric(df[protocol_col], errors="coerce").fillna(-1).to_numpy() \
        if protocol_col else np.full(len(df), -1.0)

    X_df = X_df[valid_mask]
    y = y[valid_mask]
    category = category[valid_mask]
    port_raw = port_raw[valid_mask]
    protocol_raw = protocol_raw[valid_mask]

    # 10일치 전부를 이어붙이면 수천만 행 규모라(실측: 10일 합쳐 약 1600만
    # 행) float64는 메모리를 불필요하게 두 배 쓴다 — 어차피 CLClient가 최종
    # 단계에서 torch.float32로 변환하므로 float32로 바로 저장한다(NSL-KDD/
    # UNSW-NB15는 행 수가 훨씬 적어 float64를 유지, 이 함수만 다르게 적용).
    return X_df.to_numpy(dtype=np.float32), port_raw, protocol_raw, y, category


def _dedup_rows(X: np.ndarray, port: np.ndarray, protocol: np.ndarray, y: np.ndarray,
                 category: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """완전히 동일한 트래픽 행(피처+포트+프로토콜+라벨 전부 일치)을 제거한다.

    CADE 원 논문의 IDS2018 전처리(`CADE/IDS_data_preprocess/clean_data.py:8,
    99` — "Duplicate traffic was removed")와 동일한 절차. 파일(하루) 단위로
    적용한다 — 날짜가 다른 두 흐름이 우연히 완전히 같은 값일 가능성은
    희박하고, 하루치(~160만 행) 단위로 처리하는 게 전체 병합 후 처리보다
    훨씬 가볍다. 중복 판정 기준(피처+포트+프로토콜+라벨)은 그대로 두고
    `category`는 같은 `unique_idx`로 슬라이싱만 한다(y와 1:1 대응이라
    판정 기준에 넣을 필요 없음).
    """
    combined = np.concatenate(
        [X, port.reshape(-1, 1), protocol.reshape(-1, 1), y.reshape(-1, 1).astype(np.float32)],
        axis=1)
    _, unique_idx = np.unique(combined, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    return (X[unique_idx], port[unique_idx], protocol[unique_idx], y[unique_idx],
            category[unique_idx])


class _IncrementalPortBucketer:
    """`_bucket_port_frequency`(등장 빈도 상/중/하 3단계 범주화, CADE
    `gen_IDS_data.py:184-201` 근거)를 라운드 누적 방식으로 재현한다.

    2026-08-26 발견·수정 — 기존엔 이 범주화를 10일치 전체를 병합한
    `port_full`(=모든 experience를 합친 것) 기준으로 **한 번** 계산해
    썼다. MinMaxScaler를 train 파일 전체에 한 번 fit하던 것과 정확히
    같은 종류의 문제다(위 `load_dataset()` "2026-08-26" 절 참고) —
    experience 0의 모델이 실제로는 아직 등장하지 않은 미래 experience의
    포트 빈도 분포까지 이미 알고 있는 셈이었다. 이 클래스는 `bucket()`이
    호출될 때마다(=experience 하나씩) 그 experience의 포트 등장 횟수를
    누적하고, **그 시점까지의 누적 빈도**로 분위수(상위 1%/10%)를 계산해
    그 experience의 버킷을 매긴다 — experience i는 항상 0..i의 포트
    빈도만 알고 그 이후는 전혀 모른다.
    """

    def __init__(self):
        self._counts = pd.Series(dtype=np.int64)

    def update(self, port: np.ndarray) -> None:
        """이번 라운드의 **train** 포트만 누적 카운트에 반영한다."""
        new_counts = pd.Series(port).value_counts()
        self._counts = self._counts.add(new_counts, fill_value=0)

    def transform(self, port: np.ndarray) -> np.ndarray:
        """현재까지 누적된 카운트(=`update()`가 반영한 train 데이터까지)로
        버킷을 매기기만 한다 — 카운트를 바꾸지 않는다(sklearn의
        fit/transform 분리와 같은 원칙). 누적 카운트에 전혀 없던 포트값
        (예: 이 라운드 test에만 등장)은 빈도 0으로 취급해 최하위 버킷으로
        떨어진다."""
        high_cut = self._counts.quantile(0.99)
        med_cut = self._counts.quantile(0.90)
        freq = pd.Series(port).map(self._counts).fillna(0.0).to_numpy()
        return np.where(freq >= high_cut, 0, np.where(freq >= med_cut, 1, 2))

    def bucket(self, port: np.ndarray) -> np.ndarray:
        """`update()` 후 바로 `transform()` — 편의 메서드(회귀 테스트 등
        train/test 구분이 필요 없는 호출부용). **2026-08-26 발견·수정 —
        같은 라운드 안에서도 test 쪽이 누출됨**: `load_dataset()`이 라운드
        전체(train+test 예정 데이터)를 이 메서드 하나로 처리한 뒤에야
        train_test_split을 했었다 — 그 라운드의 test 쪽 포트 빈도까지
        그 라운드 자신의 train 버킷 경계 계산에 섞여 들어간 것이다(라운드를
        넘나드는 누출과는 별개의, 같은 라운드 내부의 누출). 이제
        `load_dataset()`은 먼저 train/test로 나눈 뒤 train에만 `update()`,
        train/test 양쪽에 `transform()`을 쓴다."""
        self.update(port)
        return self.transform(port)


def _one_hot(values: np.ndarray, n_categories_cap: int = 16,
             categories: Optional[np.ndarray] = None) -> np.ndarray:
    """작은 카디널리티의 정수 범주형 배열을 원-핫으로 인코딩한다.

    Protocol처럼 실제 고유값이 몇 개뿐인 경우를 위한 것이라, 예상외로
    카디널리티가 크면(스키마 문제 등) 조용히 잘못된 거대 행렬을 만드는
    대신 바로 에러를 낸다.

    Args:
        categories: 명시적으로 주어지면 `values`에서 유도(`np.unique`)하지
            않고 이 목록/순서를 그대로 쓴다. 2026-08-26 추가 — CICIDS2018을
            experience 단위로 나눠 처리할 때(아래 `load_dataset()` 참고)
            매 experience가 우연히 일부 범주값만 포함하면 원-핫 폭이
            experience마다 달라져 공유 모델 입력 차원이 깨진다. port
            bucket(항상 {0,1,2})처럼 범주 **값 자체의 정의역**이 데이터
            관측과 무관하게 고정된 경우, 또는 protocol처럼 "이 필드가 어떤
            값들을 가질 수 있는가"라는 스키마 성격의 정보(어떤 특정 라운드의
            통계적 분포가 아니라 프로토콜 표준 자체가 정하는 값 집합)는
            전체 데이터에서 한 번 계산해 고정폭으로 재사용해도 "미래 정보
            누출"이 아니다 — `_bucket_port_frequency`의 등장 빈도 같은
            통계량과는 성격이 다르다(그건 실제로 라운드 순차 누적으로
            바꿨다, `_IncrementalPortBucketer` 참고).
    """
    if categories is None:
        categories = np.unique(values)
    if len(categories) > n_categories_cap:
        raise ValueError(
            f"원-핫 인코딩 대상 범주 수({len(categories)})가 예상보다 많습니다 "
            f"({n_categories_cap} 초과) — 실제로 범주형이 맞는지 확인이 필요합니다.")
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    onehot = np.zeros((len(values), len(categories)), dtype=np.float32)
    idx = np.array([cat_to_idx[v] for v in values])
    onehot[np.arange(len(values)), idx] = 1.0
    return onehot


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


def _load_cicids2018_raw(base_dir: str, seed: int = 42):
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

    **2026-08-26 수정 — port/protocol 원-핫 인코딩을 여기서 하지 않는다**:
    예전엔 이 함수 안에서 10일치 전체(=모든 experience를 합친 병합 풀)
    기준으로 `_bucket_port_frequency`를 한 번 계산해 X에 이미 붙여
    반환했다. `load_dataset()`의 MinMaxScaler 수정(같은 파일의
    "2026-08-26" 절 참고)과 같은 이유로, 이러면 experience 0의 모델이
    아직 등장하지 않은 미래 experience의 포트 빈도 분포까지 알게 된다.
    그래서 port/protocol은 원본 그대로(`port_full`/`protocol_full`)
    반환하고, class-incremental 분할 이후 experience 순서대로
    `_IncrementalPortBucketer`로 처리하는 건 `load_dataset()`이 한다
    (반환 튜플이 6개로 늘어난다 — `_load_nslkdd_raw`/`_load_unsw_raw`와
    는 이제 다른 시그니처).
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

    X_parts, port_parts, protocol_parts, y_parts, category_parts = [], [], [], [], []
    for path in csv_paths:
        print(f"CICIDS2018 로딩 중: {os.path.basename(path)}...")
        df = pd.read_csv(path, usecols=lambda c: c.strip() in common_cols, low_memory=False)
        X, port, protocol, y, category = _read_cicids2018_features(df)
        n_before = len(X)
        X, port, protocol, y, category = _dedup_rows(X, port, protocol, y, category)
        if len(X) != n_before:
            print(f"CICIDS2018: {os.path.basename(path)} 중복 행 "
                  f"{n_before - len(X)}개 제거 ({n_before} -> {len(X)})")
        X_parts.append(X)
        port_parts.append(port)
        protocol_parts.append(protocol)
        y_parts.append(y)
        category_parts.append(category)
    X_full = np.concatenate(X_parts, axis=0)
    port_full = np.concatenate(port_parts, axis=0)
    protocol_full = np.concatenate(protocol_parts, axis=0)
    y_full = np.concatenate(y_parts, axis=0)
    category_full = np.concatenate(category_parts, axis=0)
    # 2026-07-30: 서브샘플링을 없애고 중복 제거 후 전체(약 1200만 행)를
    # 그대로 쓴다 — normal_reference를 매 라운드 selected_data에서 직접
    # 뽑는 방식으로 바뀌면서(cl_client.py 참고) 대표성 문제가 없어졌고,
    # 사용자가 CICIDS2018을 "더 크고 다양한 3번째 데이터셋"이라는 애초
    # 취지대로 축소 없이 쓰기로 결정했다. 그리드/스모크 테스트 소요 시간이
    # NSL-KDD/UNSW-NB15보다 훨씬 길어진다(experience당 행 수가 80배 이상
    # 많음 — Track A 조합 기준 수십 분/조합, 그리드 전체는 며칠 단위가 될
    # 수 있음, 실측 근거는 docs/metric_justification.md 참고).

    return (X_full, y_full, X_full[:0], y_full[:0], category_full, category_full[:0],
            port_full, protocol_full)


def _load_unsw_attack_cat(base_dir: str, y_train: np.ndarray,
                            y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """class-incremental 분할용 UNSW-NB15 다중클래스 `attack_cat`을 공식
    원본에서 가져온다.

    지금 쓰는 SSF 전처리본(`UNSW_pre_data/UNSWTrain.csv`/`UNSWTest.csv`)엔
    이진 `label`만 있고 `attack_cat`이 없다 — SSF가 자체 전처리(원-핫,
    MinMax 스케일링 등) 과정에서 뺀 것으로 보인다. 그런데 공식
    `UNSW_NB15_training-set.csv`(175,341행)/`UNSW_NB15_testing-set.csv`
    (82,332행)의 행 수가 SSF 전처리본과 정확히 일치함을 실측 확인했다
    (docs/metric_justification.md 참고) — 같은 원본에서 나온 것으로 보고,
    250만 행짜리 원본을 처음부터 재전처리하는 대신 공식 파일의 `attack_cat`
    컬럼만 같은 행 위치에서 가져와 결합한다.

    **행 정렬은 절대 그냥 가정하지 않는다** — 공식 파일의 `label`이 SSF
    전처리본의 `label`(=y_train/y_test)과 행별로 완전히 일치하는지 확인한
    뒤에만 attack_cat을 신뢰한다. 하나라도 다르면 조용히 진행하지 않고
    바로 예외를 던진다.
    """
    manual_dir = os.path.join(base_dir, "UNSW-NB15-raw")
    train_path = os.path.join(manual_dir, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(manual_dir, "UNSW_NB15_testing-set.csv")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            "UNSW-NB15 공식 원본(attack_cat 포함)이 로컬에 없습니다. "
            "https://research.unsw.edu.au/projects/unsw-nb15-dataset 에서 "
            "UNSW_NB15_training-set.csv 와 UNSW_NB15_testing-set.csv를 받아 "
            f"{manual_dir} 에 넣어주세요(SharePoint 호스팅이라 CICIDS2018과 "
            "달리 자동 다운로드는 지원하지 않습니다).")

    def _verify_and_extract(csv_path: str, y_expected: np.ndarray, tag: str) -> np.ndarray:
        official = pd.read_csv(csv_path)
        if len(official) != len(y_expected):
            raise ValueError(
                f"UNSW-NB15 공식 {tag} 행 수({len(official)})가 SSF 전처리본"
                f"({len(y_expected)})과 다릅니다 — 같은 원본이 아닌 것으로 "
                "보여 attack_cat을 안전하게 결합할 수 없습니다.")
        official_label = official["label"].astype(int).to_numpy()
        if not np.array_equal(official_label, y_expected):
            raise ValueError(
                f"UNSW-NB15 공식 {tag}의 label 컬럼이 SSF 전처리본과 행별로 "
                "일치하지 않습니다 — 행 순서가 다른 것으로 보여 attack_cat을 "
                "안전하게 결합할 수 없습니다(추측으로 진행하지 않음).")
        # 정상 행은 attack_cat이 비어있는 게 원본 관례 — class-incremental
        # 분할 알고리즘은 정상(y=0) 행의 category 값을 아예 안 쓰므로 어떤
        # 문자열이든 상관없다("normal"로만 채워둔다).
        return official["attack_cat"].fillna("normal").astype(str).str.strip().to_numpy()

    category_train = _verify_and_extract(train_path, y_train, "training-set.csv")
    category_test = _verify_and_extract(test_path, y_test, "testing-set.csv")
    return category_train, category_test


def _load_unsw_raw(base_dir: str):
    train_path = os.path.join(
        base_dir, "SSF-Strategic-Selection-and-Forgetting", "UNSW_pre_data", "UNSWTrain.csv")
    test_path = os.path.join(
        base_dir, "SSF-Strategic-Selection-and-Forgetting", "UNSW_pre_data", "UNSWTest.csv")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    X_train, y_train = _read_unsw_features(df_train)
    X_test, y_test = _read_unsw_features(df_test)
    category_train, category_test = _load_unsw_attack_cat(base_dir, y_train, y_test)
    return X_train, y_train, X_test, y_test, category_train, category_test


def _class_incremental_split(
        X: np.ndarray, y: np.ndarray, category: np.ndarray, n_experiences: int, seed: int,
        class_order: Optional[List[List[str]]] = None,
        extra_arrays: Optional[List[np.ndarray]] = None
        ) -> Tuple[List[Tuple[np.ndarray, ...]], List[List[str]]]:
    """CND-IDS 원 논문의 실제 experience 분할 메커니즘 이식
    (`CND-IDS/utils.py:275-299`, `create_split_experiences`).

    2026-08-12 도입 — 예전엔 정상/공격 구분 없이 전체를 무작위로 섞어 그냥
    n_experiences개로 균등분할했다(`_chunk_shuffled`, 이 함수가 대체함) —
    그 결과 experience끼리 분포 차이가 전혀 없어 지속학습이 방어할 진짜
    분포 변화가 시나리오에 없었다(docs/metric_justification.md 참고).

    - 정상(y=0) 행: 고정 seed로 셔플 후 n_experiences개에 고르게 분배(예전과
      동일한 부분) — `category` 값은 무시한다(정상 행의 category가 무엇이든
      상관없음).
    - 공격(y=1) 행: `category`별로 묶어 라운드로빈으로 experience에 배정
      (`class_order[i % n_experiences].append(category)`, category는 정렬된
      순서로 순회 — 임의 순서를 만들지 않는다). 한 experience는 자신에게
      배정된 category의 공격 행만 받는다.
    - 최종적으로 각 experience = (배정된 공격 행) ∪ (해당 인덱스의 정상
      청크), 다시 한 번 셔플해 정상/공격을 섞는다(CND-IDS 원문의 마지막
      `shuffle_idx` 단계와 동일).

    Args:
        class_order: 이미 계산된 category -> experience 배정을 재사용하려면
            전달한다(예: train에서 계산한 걸 test 분할에도 똑같이 써야
            experience i의 test가 experience i의 train과 같은 category를
            반영한다 — CND-IDS 원문도 `train_experiences`/`test_experiences`
            양쪽에 같은 `class_order`를 넘긴다). None이면 이 호출에서 새로
            계산해서 두 번째 반환값으로 돌려준다.

    Args (추가):
        extra_arrays: X/y/category와 같은 행 순서/길이인 추가 배열들(예:
            CICIDS2018의 원본 port/protocol — 아래 `_load_cicids2018_raw`
            참고). 주어지면 각 experience 튜플 끝에 같은 `exp_idx`로 슬라이싱한
            값이 순서대로 덧붙는다 — X/y/category만 쓰는 기존 호출부(NSL-KDD/
            UNSW-NB15)는 이 인자를 안 넘기므로 반환 튜플 길이가 그대로 3
            (X_exp, y_exp, category_exp)이라 영향이 없다.

    Returns:
        (experiences, class_order) — experiences는 [(X_exp, y_exp, category_exp,
        *extra_exp)] * n_experiences, class_order는 재사용/기록용으로 반환.
        category_exp는 CADE의 원 논문 방식(정상 + 공격 family별 centroid,
        `min()`으로 이상 판정)을 다중클래스로 재현하는 데 쓴다
        (`components/cade/` 참고) — y_exp(이진)만으로는 CADE의 family 단위
        구조를 표현할 수 없다.
    """
    category = np.asarray(category)
    rng = np.random.RandomState(seed)

    normal_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]

    normal_idx = normal_idx[rng.permutation(len(normal_idx))]
    normal_chunks = np.array_split(normal_idx, n_experiences)

    if class_order is None:
        attack_categories = sorted(set(category[attack_idx].tolist()))
        class_order = [[] for _ in range(n_experiences)]
        for i, cat in enumerate(attack_categories):
            class_order[i % n_experiences].append(cat)
    else:
        # 2026-08-14 추가(구조 전수 감사에서 발견) — class_order는 train에서
        # 계산해 test 분할에 재사용된다(위 docstring 참고). test에만 존재하고
        # train에는 없던 공격 category가 있으면, 그 category는 class_order의
        # 어느 experience 목록에도 없어 아래 96번째 줄의 np.isin이 항상
        # False를 반환한다 — 즉 어떤 experience에도 배정되지 못하고 에러도
        # 경고도 없이 통째로 사라진다. 이 코드베이스의 다른 곳들(예:
        # _load_unsw_attack_cat의 행 정렬 검증)과 같은 원칙 — 추측으로
        # 조용히 진행하지 않고 즉시 예외를 던진다.
        covered = {cat for group in class_order for cat in group}
        present = set(category[attack_idx].tolist())
        uncovered = present - covered
        if uncovered:
            raise ValueError(
                f"_class_incremental_split: class_order가 다루지 않는 공격 "
                f"category가 있습니다: {sorted(uncovered)} — 이 데이터는 "
                "train에서 계산한 class_order를 test에 재사용할 때 train에 "
                "없던 category가 test에만 존재한다는 뜻입니다. 이대로 두면 "
                "해당 표본이 어떤 experience에도 배정되지 못하고 조용히 "
                "누락됩니다.")

    experiences = []
    for i in range(n_experiences):
        cat_mask = np.isin(category[attack_idx], class_order[i])
        exp_attack_idx = attack_idx[cat_mask]
        exp_idx = np.concatenate([exp_attack_idx, normal_chunks[i]])
        exp_idx = exp_idx[rng.permutation(len(exp_idx))]
        extras = tuple(arr[exp_idx] for arr in extra_arrays) if extra_arrays else ()
        experiences.append((X[exp_idx], y[exp_idx], category[exp_idx]) + extras)

    return experiences, class_order


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
        {'input_dim': int, 'experiences': [{'train_X','train_y','test_X','test_y',
        'train_category'}] * n_experiences}. train_category는 numpy 문자열
        배열(정상="normal"/"Benign"/데이터셋별 표기, 공격은 family 이름) —
        train_y와 같은 행 순서, 길이만 대응. CADE의 다중클래스 centroid
        구성에만 쓰이고(선택적 소비, `pipeline/cl_client.py` 참고) 다른
        컴포넌트는 이 키를 몰라도 된다.
    """
    port_full = protocol_full = None
    if name == "nsl-kdd":
        X_train, y_train, X_test, y_test, category_train, category_test = _load_nslkdd_raw(base_dir)
    elif name == "unsw-nb15":
        X_train, y_train, X_test, y_test, category_train, category_test = _load_unsw_raw(base_dir)
    elif name == "cicids2018":
        (X_train, y_train, X_test, y_test, category_train, category_test,
         port_full, protocol_full) = _load_cicids2018_raw(base_dir, seed=seed)
        preserve_official_split = False
    else:
        raise ValueError(
            f"Unknown dataset: {name!r} (expected 'nsl-kdd', 'unsw-nb15', or 'cicids2018')")

    experiences = []

    if preserve_official_split:
        # class_order는 train에서 계산해 test 분할에 그대로 재사용한다 —
        # experience i의 test가 experience i의 train과 같은 공격 category를
        # 반영하도록(모듈 docstring "2026-08-12 재검토" 참고, CND-IDS 원문도
        # train/test 양쪽에 같은 class_order를 쓴다). train/test는 서로
        # 다른 seed로 셔플해 두 파일이 독립적인 난수를 쓰도록 한다.
        # **먼저(원본=미정규화) 데이터로 분할한 뒤 scaler를 적용한다** — 아래
        # "2026-08-26" 절 참고.
        train_chunks, class_order = _class_incremental_split(
            X_train, y_train, category_train, n_experiences, seed)
        test_chunks, _ = _class_incremental_split(
            X_test, y_test, category_test, n_experiences, seed + 1, class_order=class_order)

        # 2026-08-26 발견·수정 — MinMaxScaler가 지금까지 "train 파일 전체"에
        # 한 번에 fit됐다(정규화 자체는 train에만 fit해 test 누출은 막았지만,
        # train 파일 안에서도 5개 experience로 나뉘는데 scaler는 그 분할
        # 전체를 한꺼번에 봤다) — 즉 experience 0의 모델이 실제로는 아직
        # 등장하지 않은 experience 4의 데이터 범위(min/max)까지 이미 알고
        # 있는 셈이었다. 지속학습 벤치마크가 지켜야 할 "미래 정보 없음"
        # 원칙에 어긋난다(4개 논문 컴포넌트 전수 재감사와는 별개로, 이
        # 재검토 과정에서 재확인). `partial_fit()`으로 라운드마다 그 라운드
        # train 데이터만큼만 누적 갱신하고, 그 시점까지의 scaler로 해당
        # 라운드의 train/test를 변환한다 — experience i는 항상 experience
        # 0..i의 통계만 알고, i+1 이후는 전혀 모른다(실제 스트리밍 배포와
        # 동일한 제약). 원래 취지(test 통계가 정규화에 안 새어 들어가는 것)
        # 는 그대로 유지된다 — test는 fit에 전혀 참여하지 않는다.
        scaler = MinMaxScaler()
        for (X_tr, y_tr, cat_tr), (X_te, y_te, cat_te) in zip(train_chunks, test_chunks):
            scaler.partial_fit(X_tr)
            X_tr_scaled = scaler.transform(X_tr)
            X_te_scaled = scaler.transform(X_te)
            experiences.append({
                "train_X": torch.tensor(X_tr_scaled, dtype=torch.float32),
                "train_y": torch.tensor(y_tr, dtype=torch.long),
                "test_X": torch.tensor(X_te_scaled, dtype=torch.float32),
                "test_y": torch.tensor(y_te, dtype=torch.long),
                "train_category": cat_tr,
            })
        input_dim = X_train.shape[1]
    else:
        df_X = np.concatenate([X_train, X_test], axis=0)
        df_y = np.concatenate([y_train, y_test], axis=0)
        df_category = np.concatenate([category_train, category_test], axis=0)

        # 2026-08-26 수정 — 위와 같은 이유로, 정규화 전(원본) 데이터로 먼저
        # class-incremental 분할을 한 뒤, experience 순서대로 scaler를
        # 누적(`partial_fit`) 적용한다. CICIDS2018(preserve_official_split
        # 강제 False)의 port 빈도 범주화도 같은 원칙으로 라운드 누적 처리한다
        # (`_IncrementalPortBucketer` 참고) — protocol 원-핫의 범주 **집합**
        # 자체는 전체 데이터에서 한 번 고정하는데, 이건 통계가 아니라 스키마
        # 정보라 미래 누출이 아니다(`_one_hot()`의 "2026-08-26" 절 참고).
        port_bucketer = None
        protocol_categories = None
        extra_arrays = None
        if port_full is not None:
            port_bucketer = _IncrementalPortBucketer()
            protocol_categories = np.unique(protocol_full)
            extra_arrays = [port_full, protocol_full]

        exp_chunks, _ = _class_incremental_split(
            df_X, df_y, df_category, n_experiences, seed, extra_arrays=extra_arrays)
        scaler = MinMaxScaler()
        input_dim = None
        for chunk in exp_chunks:
            # 2026-08-26 발견·수정 — 라운드를 넘나드는 누출(위 절)과는 별개로,
            # **같은 라운드 안에서도** 이 라운드의 test 쪽 데이터가 그 라운드
            # 자신의 train 통계(스케일러 min/max, 포트 빈도 버킷 경계) 계산에
            # 섞여 들어가고 있었다 — train_test_split을 스케일링/버킷 이후에
            # 했기 때문이다. 먼저 원본(raw) 상태로 train/test를 나누고, 그
            # 다음 train 쪽에만 통계를 적합(fit/update)한 뒤 train/test 양쪽에
            # 적용(transform)한다.
            if port_bucketer is not None:
                X_exp_raw, y_exp, cat_exp, port_exp, protocol_exp = chunk
                (X_tr_raw, X_te_raw, y_tr, y_te, cat_tr, _cat_te,
                 port_tr, _port_te, protocol_tr, protocol_te) = train_test_split(
                    X_exp_raw, y_exp, cat_exp, port_exp, protocol_exp,
                    test_size=0.2, stratify=y_exp, random_state=seed)
                port_bucketer.update(port_tr)
                port_bucket_tr = port_bucketer.transform(port_tr)
                port_bucket_te = port_bucketer.transform(_port_te)
                port_onehot_tr = _one_hot(port_bucket_tr, n_categories_cap=3,
                                           categories=np.array([0, 1, 2]))
                port_onehot_te = _one_hot(port_bucket_te, n_categories_cap=3,
                                           categories=np.array([0, 1, 2]))
                protocol_onehot_tr = _one_hot(protocol_tr, categories=protocol_categories)
                protocol_onehot_te = _one_hot(protocol_te, categories=protocol_categories)
                X_tr_raw = np.concatenate(
                    [port_onehot_tr, protocol_onehot_tr, X_tr_raw], axis=1).astype(np.float32)
                X_te_raw = np.concatenate(
                    [port_onehot_te, protocol_onehot_te, X_te_raw], axis=1).astype(np.float32)
            else:
                X_exp_raw, y_exp, cat_exp = chunk
                X_tr_raw, X_te_raw, y_tr, y_te, cat_tr, _cat_te = train_test_split(
                    X_exp_raw, y_exp, cat_exp, test_size=0.2, stratify=y_exp, random_state=seed)

            scaler.partial_fit(X_tr_raw)
            X_tr = scaler.transform(X_tr_raw)
            X_te = scaler.transform(X_te_raw)
            experiences.append({
                "train_X": torch.tensor(X_tr, dtype=torch.float32),
                "train_y": torch.tensor(y_tr, dtype=torch.long),
                "test_X": torch.tensor(X_te, dtype=torch.float32),
                "test_y": torch.tensor(y_te, dtype=torch.long),
                "train_category": cat_tr,
            })
            input_dim = X_tr.shape[1]

    return {"input_dim": input_dim, "experiences": experiences}
