"""NSL-KDD / UNSW-NB15 / X-IIoTID 데이터셋 로더 — PRD 9.1/9.2절.

원본 위치:
  NSL-KDD    : SSF-Strategic-Selection-and-Forgetting/NSL_pre_data/{PKDDTrain+.csv, PKDDTest+.csv}
               (SSF 저장소 전처리 버전)
  UNSW-NB15  : SSF-Strategic-Selection-and-Forgetting/UNSW_pre_data/{UNSWTrain.csv, UNSWTest.csv}
               (SSF 저장소 전처리 버전)
  X-IIoTID   : X-IIoTID-raw/X-IIoTID dataset.csv (IEEE DataPort 원본, 이
               테스트베드가 직접 전처리 — SSF 전처리 버전 없음, `_preprocess_
               xiiotid()` 참고). CND-IDS 원 논문이 실제로 쓴 데이터셋 중
               하나로, 3번째 데이터셋으로 2026-09-13 사용자 확정.

라벨 극성: 0=정상, 1=공격으로 프로젝트 전체에서 통일한다(9.1절). 이 변환은
이 모듈에서 한 번만 수행한다.

이 로더는 두 가지 프로토콜을 지원한다(`preserve_official_split` 인자):

- **True(기본, 원 논문과 동일한 문제)** — KDDTrain+/KDDTest+(UNSW도 동일)를
  합치지 않고 끝까지 분리해서 쓰되, 각 파일을 아래 class-incremental
  분할로 각각 나눈다. MinMaxScaler도 train 파일에만 fit하고 test 파일은
  transform만 한다(test 통계가 정규화에 새어 들어가지 않도록).

- **False** — PRD 9.1절 "병합 규칙": Train 파일 다음 Test 파일 순서로
  pd.concat한 뒤, 병합한 풀을 아래 class-incremental 분할로 나눈다.

**experience 분할은 class-incremental 구조**: CND-IDS 원문의 실제 분할
메커니즘(`CND-IDS/utils.py:275-299`, `create_split_experiences`)을 이식
(`_class_incremental_split`) — 정상 트래픽은 무작위로 고르게 n_experiences개
에 나누고, 공격은 세부 카테고리별로 묶어 라운드로빈으로 배정해 한
experience는 자신에게 배정된 공격 유형만 본다. SSF 원 논문 방식(가변
길이 스트리밍 + drift-조건부 로직)은 기각했다 — SSF는 drift 감지 여부와
무관하게 매 라운드 재학습하고, 라운드 수가 데이터 크기에 따라 달라져
이 테스트베드의 고정 n_experiences 구조와 안 맞으며, SSF 고유 알고리즘을
공유 시나리오로 채택하면 SSF 방법론을 전체 그리드에 강제하는 셈이 된다.

Experience 분할의 test split은 로딩 시점에 한 번만 확정되고 실험 내내 다시
나누지 않는다.
"""

import hashlib
import io
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def _read_nslkdd_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = (df["labels2"].astype(str) != "normal").astype(int).to_numpy()
    # class-incremental 분할용 세부 카테고리(정상/DoS/Probe/R2L/U2R,
    # _class_incremental_split 참고). 정상(y=0) 행의 값은 분할 알고리즘이
    # 아예 안 쓰므로 상관없다.
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
            f"{manual_dir} 에 넣어주세요(SharePoint 호스팅이라 자동 다운로드는 "
            "지원하지 않습니다).")

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


# X-IIoTID(CND-IDS 원 논문이 실제로 쓴 데이터셋, 2026-09-13 추가) — 3번째
# 데이터셋으로 사용자가 확정. NSL-KDD/UNSW-NB15와 달리 SSF가 미리
# 전처리해둔 버전이 없다 — IEEE DataPort 원본 CSV(단일 파일, 공식
# train/test 분리 없음, 820,834행 68컬럼)를 이 테스트베드가 직접
# 전처리한다. 컬럼별 실측 확인(2026-09-13, 전체 820,834행 직접 로드해
# 확인) 근거는 아래 `_preprocess_xiiotid()` docstring 참고.
_XIIOTID_DROP_COLS = [
    # 식별자/시각 — 특정 테스트베드 장치의 IP·포트·날짜에 학습이 묶이면
    # 일반화가 안 된다(NSL-KDD/UNSW-NB15 로더도 IP를 피처로 쓰지 않음).
    "Date", "Timestamp", "Scr_IP", "Des_IP", "Scr_port", "Des_port",
]
_XIIOTID_ONEHOT_COLS = ["Protocol", "Service"]


def _preprocess_xiiotid(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X-IIoTID 원본 CSV(IEEE DataPort, 68컬럼) 전처리.

    **라벨 체계(실측, 2026-09-13)**: `class1`(세부 공격기법 18종, 최소
    Fake_notification=28건/MitM=117건 — 지나치게 희소), `class2`(공격
    단계 9종, 최소 crypto-ransomware=458건), `class3`(이진 Normal/Attack)
    3단 계층. `class1=='Normal'`과 `class3=='Normal'`이 전체 820,834행에서
    완벽히 일치함을 확인(교차표로 검증) — 3개 컬럼이 서로 모순 없는 계층
    구조다. 이 테스트베드는 NSL-KDD(`labels5`, 5종)/UNSW-NB15(`attack_cat`,
    10종)와 비슷한 중간 세분화를 쓰는 관례라, class1(18종, 희소 카테고리
    다수)이 아니라 **class2(9종)를 category로 채택** — U2R(52건, NSL-KDD의
    기존 최소 사례)보다도 작은 class1 카테고리가 여럿이라 라운드 자체가
    무의미해질 위험이 있었다. `y`는 class3에서 유도(0=Normal/1=Attack,
    9.1절 라벨 극성).

    **컬럼별 처리(실측, 2026-09-13 — 전체 파일에서 object dtype 컬럼마다
    `pd.to_numeric(errors='coerce')`로 비수치 토큰을 직접 확인)**:
    - 식별자/시각 6개(`_XIIOTID_DROP_COLS`) — 드롭.
    - `Protocol`(tcp/udp/icmp/'?', 4종)/`Service`(http/dns/mqtt/modbus 등
      17종) — 저카디널리티 범주형, one-hot(`_XIIOTID_ONEHOT_COLS`).
    - `anomaly_alert` — 다른 bool 컬럼(`is_syn_only` 등)과 달리 유일하게
      문자열('TRUE'/'FALSE'/'-')로 저장돼 있다. '-'(결측)는 "경보 없음"과
      같은 의미로 보고 FALSE와 함께 0으로 묶는다(TRUE만 1).
    - `Conn_state` — 이미 원본이 {0,1} 정수로 인코딩해뒀음을 확인(실측),
      그대로 수치 컬럼 취급.
    - 나머지 전부(바이트/패킷 수, rate, 시스템 리소스 Avg/Std 등 약 50개) —
      원본이 결측을 다양한 placeholder로 표기한다: `-`(가장 흔함),
      `?`, `#DIV/0!`, 심지어 `aza`/`excel`(오타로 추정되는 진짜 깨진 값,
      Avg_user_time/Scr_ip_bytes에서 각 1건 실측 확인)까지 나온다.
      `pd.to_numeric(errors='coerce')`로 전부 NaN 처리 후 0으로 채운다
      — "측정 안 됨" ≈ "해당 수량 0"이라는 이 테스트베드의 임의 규칙
      (원 데이터셋 배포자가 결측 사유를 문서화하지 않아 다른 대안(평균
      대치 등)도 똑같이 임의적이다 — 0이 바이트/패킷 카운트류에는
      의미상 가장 자연스러워 채택).

    Returns:
        (X, y, category) — 아직 train/test 분할 전 전체 820,834행.
    """
    y = (df["class3"].astype(str) == "Attack").astype(int).to_numpy()
    category = df["class2"].astype(str).to_numpy()

    X_df = df.drop(columns=_XIIOTID_DROP_COLS + ["class1", "class2", "class3"])
    X_df["anomaly_alert"] = (X_df["anomaly_alert"].astype(str) == "TRUE").astype(float)
    X_df = pd.get_dummies(X_df, columns=_XIIOTID_ONEHOT_COLS, dtype=float)

    for col in X_df.columns:
        if X_df[col].dtype == bool:
            X_df[col] = X_df[col].astype(float)
        elif X_df[col].dtype == object:
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
    X_df = X_df.fillna(0.0)

    return X_df.to_numpy(dtype=np.float64), y, category


def _load_xiiotid_raw(base_dir: str):
    path = os.path.join(base_dir, "X-IIoTID-raw", "X-IIoTID dataset.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"X-IIoTID 원본이 로컬에 없습니다({path}). IEEE DataPort"
            "(https://ieee-dataport.org/documents/x-iiotid-connectivity-agnostic-and-device-agnostic-intrusion-data-set-industrial)"
            f"에서 CSV를 받아 {os.path.join(base_dir, 'X-IIoTID-raw')} 에 "
            "'X-IIoTID dataset.csv'라는 이름으로 넣어주세요.")
    df = pd.read_csv(path, low_memory=False)
    X, y, category = _preprocess_xiiotid(df)
    # 공식 train/test 분리가 없는 유일한 데이터셋(NSL-KDD/UNSW-NB15는
    # 각자 원 논문의 공식 분리 파일이 있음) — 여기서 한 번 분할해 이후
    # 파이프라인(`_load_dataset_uncached`의 preserve_official_split=True
    # 경로)이 NSL-KDD/UNSW-NB15와 동일하게 다루도록 한다. category로
    # stratify해 희소 카테고리(crypto-ransomware=458건)도 train/test
    # 양쪽에 반드시 나타나게 한다 — 안 그러면 `_class_incremental_split`의
    # "test에만 있는 카테고리" 안전장치가 곧바로 예외를 던진다.
    #
    # random_state=42를 `load_dataset()`이 받는 `seed` 인자와 무관하게
    # 고정한다(2026-09-14 재검토로 명확화, 실제로는 처음부터 의도된
    # 동작 — 멀티시드 실행 시 헷갈리기 쉬워 남겨둠) — NSL-KDD/UNSW-NB15의
    # "공식 분리 파일"과 정확히 같은 역할을 한다. 그 둘도 seed가 바뀌어도
    # train/test 행 소속 자체는 절대 안 바뀐다(파일이 고정이므로) — seed는
    # 오직 `_class_incremental_split`의 라운드 배정/블러링에만 쓰인다.
    # X-IIoTID는 공식 파일이 없어 이 한 번의 분할을 "고정 공식 분리"의
    # 대용으로 삼는 것이라, 같은 원칙으로 seed와 독립적이어야 다른 두
    # 데이터셋과 "무엇이 seed에 따라 변하는가"가 일관된다 — 멀티시드
    # 재실행에서 X-IIoTID의 train/test 구성 자체가 시드마다 바뀌면 오히려
    # NSL-KDD/UNSW-NB15와 다른 종류의 변동성을 추가로 섞는 셈이 된다.
    X_train, X_test, y_train, y_test, category_train, category_test = train_test_split(
        X, y, category, test_size=0.2, random_state=42, stratify=category)
    return X_train, y_train, X_test, y_test, category_train, category_test


def _count_attack_categories(category: np.ndarray, y: np.ndarray) -> int:
    """공격(y==1) 행에 등장하는 고유 category 값의 개수.

    2026-09-03 추가 — 사용자 결정: n_experiences를 데이터셋 공통 상수(기존
    5, CND-IDS 원 논문이 X-IIoTID/CICIDS2017/UNSW-NB15에 쓴 값을 그리드
    전체 동일 적용 원칙으로 확장한 것뿐이었다 — global_hparams.yaml 옛
    주석 참고)이 아니라, 데이터셋 자신의 실제 공격 유형 수에 맞춘다 —
    "라운드 하나 = 새 공격 유형 하나"가 정확히 성립하도록. `n_experiences`가
    데이터셋 간 일치해야 할 통계적 필요성은 없다(리더보드가 데이터셋별로
    완전히 분리돼 있어 데이터셋을 가로질러 비교하는 일 자체가 없음,
    `leaderboard_builder.py` 참고) — 반면 데이터셋 자신의 공격 유형 수와
    라운드 수가 어긋나면 한 라운드에 여러 공격이 섞이거나 공격이 아예 없는
    라운드가 생겨, per_category_final의 forgetting 지표(그 공격이 처음
    등장한 라운드 recall - 마지막 라운드 recall)가 "그 공격 하나의 순수한
    학습/망각"이 아니라 같은 라운드에 섞인 다른 공격들과의 상호작용까지
    반영하게 된다."""
    category = np.asarray(category)
    y = np.asarray(y)
    return len(set(category[y == 1].tolist()))


def _hold_out_normal_reference(
        X: np.ndarray, y: np.ndarray, category: np.ndarray,
        p: float = 0.1, seed: int = 42,
        ) -> Tuple[np.ndarray, ...]:
    """CND-IDS 원문(`CND-IDS/utils.py`의 `get_*_benchmark` 계열)의
    `init_normal` 추출 — 정상(y=0) 행마다 독립적인 베르누이(p) 코인플립으로
    참조 세트를 뽑아 스트림 풀에서 제거한다(정확히 p 비율의 슬라이스가
    아니다). train/test 분할보다도, 클래스-증분 분할보다도 먼저 적용한다
    — 원문이 실제로 이 순서(`get_*_benchmark` → `train_test_split`이 이미
    축소된 풀 위에서 일어남)를 따르기 때문이다. 공격(y=1) 행은 대상이
    아니다.

    Returns:
        (X_remain, y_remain, category_remain, X_held, y_held, category_held)
        held 쪽은 전부 y=0(정상)이다.
    """
    rng = np.random.default_rng(seed)
    is_normal = (y == 0)
    coin = rng.random(len(y)) < p
    held_mask = is_normal & coin
    remain_mask = ~held_mask
    return (X[remain_mask], y[remain_mask], category[remain_mask],
            X[held_mask], y[held_mask], category[held_mask])


def _class_incremental_split(
        X: np.ndarray, y: np.ndarray, category: np.ndarray, n_experiences: int, seed: int,
        class_order: Optional[List[List[str]]] = None,
        blurry_ratio: float = 0.0,
        blurry_categories: Optional[set] = None,
        min_per_round_share: int = 30,
        ) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[List[str]], set]:
    """CND-IDS 원 논문의 실제 experience 분할 메커니즘 이식
    (`CND-IDS/utils.py:275-299`, `create_split_experiences`) + i-Blurry
    재등장 확장(Koh et al., "Online Continual Learning on Class Incremental
    Blurry Task Configuration with Anytime Inference", ICLR 2022 — 수정
    계획 3절, 2026-09-12 추가).

    - 정상(y=0) 행: 고정 seed로 셔플 후 n_experiences개에 고르게 분배 —
      `category` 값은 무시한다(정상 행의 category가 무엇이든 상관없음).
    - 공격(y=1) 행: `category`별로 묶어 라운드로빈으로 "주 라운드"에 배정
      (`class_order[i % n_experiences].append(category)`, category는 정렬된
      순서로 순회 — 임의 순서를 만들지 않는다).
    - **블러링(`blurry_ratio > 0`일 때만)**: 카테고리 수가 적은 NIDS
      데이터셋에서는 원문의 "클래스의 N%를 블러리 그룹으로"가 잘 안 맞아
      (예: NSL-KDD 4개 카테고리 중 30%는 1.2개), 표본 단위로 재정의한다 —
      블러링 대상 카테고리(아래 `blurry_categories` 참고)의 표본 중
      `blurry_ratio`만큼을 무작위로 골라 주 라운드가 아닌 **다른 모든
      라운드에 고르게** 재배정한다. 나머지 `(1-blurry_ratio)`는 주 라운드에
      그대로 남는다. `blurry_ratio=0`(기본값)이면 새 RNG 소비 없이 기존
      경로를 그대로 타 하위호환이 완전하다(아래 "빠른 경로" 참고).
    - 최종적으로 각 experience = (배정된 공격 행) ∪ (해당 인덱스의 정상
      청크), 다시 한 번 셔플해 정상/공격을 섞는다(CND-IDS 원문의 마지막
      `shuffle_idx` 단계와 동일).

    Args:
        class_order: 이미 계산된 category -> 주 라운드 배정을 재사용하려면
            전달한다(예: train에서 계산한 걸 test 분할에도 똑같이 써야
            experience i의 test가 experience i의 train과 같은 category를
            반영한다 — CND-IDS 원문도 `train_experiences`/`test_experiences`
            양쪽에 같은 `class_order`를 넘긴다). None이면 이 호출에서 새로
            계산해서 두 번째 반환값으로 돌려준다.
        blurry_categories: 이미 계산된 "블러링 대상 카테고리 집합"을
            재사용하려면 전달한다(class_order와 같은 재사용 원칙 — 어떤
            카테고리가 블러링 대상인지는 train에서 1회 판정해 test에도
            동일 적용한다. 카테고리별 train/test 표본 수가 서로 달라 test
            에서 독립적으로 재판정하면 같은 라운드의 train/test가 다른
            규칙을 쓰는 모순이 생길 수 있다 — 실측: NSL-KDD R2L은 train
            995건/test 2754건). None이면 이 호출에서 새로 판정해서 세 번째
            반환값으로 돌려준다 — 판정 기준은
            `count * blurry_ratio / (n_experiences-1) >= min_per_round_share`
            (한 비-주 라운드가 받는 몫이 이 값 미만이면 disjoint 유지).
        min_per_round_share: 위 판정 기준의 임계값(기본 30). `blurry_
            categories`를 직접 넘기면 무시된다.

    Returns:
        (experiences, class_order, blurry_categories) — experiences는
        [(X_exp, y_exp, category_exp)] * n_experiences, class_order/
        blurry_categories는 재사용/기록용으로 반환(blurry_ratio=0이면
        blurry_categories는 항상 빈 set).
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
        # class_order는 train에서 계산해 test 분할에 재사용된다. test에만
        # 존재하고 train에는 없던 공격 category가 있으면 class_order의
        # 어느 experience 목록에도 없어 np.isin이 항상 False를 반환하고
        # 조용히 누락된다 — 즉시 예외를 던진다.
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

    if blurry_categories is None:
        blurry_categories = set()
        if blurry_ratio > 0 and n_experiences > 1:
            cats, counts = np.unique(category[attack_idx], return_counts=True)
            for cat, cnt in zip(cats, counts):
                per_round_share = cnt * blurry_ratio / (n_experiences - 1)
                if per_round_share >= min_per_round_share:
                    blurry_categories.add(str(cat))

    if blurry_ratio <= 0 or n_experiences <= 1 or not blurry_categories:
        # 빠른 경로 — blurry_ratio=0(기본값)이면 아래 블러링 코드가 쓰는
        # rng.random()/rng.randint() 호출 자체를 건너뛰어, 원래 코드와
        # RNG 소비량까지 완전히 동일하게 유지한다("blurry_ratio=0은
        # disjoint와 완전히 동일해야 한다"는 이 함수의 요구사항).
        experiences = []
        for i in range(n_experiences):
            cat_mask = np.isin(category[attack_idx], class_order[i])
            exp_attack_idx = attack_idx[cat_mask]
            exp_idx = np.concatenate([exp_attack_idx, normal_chunks[i]])
            exp_idx = exp_idx[rng.permutation(len(exp_idx))]
            experiences.append((X[exp_idx], y[exp_idx], category[exp_idx]))
        return experiences, class_order, blurry_categories

    # --- 블러링 적용 ---
    cat_to_primary_round: Dict[str, int] = {}
    for i, cats_i in enumerate(class_order):
        for c in cats_i:
            cat_to_primary_round[c] = i
    primary_round_per_row = np.array(
        [cat_to_primary_round[c] for c in category[attack_idx]])
    is_eligible = np.isin(category[attack_idx], list(blurry_categories))
    coin = rng.random(len(attack_idx)) < blurry_ratio
    should_blur = is_eligible & coin
    row_round = primary_round_per_row.copy()
    n_blur = int(should_blur.sum())
    if n_blur > 0:
        # primary가 아닌 다른 라운드로 균등하게 보낸다: offset을
        # 1..n_experiences-1에서 뽑아 (primary+offset) % n_experiences로
        # 계산하면 primary 자신은 절대 나오지 않으면서 나머지
        # n_experiences-1개 라운드에 고르게 분포한다.
        offsets = rng.randint(1, n_experiences, size=n_blur)
        row_round[should_blur] = (
            (primary_round_per_row[should_blur] + offsets) % n_experiences)

    experiences = []
    for i in range(n_experiences):
        exp_attack_idx = attack_idx[row_round == i]
        exp_idx = np.concatenate([exp_attack_idx, normal_chunks[i]])
        exp_idx = exp_idx[rng.permutation(len(exp_idx))]
        experiences.append((X[exp_idx], y[exp_idx], category[exp_idx]))

    return experiences, class_order, blurry_categories


def _source_csv_signature(name: str, base_dir: str) -> List[Tuple[str, int, int]]:
    """이 데이터셋이 실제로 읽는 원본 CSV들의 (상대경로, 크기, mtime) 목록.

    코드가 안 바뀌었어도 원본 파일을 교체/재다운로드하면 캐시가 낡은
    데이터를 반환하면 안 된다."""
    if name == "nsl-kdd":
        paths = [
            os.path.join(base_dir, "SSF-Strategic-Selection-and-Forgetting",
                         "NSL_pre_data", "PKDDTrain+.csv"),
            os.path.join(base_dir, "SSF-Strategic-Selection-and-Forgetting",
                         "NSL_pre_data", "PKDDTest+.csv"),
        ]
    elif name == "unsw-nb15":
        paths = [
            os.path.join(base_dir, "SSF-Strategic-Selection-and-Forgetting",
                         "UNSW_pre_data", "UNSWTrain.csv"),
            os.path.join(base_dir, "SSF-Strategic-Selection-and-Forgetting",
                         "UNSW_pre_data", "UNSWTest.csv"),
            os.path.join(base_dir, "UNSW-NB15-raw", "UNSW_NB15_training-set.csv"),
            os.path.join(base_dir, "UNSW-NB15-raw", "UNSW_NB15_testing-set.csv"),
        ]
    elif name == "x-iiotid":
        paths = [os.path.join(base_dir, "X-IIoTID-raw", "X-IIoTID dataset.csv")]
    else:
        paths = []
    sig = []
    for p in paths:
        if os.path.exists(p):
            st = os.stat(p)
            sig.append((os.path.relpath(p, base_dir), st.st_size, int(st.st_mtime)))
    return sorted(sig)


def _dataset_cache_key(name: str, base_dir: str, n_experiences: Optional[int], seed: int,
                        preserve_official_split: bool,
                        blurry_ratio: float = 0.0, min_per_round_share: int = 30) -> str:
    """이 모듈(dataset_loader.py) 코드 해시 + 인자 + 원본 CSV 서명을 합쳐
    캐시 유효성 판단 키를 만든다. 이 중 하나라도 바뀌면 다른 해시가 나와
    캐시를 못 찾고 자동으로 재계산한다(grid_runner.py의 code_version
    캐시와 같은 원칙 — 절대 "낡았을 수도 있는 캐시"를 조용히 재사용하지
    않는다). blurry_ratio/min_per_round_share도 포함(2026-09-12, i-Blurry)
    — 안 넣으면 blurry_ratio=0으로 만든 캐시를 blurry_ratio=0.3 요청이
    조용히 재사용하는 사고가 난다."""
    with io.open(__file__, "rb") as f:
        code_hash = hashlib.sha256(f.read()).hexdigest()
    key_parts = repr((
        name, n_experiences, seed, preserve_official_split, blurry_ratio,
        min_per_round_share, code_hash,
        _source_csv_signature(name, base_dir),
    ))
    return hashlib.sha256(key_parts.encode("utf-8")).hexdigest()[:24]


def _dataset_cache_path(base_dir: str, name: str, cache_key: str) -> str:
    cache_dir = os.path.join(base_dir, "testbed", "data", ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{name}_{cache_key}.pkl")


def load_dataset(name: str, base_dir: str, n_experiences: Optional[int] = None,
                  seed: int = 42, preserve_official_split: bool = True,
                  blurry_ratio: float = 0.0, min_per_round_share: int = 30) -> Dict:
    """`_load_dataset_uncached()`의 결과를 디스크에 캐싱하는 래퍼.

    n_experiences=None(기본값)이면 데이터셋 자신의 실제 공격 유형 수로
    자동 결정된다(`_count_attack_categories()`/`_load_dataset_uncached()`
    참고, 2026-09-03 추가 — "라운드 하나 = 새 공격 유형 하나"가 되도록
    사용자가 결정, 예전의 데이터셋 공통 고정값 5는 폐기). 명시적으로 정수를
    넘기면(디버깅 등) 그 값을 그대로 쓴다.

    blurry_ratio(기본 0.0, i-Blurry — 2026-09-12 도입): 0이면
    기존 disjoint 분할과 완전히 동일(`_class_incremental_split` 참고).
    0보다 크면 블러링 대상 카테고리 표본 중 이 비율만큼을 주 라운드가
    아닌 다른 모든 라운드에 재배정한다. min_per_round_share는 블러링 대상
    판정 임계값(기본 30, `_class_incremental_split` docstring 참고)."""
    cache_key = _dataset_cache_key(name, base_dir, n_experiences, seed, preserve_official_split,
                                    blurry_ratio, min_per_round_share)
    cache_path = _dataset_cache_path(base_dir, name, cache_key)
    if os.path.exists(cache_path):
        print(f"{name}: 캐시에서 로드합니다 ({cache_path})")
        with io.open(cache_path, "rb") as f:
            return pickle.load(f)

    result = _load_dataset_uncached(
        name, base_dir, n_experiences=n_experiences, seed=seed,
        preserve_official_split=preserve_official_split,
        blurry_ratio=blurry_ratio, min_per_round_share=min_per_round_share)

    tmp_path = f"{cache_path}.tmp{os.getpid()}"
    with io.open(tmp_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_path)
    print(f"{name}: 전처리 결과를 캐시에 저장했습니다 ({cache_path})")
    return result


def _load_dataset_uncached(name: str, base_dir: str, n_experiences: Optional[int] = None,
                            seed: int = 42, preserve_official_split: bool = True,
                            blurry_ratio: float = 0.0, min_per_round_share: int = 30) -> Dict:
    """PRD 9.1/9.2절 로드. `preserve_official_split=True`면 모듈 docstring의
    "원 논문과 동일한 문제" 프로토콜을 쓴다.

    Args:
        name: 'nsl-kdd' | 'unsw-nb15'
        base_dir: FCL 저장소 루트 경로(SSF 원본 데이터 폴더를 포함하는 위치).
        n_experiences: None(기본값)이면 이 데이터셋의 실제 공격 category 수로
              자동 결정한다(`_count_attack_categories()` 참고, 2026-09-03
              추가) — 데이터셋 간에 일치해야 할 필요는 없다는 게 확인되어
              (리더보드가 데이터셋별로 완전히 분리되어 데이터셋을 가로질러
              비교하지 않음) 데이터셋 자신의 공격 유형 수에 맞추기로 사용자가
              결정했다. 정수를 명시하면(디버깅 등) 그 값을 그대로 쓴다.
        seed: 10.1절 기본값 42 — preserve_official_split=False일 때 experience
              내부 stratified split에 사용.
        preserve_official_split: True면 원본 train/test 파일을 합치지 않고
              끝까지 분리해서 쓴다(모듈 docstring 참고).
        blurry_ratio, min_per_round_share: i-Blurry 재등장 비율/임계값
              (`_class_incremental_split` 참고). 기본값(0.0)이면 기존
              disjoint 분할과 완전히 동일.

    Returns:
        {'input_dim': int, 'experiences': [{'train_X','train_y','test_X','test_y',
        'train_category'}] * n_experiences, 'held_out_normal_reference':
        [torch.Tensor] * n_experiences}. train_category는 numpy 문자열
        배열(정상="normal"/데이터셋별 표기, 공격은 family 이름) —
        train_y와 같은 행 순서, 길이만 대응. CADE의 다중클래스 centroid
        구성에만 쓰이고(선택적 소비, `pipeline/cl_client.py` 참고) 다른
        컴포넌트는 이 키를 몰라도 된다. `test_category`는 test_y와 같은
        행 순서의 category 배열. 학습 경로(`CLClient`)는 이 키를 읽지
        않고, `experiments/grid_runner.py`가 공격 category별 recall
        리포팅에만 쓴다. `held_out_normal_reference`는 CND-IDS N_c(원문의
        `init_normal`) 이식 — 라운드별로 그 시점 scaler로 재스케일된
        같은 정상 참조 데이터(`_hold_out_normal_reference` 참고), Track B의
        `CNDIDSAntiForgetting`/`PCAScorer`만 소비한다(`consumes_held_out_
        reference` 클래스 속성으로 게이팅, `pipeline/cl_client.py` 참고).
    """
    if name == "nsl-kdd":
        X_train, y_train, X_test, y_test, category_train, category_test = _load_nslkdd_raw(base_dir)
    elif name == "unsw-nb15":
        X_train, y_train, X_test, y_test, category_train, category_test = _load_unsw_raw(base_dir)
    elif name == "x-iiotid":
        X_train, y_train, X_test, y_test, category_train, category_test = _load_xiiotid_raw(base_dir)
    else:
        raise ValueError(
            f"Unknown dataset: {name!r} (expected 'nsl-kdd', 'unsw-nb15', or 'x-iiotid')")

    if n_experiences is None:
        # 2026-09-03 추가 — "라운드 하나 = 새 공격 유형 하나"가 되도록
        # 데이터셋 자신의 실제 공격 category 수로 라운드 수를 정한다.
        # category_train/y_train만 본다 — class_order도 train 기준으로만
        # 계산되므로(`_class_incremental_split` 참고, test에만 있고 train에
        # 없는 category는 애초에 예외를 던짐) 일관된 기준이다.
        n_experiences = _count_attack_categories(category_train, y_train)
        print(f"{name}: n_experiences를 실제 공격 유형 수({n_experiences}개)로 자동 설정합니다.")

    experiences = []

    held_out_normal_reference: List[torch.Tensor] = []

    if preserve_official_split:
        # CND-IDS N_c(held-out 정상 참조) — 원문 그대로 클래스-증분 분할
        # 이전, train 풀에서 즉시 떼어낸다(모듈 "_hold_out_normal_reference"
        # 참고). test 쪽은 대상이 아니다 — 이 값을 소비하는 컴포넌트
        # (CNDIDSAntiForgetting/PCAScorer)는 전부 train-side 데이터에서
        # 파생된 "정상 참조"만 다룬다.
        X_train, y_train, category_train, held_X, held_y, held_cat = (
            _hold_out_normal_reference(X_train, y_train, category_train, seed=seed))

        # class_order는 train에서 계산해 test 분할에 그대로 재사용한다 —
        # experience i의 test가 experience i의 train과 같은 공격 category를
        # 반영하도록(CND-IDS 원문도 train/test 양쪽에 같은 class_order를
        # 쓴다). train/test는 서로 다른 seed로 셔플한다. 원본(미정규화)
        # 데이터로 먼저 분할한 뒤 scaler를 적용한다.
        train_chunks, class_order, blurry_categories = _class_incremental_split(
            X_train, y_train, category_train, n_experiences, seed,
            blurry_ratio=blurry_ratio, min_per_round_share=min_per_round_share)
        # blurry_categories도 class_order와 같은 원칙으로 train에서 계산해
        # test에 재사용한다(함수 docstring 참고) — test 자신의 카테고리별
        # 표본 수로 독립적으로 재판정하면 같은 라운드의 train/test가 다른
        # 블러링 규칙을 쓰는 모순이 생길 수 있다.
        test_chunks, _, _ = _class_incremental_split(
            X_test, y_test, category_test, n_experiences, seed + 1, class_order=class_order,
            blurry_ratio=blurry_ratio, blurry_categories=blurry_categories)

        # MinMaxScaler를 라운드마다 partial_fit으로 누적 적용 — train 파일
        # 전체에 한 번에 fit하면 experience 0의 모델이 아직 등장하지 않은
        # experience 4의 데이터 범위(min/max)까지 알게 된다(미래 정보 누출).
        # experience i는 항상 0..i의 통계만 안다. test는 fit에 참여하지 않는다.
        scaler = MinMaxScaler()
        for (X_tr, y_tr, cat_tr), (X_te, y_te, cat_te) in zip(train_chunks, test_chunks):
            scaler.partial_fit(X_tr)
            X_tr_scaled = scaler.transform(X_tr)
            X_te_scaled = scaler.transform(X_te)
            # held-out 참조는 라운드마다 그 시점까지의 scaler 상태로
            # causally 재스케일한다(test_X와 동일한 방식) — 참조 자체는
            # 전 라운드에 걸쳐 불변이지만 스케일은 라운드별로 다르다.
            held_out_normal_reference.append(
                torch.tensor(scaler.transform(held_X), dtype=torch.float32))
            experiences.append({
                "train_X": torch.tensor(X_tr_scaled, dtype=torch.float32),
                "train_y": torch.tensor(y_tr, dtype=torch.long),
                "test_X": torch.tensor(X_te_scaled, dtype=torch.float32),
                "test_y": torch.tensor(y_te, dtype=torch.long),
                "train_category": cat_tr,
                "test_category": cat_te,
            })
        input_dim = X_train.shape[1]
    else:
        df_X = np.concatenate([X_train, X_test], axis=0)
        df_y = np.concatenate([y_train, y_test], axis=0)
        df_category = np.concatenate([category_train, category_test], axis=0)

        # 위와 같은 이유로, 정규화 전(원본) 데이터로 먼저 class-incremental
        # 분할을 한 뒤, experience 순서대로 scaler를 누적(`partial_fit`) 적용한다.
        df_X, df_y, df_category, held_X, held_y, held_cat = _hold_out_normal_reference(
            df_X, df_y, df_category, seed=seed)

        exp_chunks, _, _ = _class_incremental_split(
            df_X, df_y, df_category, n_experiences, seed,
            blurry_ratio=blurry_ratio, min_per_round_share=min_per_round_share)
        scaler = MinMaxScaler()
        input_dim = None
        for X_exp_raw, y_exp, cat_exp in exp_chunks:
            # 같은 라운드 안에서도 test 쪽이 train 통계(스케일러 min/max)
            # 계산에 섞이지 않도록, 먼저 원본(raw) 상태로 train/test를 나눈
            # 뒤 train에만 적합(fit)하고 train/test 양쪽에 적용(transform)한다.
            X_tr_raw, X_te_raw, y_tr, y_te, cat_tr, cat_te = train_test_split(
                X_exp_raw, y_exp, cat_exp, test_size=0.2, stratify=y_exp, random_state=seed)

            scaler.partial_fit(X_tr_raw)
            X_tr = scaler.transform(X_tr_raw)
            X_te = scaler.transform(X_te_raw)
            held_out_normal_reference.append(
                torch.tensor(scaler.transform(held_X), dtype=torch.float32))
            experiences.append({
                "train_X": torch.tensor(X_tr, dtype=torch.float32),
                "train_y": torch.tensor(y_tr, dtype=torch.long),
                "test_X": torch.tensor(X_te, dtype=torch.float32),
                "test_y": torch.tensor(y_te, dtype=torch.long),
                "train_category": cat_tr,
                "test_category": cat_te,
            })
            input_dim = X_tr.shape[1]

    return {"input_dim": input_dim, "experiences": experiences,
            "held_out_normal_reference": held_out_normal_reference}
