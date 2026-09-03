"""평가 지표 — PRD 3절.

주지표 4개(F1/Precision/Recall/PR-AUC)만 리더보드 정렬에 쓴다. F1/Precision/
Recall은 TP/FP/FN을 직접 계산하는 방식으로 구현하고(3.1절), PR-AUC만
`sklearn.metrics.average_precision_score`를 그대로 사용한다(재구현 금지).

BWT는 표준 continual-learning 문헌의 정의와 일치하는 공식으로 내부 로그·회귀
테스트용으로 계산한다(3.3절). 이 공식은 CND-IDS 저자 자신의 제안 방법(라벨-프리
CFE+클러스터링) 경로가 아니라, 같은 저장소에 포함된 ADCN 비교 베이스라인의
평가 코드에서 가져온 것이다 — CND-IDS 자체 방법은 BWT를 계산하지 않는다.
`bwt()` docstring 참고.
"""

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import average_precision_score


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, fn


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Precision = TP / (TP + FP). (PRD 3.1절, SSF Table I / CADE Table 3 근거)"""
    tp, fp, _ = _confusion_counts(y_true, y_pred)
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall = TP / (TP + FN). (PRD 3.1절, SSF Table I / CADE Table 3 근거)"""
    tp, _, fn = _confusion_counts(y_true, y_pred)
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 = 2*P*R / (P+R). (PRD 3.1절, SSF Table I / CADE Table 3 근거)"""
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    if p + r == 0.0:
        return 0.0
    return float(2 * p * r / (p + r))


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """PR-AUC = sklearn.metrics.average_precision_score (재구현 금지, PRD 3.1절).

    근거: CND-IDS "Pre-threshold Evaluation" 절, SPIDER Table IV/V/VII.
    """
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_score = np.asarray(y_score).reshape(-1).astype(float)
    if len(np.unique(y_true)) < 2:
        # 한 클래스만 존재하면 average_precision_score가 정의되지 않는다.
        return 0.0
    return float(average_precision_score(y_true, y_score))


def bwt(r_matrix: Sequence[Sequence[float]]) -> float:
    """Backward Transfer (PRD 3.3절).

    BWT = (1 / (T-1)) * sum_{i=0}^{T-2} (R_{T-1,i} - R_{i,i})

    주의(2026-08-12 정정): 이 공식은 "CND-IDS 원 논문"이 아니라 CND-IDS
    저장소에 포함된 **ADCN 비교 베이스라인**의 평가 코드
    (`AutonomousDCN/ADCNmainloop.py:418`)에서 옮긴 것이다—
    `BWT = 1/(nTask-1)*(sum(allTaskAccuracies)-sum(postTaskAcc))`.
    CND-IDS 저자 자신의 제안 방법(라벨-프리 CFE+클러스터링) 경로는 BWT를
    전혀 계산하지 않는다. 다만 이 공식 자체는 표준 continual-learning
    문헌의 BWT 정의와 정확히 일치하므로(Lopez-Paz & Ranzato 2017 스타일),
    "표준 정의를 그대로 구현했다"는 근거로는 유효하다 — 원 구현대로
    `allTaskAccuracies`/`postTaskAcc` 둘 다 **마지막 태스크를 제외한**
    T-1개 태스크만 순회한다(같은 파일 406행 주석 "except the last
    task. For calculating BWT"). 분모도 `(T-1)`이지 `T(T-1)/2`가 아니다 —
    이전 구현은 분모를 `T(T-1)/2`로 잘못 써서(값이 2.5배 작게 나옴, T=5 기준)
    실측 대조 없이 넘어간 버그였다.

    r_matrix는 (T, T) 크기의 F1 성능 행렬 (PRD 3.4절 정의). R_{T-1,i}는
    마지막 행(0-indexed T-1행), R_{i,i}는 대각 원소다.

    Args:
        r_matrix: T x T performance matrix (R_ij = experience i까지 학습한
                  모델을 experience j에서 평가한 F1).

    Returns:
        BWT 스칼라 값. T < 2 이면 정의되지 않으므로 0.0을 반환한다.
    """
    R = np.asarray(r_matrix, dtype=float)
    T = R.shape[0]
    if T < 2:
        return 0.0
    last = T - 1  # 0-indexed 마지막 행
    total = 0.0
    for i in range(T - 1):  # 마지막 태스크(i=last) 제외, CND-IDS 원본과 동일
        total += R[last, i] - R[i, i]
    return float(total / (T - 1))


def build_r_matrix(f1_grid: List[List[float]]) -> np.ndarray:
    """T x T 성능 행렬을 numpy 배열로 정규화한다 (PRD 3.4절 R 정의).

    Args:
        f1_grid: 리스트의 리스트. f1_grid[i][j] = experience i까지 학습한
                 모델을 experience j의 test split에서 평가한 F1.

    Returns:
        (T, T) numpy 배열.

    Raises:
        ValueError: 정사각 행렬이 아닌 경우.
    """
    R = np.asarray(f1_grid, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"R matrix must be square (T, T), got shape {R.shape}")
    return R


def per_category_counts(y_true: np.ndarray, y_pred: np.ndarray,
                        category: np.ndarray) -> Dict[str, Dict[str, int]]:
    """category(문자열 라벨)별 혼동 계수 — 2026-09-03 추가(공격 유형별 탐지
    성능 리포팅용, 리더보드 정렬에는 쓰지 않는다).

    정상/공격 판정은 category 문자열이 아니라 `y_true`(0=정상, 1=공격)로 한다
    — 데이터셋마다 정상 표기가 다르고("normal"/"Benign"), UNSW-NB15는 정상
    행의 attack_cat이 비어 있어 로더가 임의 문자열로 채운다(dataset_loader.py
    `_load_unsw_attack_cat` 참고). 그래서 한 category 안에 y=0/1이 섞여
    있어도(원칙적으로는 없어야 하지만) 정상 행은 fp/n에, 공격 행은
    tp/n_attack에 각각 정확히 들어간다.

    Returns:
        {category: {"n": 행 수, "n_attack": y==1 수, "tp": y==1 & pred==1,
                    "fp": y==0 & pred==1}}
    """
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    category = np.asarray(category).reshape(-1)
    if not (len(y_true) == len(y_pred) == len(category)):
        raise ValueError(
            f"per_category_counts: 길이 불일치 y_true={len(y_true)} "
            f"y_pred={len(y_pred)} category={len(category)}")
    counts: Dict[str, Dict[str, int]] = {}
    for cat in np.unique(category):
        m = category == cat
        yt, yp = y_true[m], y_pred[m]
        counts[str(cat)] = {
            "n": int(m.sum()),
            "n_attack": int((yt == 1).sum()),
            "tp": int(((yt == 1) & (yp == 1)).sum()),
            "fp": int(((yt == 0) & (yp == 1)).sum()),
        }
    return counts


def per_category_recall(counts: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """`per_category_counts()` 결과에서 **공격 행이 있는** category만 골라
    recall(tp / n_attack)을 돌려준다. 공격 행이 하나도 없는 category(정상
    category)는 제외한다 — 정상 쪽은 `per_category_fpr()`로 본다."""
    return {
        cat: float(c["tp"] / c["n_attack"])
        for cat, c in counts.items() if c["n_attack"] > 0
    }


def per_category_fpr(counts: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """정상 행(y=0)이 있는 category의 FPR(fp / 정상 행 수)."""
    return {
        cat: float(c["fp"] / (c["n"] - c["n_attack"]))
        for cat, c in counts.items() if (c["n"] - c["n_attack"]) > 0
    }
