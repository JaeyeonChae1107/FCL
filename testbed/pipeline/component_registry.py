"""Component registry — 문자열 키를 컴포넌트 클래스에 매핑한다 (PRD 4절 표 그대로).

Paper → component mapping (슬롯 순서: drift / sample / memory / anti / anomaly):
  SSF     : ssf    / ssf    / ssf 또는 spider / lwf_ssf 또는 none / cade_mad 또는 none
  CADE    : cade   / random / none   / none    / cade_mad
  SPIDER  : none   / random / spider / gpm     / cade_mad 또는 none
  CND-IDS : none   / random / cndids 또는 spider / cndids  / pca

memory_manager='spider'는 SPIDER 원 논문의 "유한 버퍼 메모리(M)" 메커니즘이다
(components/spider_gpm/spider_memory_manager.py 참고, 사용자가 원 논문을
직접 확인해 제공한 근거). Track A/B 양쪽에서 쓸 수 있다(사용자 지시).
기존 memory_manager='fifo'(특정 논문 근거 없는 공통 baseline)는 이 SPIDER
메커니즘으로 대체되어 삭제했다.

anomaly_scorer는 Track B에서 pca만 남겼다 — lof/dif는 CND-IDS 원 논문이
"자체 제안한 방법"이 아니라 "비교를 위해 인용한 제3자(다른 논문) baseline"
이라(부록A: "CND-IDS 비교 baseline"), PRD 0절의 "네 논문이 실제로 제안한
메커니즘만 재조합한다" 원칙에 비춰볼 때 포함 근거가 약했다(사용자 지시).
"""

import inspect
from typing import Any, Dict

from testbed.components.cade import CADEDriftDetector, CADEMADScorer
from testbed.components.cndids import CNDIDSAntiForgetting, CNDIDSMemoryManager
from testbed.components.novelty_baselines import PCAScorer
from testbed.components.spider_gpm import GPMAntiForgetting, SPIDERMemoryManager
from testbed.components.ssf import (
    SSFDriftDetector,
    SSFSampleSelector,
    SSFMemoryManager,
    SSFAntiForgetting,
)
from testbed.pipeline.common_baselines import (
    NoDriftDetector,
    RandomSelector,
    NoMemoryManager,
    NoAntiForgetting,
    NoAnomalyScorer,
)

REGISTRY: Dict[str, Dict[str, Any]] = {
    "drift_detector": {
        "none": NoDriftDetector,
        "ssf": SSFDriftDetector,
        "cade": CADEDriftDetector,
    },
    "sample_selector": {
        "random": RandomSelector,
        "ssf": SSFSampleSelector,
    },
    "memory_manager": {
        "none": NoMemoryManager,
        "spider": SPIDERMemoryManager,
        "ssf": SSFMemoryManager,
        "cndids": CNDIDSMemoryManager,
    },
    "anti_forgetting": {
        "none": NoAntiForgetting,
        "lwf_ssf": SSFAntiForgetting,
        "gpm": GPMAntiForgetting,
        "cndids": CNDIDSAntiForgetting,
    },
    "anomaly_scorer": {
        "cade_mad": CADEMADScorer,
        "none": NoAnomalyScorer,
        "pca": PCAScorer,
    },
}


def build(slot: str, name: str, **kwargs):
    """레지스트리에서 컴포넌트를 생성한다.

    Args:
        slot: 슬롯 이름 (예: 'drift_detector').
        name: 슬롯 내 컴포넌트 키 (예: 'ssf').
        **kwargs: 생성자에 전달할 키워드 인자(불필요한 키는 생성자 시그니처에
                  맞춰 자동으로 걸러진다).

    Returns:
        생성된 컴포넌트 인스턴스.

    Raises:
        KeyError: slot 또는 name이 등록되어 있지 않은 경우.
    """
    if slot not in REGISTRY:
        raise KeyError(f"Unknown slot: {slot!r}. Available: {list(REGISTRY)}")
    slot_map = REGISTRY[slot]
    if name not in slot_map:
        raise KeyError(
            f"Unknown component {name!r} in slot {slot!r}. Available: {list(slot_map)}")
    cls = slot_map[name]
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters) - {"self"}
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}
    return cls(**filtered)
