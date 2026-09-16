"""전역 RNG 오염 격리(`design_decisions.md` 2절)에 쓰는 파생 시드 계산.

**2026-09-14 재검토로 발견·수정**: 처음엔 각 컴포넌트가 `base_seed + 고정
오프셋 + 카운터`(오프셋끼리 100 간격) 방식을 썼다. 그런데 `GPMAntiForgetting`
의 자기학습 카운터(`_self_training_call_count`)는 라운드가 아니라 **매
미니배치마다** 증가하고 라운드 사이에도 리셋되지 않는다 — Track A 실전
설정(`epochs_per_experience=200`)에서는 한 콤보 안에서 이 카운터가 수백~
수천까지 쉽게 도달해, 100 간격으로 떼어둔 다른 컴포넌트(SPIDER/SSF/
CND-IDS 메모리 매니저)의 오프셋 구간을 실제로 침범한다. `fork_rng()`가
바깥 RNG 스트림을 오염시키는 걸 막는다는 핵심 목적 자체는 여전히
지켜지지만(그 침범이 컴포넌트 사이의 격리를 깨지는 않는다 — 각자 독립된
fork_rng 블록이므로), 서로 다른 컴포넌트가 우연히 같은 파생 시드를 받는
건 "각 컴포넌트가 서로 독립적인 난수열을 쓴다"는 설계 의도에 어긋나는
불필요한 결함이다.

정수 오프셋 방식을 SHA-256 해시 기반으로 교체해 이 문제를 근본적으로
없앤다 — `tag` 문자열이 다르면(컴포넌트마다 고유 문자열을 쓴다) 카운터
값이 아무리 커져도 실질적으로 절대 충돌하지 않는다.

Python 내장 `hash()`는 문자열에 대해 `PYTHONHASHSEED`에 따라 프로세스마다
달라져(보안을 위한 해시 무작위화) 재현성이 깨진다 — 반드시 `hashlib`로
직접 계산해야 한다.
"""

import hashlib


def derived_seed(base_seed: int, tag: str, counter: int = 0) -> int:
    """`base_seed`로부터 `(tag, counter)` 조합별로 결정론적이고(같은 입력
    이면 항상 같은 출력) 서로 다른 tag 사이에서는 사실상 충돌하지 않는
    파생 시드를 만든다. `torch.manual_seed()`에 바로 넘길 수 있다.

    Args:
        base_seed: 전역 설정 seed(`global_hparams.yaml`의 `seed`).
        tag: 컴포넌트+용도를 가리키는 고유 문자열(예: "cade_encoder_init",
            "gpm_self_training"). 서로 다른 소비처는 반드시 서로 다른
            tag를 써야 한다.
        counter: 같은 tag 안에서 호출마다(라운드/미니배치 등) 다른 값을
            받기 위한 정수. 기본 0(1회성 소비처, 예: 생성자에서의 가중치
            초기화).
    """
    key = f"{base_seed}:{tag}:{counter}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)
