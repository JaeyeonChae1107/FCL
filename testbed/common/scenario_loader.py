"""시나리오 로더 — PRD 9절."""

import io
from typing import Any, Dict

import yaml

_REQUIRED_FIELDS = [
    "scenario_name", "dataset", "experience_definition",
    "labeling_budget", "supervision", "seed",
]


def load_scenario(path: str) -> Dict[str, Any]:
    with io.open(path, encoding="utf-8") as f:
        scenario = yaml.safe_load(f)
    missing = [k for k in _REQUIRED_FIELDS if k not in scenario]
    if missing:
        raise ValueError(f"Scenario {path!r} missing required fields: {missing}")
    return scenario
