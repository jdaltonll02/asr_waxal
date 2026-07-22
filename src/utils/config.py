from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item) for item in value]
    return value


def resolve_experiment(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    experiment = load_yaml(path)
    root = path.parents[2]
    data = load_yaml(root / experiment["data_config"])
    model = load_yaml(root / experiment["model_config"])
    resolved = copy.deepcopy(experiment)
    resolved["data"] = data
    resolved["model"] = model
    # Per-experiment language capability/tags intentionally override shared model
    # cards without duplicating a model YAML per language.
    for key in ("supports_language", "language_tag"):
        if key in experiment:
            resolved["model"][key] = experiment[key]
    return expand(resolved)


def write_resolved_config(config: dict[str, Any], output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with open(destination / "resolved_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
