from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "storm_kit").is_dir() and (parent / "examples").is_dir():
            return parent
    raise RuntimeError("Could not locate repository root from static_compare")


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root() / path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> Any:
    with open(resolve_repo_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str | Path) -> Any:
    with open(resolve_repo_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def write_json(path: str | Path, payload: Any) -> None:
    path = resolve_repo_path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def project_path(path: str | Path) -> str:
    path = resolve_repo_path(path)
    try:
        return str(path.relative_to(repo_root()))
    except ValueError:
        return str(path)


def nan() -> float:
    return float("nan")
