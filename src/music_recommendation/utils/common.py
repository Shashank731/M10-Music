"""Common filesystem and serialization helpers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root from the installed package location."""
    return Path(__file__).resolve().parents[3]


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file, returning an empty dict for empty files."""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Persist a JSON object with stable formatting."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_pickle(obj: Any, path: str | Path) -> None:
    """Serialize a Python object with pickle."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("wb") as file:
        pickle.dump(obj, file)


def load_pickle(path: str | Path) -> Any:
    """Load a pickle artifact."""
    with Path(path).open("rb") as file:
        return pickle.load(file)
