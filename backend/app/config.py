"""Backend settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from music_recommendation.utils.common import load_yaml, project_root


@dataclass(frozen=True)
class Settings:
    """FastAPI runtime settings."""

    config_path: Path
    dataset_dir: Path
    content_dir: Path
    collaborative_dir: Path
    cors_origins: list[str]
    jwt_secret: str


def get_settings() -> Settings:
    """Load settings from YAML and environment overrides."""
    root = project_root()
    config_path = Path(os.getenv("APP_CONFIG", root / "configs" / "config.yaml"))
    config: dict[str, Any] = load_yaml(config_path)
    jwt_secret = os.getenv(
        "APP_SECRET",
        config.get(
            "backend",
            {},
        ).get(
            "jwt_secret",
            "dev-secret",
        ),
    )
    return Settings(
        config_path=config_path,
        dataset_dir=root / config["data"]["dataset_dir"],
        content_dir=root / config["artifacts"]["content_dir"],
        collaborative_dir=root / config["artifacts"]["collaborative_dir"],
        cors_origins=config.get("backend", {}).get("cors_origins", ["*"]),
        jwt_secret=jwt_secret,
    )
