"""Hybrid pipeline placeholder for combining trained content and ALS artifacts."""

from __future__ import annotations

from pathlib import Path

from music_recommendation.utils.common import load_yaml


class HybridTrainingPipeline:
    """Validate hybrid configuration after base recommenders are trained."""

    def __init__(self, config_path: str | Path = "configs/config.yaml") -> None:
        self.config = load_yaml(config_path)

    def run(self) -> dict[str, float]:
        """Return the configured hybrid weights."""
        return {"content_weight": float(self.config["model"]["hybrid"]["content_weight"])}


if __name__ == "__main__":
    HybridTrainingPipeline().run()
