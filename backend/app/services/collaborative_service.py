"""Collaborative recommender service loader."""

from __future__ import annotations

from pathlib import Path

from music_recommendation.recommender.collaborative import CollaborativeRecommender
from music_recommendation.utils.common import load_pickle


class CollaborativeService:
    """Lazy-loading collaborative recommendation service."""

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        self._recommender: CollaborativeRecommender | None = None

    def load(self) -> CollaborativeRecommender:
        """Load collaborative artifacts from disk."""
        if self._recommender is not None:
            return self._recommender
        model_path = self.artifact_dir / "collaborative_recommender.pkl"
        if not model_path.exists():
            raise RuntimeError(
                "Collaborative artifacts are missing. Run "
                "`python -m music_recommendation.pipeline.als_training`."
            )
        self._recommender = load_pickle(model_path)
        return self._recommender
