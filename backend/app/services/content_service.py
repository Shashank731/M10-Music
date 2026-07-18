"""Content recommender service loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from music_recommendation.recommender.content import ContentRecommender
from music_recommendation.utils.common import load_pickle
from music_recommendation.vector_store.faiss_store import FaissVectorStore


class ContentService:
    """Lazy-loading content recommendation service."""

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        self._recommender: ContentRecommender | None = None

    def load(self) -> ContentRecommender:
        """Load content artifacts from disk."""
        if self._recommender is not None:
            return self._recommender

        songs_path = self.artifact_dir / "songs.pkl"
        embeddings_path = self.artifact_dir / "embeddings.npy"
        ids_path = self.artifact_dir / "track_ids.pkl"
        index_path = self.artifact_dir / "faiss.index"
        missing = [
            str(path)
            for path in (songs_path, embeddings_path, ids_path)
            if not path.exists()
        ]
        if missing:
            raise RuntimeError(
                "Content artifacts are missing. Run "
                "`python -m music_recommendation.pipeline.content_training`. "
                f"Missing: {missing}"
            )

        songs: pd.DataFrame = load_pickle(songs_path)
        embeddings = np.load(embeddings_path)
        ids = load_pickle(ids_path)
        try:
            store = (
                FaissVectorStore.load(index_path, ids)
                if index_path.exists()
                else FaissVectorStore()
            )
        except ImportError:
            store = FaissVectorStore()
        self._recommender = ContentRecommender(songs, embeddings, store)
        return self._recommender
