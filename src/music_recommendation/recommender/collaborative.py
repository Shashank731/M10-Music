"""Collaborative filtering recommender built from notebook sparse matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from music_recommendation.pipeline.feature_engineering import InteractionMatrix

try:
    from implicit.als import AlternatingLeastSquares
except ImportError:  # pragma: no cover - optional dependency
    AlternatingLeastSquares = None  # type: ignore[assignment]


@dataclass
class CollaborativeRecommender:
    """ALS recommender with a popularity fallback for lightweight runtime."""

    interaction_matrix: InteractionMatrix
    songs: pd.DataFrame
    factors: int = 64
    regularization: float = 0.01
    iterations: int = 20
    alpha: float = 40.0

    def __post_init__(self) -> None:
        self.model = None
        self.track_popularity = np.asarray(
            self.interaction_matrix.matrix.sum(axis=0)
        ).ravel()

    def fit(self) -> CollaborativeRecommender:
        """Fit ALS when available; otherwise keep popularity fallback."""
        if AlternatingLeastSquares is None:
            return self
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
        )
        item_user = (self.interaction_matrix.matrix.T * self.alpha).astype("double")
        self.model.fit(item_user)
        return self

    def recommend(self, user_id: str, top_k: int = 10) -> list[dict]:
        """Recommend tracks for a user."""
        user_idx = self.interaction_matrix.user_to_idx.get(str(user_id))
        if user_idx is None:
            return self._tracks_to_records(self._popular_track_ids(top_k))

        if self.model is not None:
            ids, _ = self.model.recommend(
                user_idx,
                self.interaction_matrix.matrix[user_idx],
                N=top_k,
                filter_already_liked_items=True,
            )
            track_ids = [
                self.interaction_matrix.idx_to_track[int(track_idx)]
                for track_idx in ids
            ]
            return self._tracks_to_records(track_ids)

        seen = set(self.interaction_matrix.matrix[user_idx].indices)
        return self._tracks_to_records(self._popular_track_ids(top_k, seen))

    def similar_items(self, track_id: str, top_k: int = 10) -> list[dict]:
        """Return collaborative item-neighbor recommendations."""
        track_idx = self.interaction_matrix.track_to_idx.get(str(track_id))
        if track_idx is None:
            return []
        if self.model is not None:
            ids, _ = self.model.similar_items(track_idx, N=top_k + 1)
            track_ids = [
                self.interaction_matrix.idx_to_track[int(idx)]
                for idx in ids
                if int(idx) != track_idx
            ][:top_k]
            return self._tracks_to_records(track_ids)
        matrix: csr_matrix = self.interaction_matrix.matrix
        item_vectors = matrix.T
        target = item_vectors[track_idx]
        scores = item_vectors @ target.T
        scores = np.asarray(scores.todense()).ravel()
        order = np.argsort(scores)[::-1]
        ids = [
            self.interaction_matrix.idx_to_track[int(idx)]
            for idx in order
            if int(idx) != track_idx
        ][:top_k]
        return self._tracks_to_records(ids)

    def _popular_track_ids(
        self, top_k: int, exclude_indices: set[int] | None = None
    ) -> list[str]:
        exclude_indices = exclude_indices or set()
        order = np.argsort(self.track_popularity)[::-1]
        return [
            self.interaction_matrix.idx_to_track[int(idx)]
            for idx in order
            if int(idx) not in exclude_indices
        ][:top_k]

    def _tracks_to_records(self, track_ids: list[str]) -> list[dict]:
        records = self.songs[
            self.songs["track_id"].astype(str).isin([str(track) for track in track_ids])
        ].to_dict(orient="records")
        by_id = {str(row["track_id"]): row for row in records}
        return [by_id[track_id] for track_id in track_ids if track_id in by_id]
