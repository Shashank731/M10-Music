"""Hybrid recommendation strategies."""

from __future__ import annotations

from dataclasses import dataclass

from music_recommendation.recommender.collaborative import CollaborativeRecommender
from music_recommendation.recommender.content import ContentRecommender


@dataclass
class HybridRecommender:
    """Blend content and collaborative candidate lists."""

    content: ContentRecommender
    collaborative: CollaborativeRecommender
    content_weight: float = 0.5

    def recommend_for_song(self, track_id: str, top_k: int = 10) -> list[dict]:
        """Blend similar songs from both content and collaborative item signals."""
        content_rows = self.content.recommend(track_id, top_k=top_k)
        collab_rows = self.collaborative.similar_items(track_id, top_k=top_k)
        return self._merge_ranked(content_rows, collab_rows, top_k)

    def recommend_for_user(
        self, user_id: str, seed_track_id: str | None = None, top_k: int = 10
    ) -> list[dict]:
        """Recommend for a user, optionally nudged by a seed song."""
        collab_rows = self.collaborative.recommend(user_id, top_k=top_k)
        if seed_track_id is None:
            return collab_rows[:top_k]
        content_rows = self.content.recommend(seed_track_id, top_k=top_k)
        return self._merge_ranked(content_rows, collab_rows, top_k)

    def _merge_ranked(
        self, content_rows: list[dict], collab_rows: list[dict], top_k: int
    ) -> list[dict]:
        scores: dict[str, float] = {}
        rows: dict[str, dict] = {}
        for rank, row in enumerate(content_rows):
            track_id = str(row["track_id"])
            rows[track_id] = row
            scores[track_id] = scores.get(track_id, 0.0) + self.content_weight / (
                rank + 1
            )
        for rank, row in enumerate(collab_rows):
            track_id = str(row["track_id"])
            rows[track_id] = row
            scores[track_id] = scores.get(track_id, 0.0) + (
                1.0 - self.content_weight
            ) / (rank + 1)
        ordered_ids = sorted(scores, key=scores.get, reverse=True)
        return [rows[track_id] for track_id in ordered_ids[:top_k]]
