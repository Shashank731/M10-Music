"""Content-based song recommendation using notebook-derived embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from music_recommendation.vector_store.faiss_store import VectorStore


@dataclass
class ContentRecommender:
    """Serve similar-song recommendations from precomputed embeddings."""

    songs: pd.DataFrame
    embeddings: np.ndarray
    vector_store: VectorStore

    def __post_init__(self) -> None:
        if "track_id" not in self.songs.columns:
            raise ValueError("songs must include track_id")
        self.songs = self.songs.reset_index(drop=True)
        self.songs = self.songs.replace({np.nan: None})
        self.track_to_row = {
            str(track_id): idx for idx, track_id in enumerate(self.songs["track_id"])
        }
        if not getattr(self.vector_store, "ids", []):
            self.vector_store.add(
                self.embeddings,
                [str(track_id) for track_id in self.songs["track_id"]],
            )

    def recommend(self, track_id: str, top_k: int = 10) -> list[dict]:
        """Return top-k songs similar to a given track."""
        if track_id not in self.track_to_row:
            return []

        row_idx = self.track_to_row[track_id]
        results = self.vector_store.search(self.embeddings[row_idx], top_k + 1)
        ids = [song_id for song_id, _ in results if song_id != track_id][:top_k]
        records = self.songs[self.songs["track_id"].astype(str).isin(ids)].to_dict(
            orient="records"
        )
        by_id = {str(row["track_id"]): row for row in records}
        return [by_id[song_id] for song_id in ids if song_id in by_id]

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search songs by name, artist, or tag text."""
        if not query:
            return self.songs.head(limit).to_dict(orient="records")
        mask = (
            self.songs.get("name", pd.Series("", index=self.songs.index))
            .fillna("")
            .str.contains(query, case=False, na=False)
            | self.songs.get("artist", pd.Series("", index=self.songs.index))
            .fillna("")
            .str.contains(query, case=False, na=False)
            | self.songs.get("tags", pd.Series("", index=self.songs.index))
            .fillna("")
            .str.contains(query, case=False, na=False)
        )
        return self.songs.loc[mask].head(limit).to_dict(orient="records")

    def get_song(self, track_id: str) -> dict | None:
        """Return song metadata by track ID."""
        matches = self.songs[self.songs["track_id"].astype(str) == track_id]
        if matches.empty:
            return None
        return matches.iloc[0].to_dict()
