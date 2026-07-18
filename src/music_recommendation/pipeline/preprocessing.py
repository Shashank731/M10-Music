"""Preprocessing routines extracted from the research notebooks."""

from __future__ import annotations

import pandas as pd


class SongPreprocessor:
    """Clean song metadata while preserving notebook behavior."""

    def __init__(
        self,
        duplicate_subset: list[str] | None = None,
        tag_fill_value: str = "no_tags",
    ) -> None:
        self.duplicate_subset = duplicate_subset or ["spotify_id"]
        self.tag_fill_value = tag_fill_value

    def transform(self, songs: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicate songs, fill tags, and reset the row index."""
        clean = songs.copy()
        subset = [col for col in self.duplicate_subset if col in clean.columns]
        if subset:
            clean = clean.drop_duplicates(subset=subset)
        if "tags" in clean.columns:
            clean["tags"] = clean["tags"].fillna(self.tag_fill_value)
        return clean.reset_index(drop=True)


class InteractionPreprocessor:
    """Prepare listening history for sparse collaborative filtering."""

    REQUIRED_COLUMNS = {"track_id", "user_id", "playcount"}

    def transform(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Validate, type-cast, and aggregate user-track playcounts."""
        missing = self.REQUIRED_COLUMNS.difference(interactions.columns)
        if missing:
            raise ValueError(f"Missing interaction columns: {sorted(missing)}")

        clean = interactions.loc[:, ["track_id", "user_id", "playcount"]].copy()
        clean["playcount"] = clean["playcount"].astype(float)
        return (
            clean.groupby(["user_id", "track_id"], as_index=False)["playcount"]
            .sum()
            .reset_index(drop=True)
        )
