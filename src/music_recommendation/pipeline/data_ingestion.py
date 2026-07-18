"""Data ingestion for Million Song Dataset + Spotify + Last.fm files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from music_recommendation.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class DatasetPaths:
    """Resolved source CSV paths used by the pipeline."""

    songs: Path
    listening_history: Path


class MusicDataIngestion:
    """Load the two CSV files used by the initial recommendation system."""

    SONG_FILE = "Music Info.csv"
    HISTORY_FILE = "User Listening History.csv"

    def __init__(self, dataset_dir: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir)

    @property
    def paths(self) -> DatasetPaths:
        """Return canonical dataset file paths."""
        return DatasetPaths(
            songs=self.dataset_dir / self.SONG_FILE,
            listening_history=self.dataset_dir / self.HISTORY_FILE,
        )

    def validate(self) -> None:
        """Raise FileNotFoundError if required files are missing."""
        for path in (self.paths.songs, self.paths.listening_history):
            if not path.exists():
                raise FileNotFoundError(f"Required dataset file not found: {path}")

    def load_songs(self, usecols: list[str] | None = None) -> pd.DataFrame:
        """Load song metadata."""
        self.validate()
        LOGGER.info("Loading song metadata from %s", self.paths.songs)
        return pd.read_csv(self.paths.songs, usecols=usecols, encoding="utf-8-sig")

    def load_listening_history(
        self, usecols: list[str] | None = None
    ) -> pd.DataFrame:
        """Load user listening events."""
        self.validate()
        LOGGER.info("Loading listening history from %s", self.paths.listening_history)
        return pd.read_csv(
            self.paths.listening_history,
            usecols=usecols,
            encoding="utf-8-sig",
        )

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load both dataset tables."""
        return self.load_songs(), self.load_listening_history()
