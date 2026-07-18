"""Feature engineering for content and collaborative recommenders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler, normalize

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional production dependency
    SentenceTransformer = None  # type: ignore[assignment]


@dataclass(frozen=True)
class InteractionMatrix:
    """Sparse matrix plus ID mappings for collaborative filtering."""

    matrix: csr_matrix
    user_to_idx: dict[str, int]
    idx_to_user: dict[int, str]
    track_to_idx: dict[str, int]
    idx_to_track: dict[int, str]


class ContentFeatureEngineer:
    """Build notebook-style text and audio embeddings for FAISS search."""

    AUDIO_STANDARD_COLUMNS = ["duration_ms", "loudness", "tempo"]
    AUDIO_MINMAX_COLUMNS = [
        "danceability",
        "energy",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "year",
    ]
    AUDIO_OHE_COLUMNS = ["time_signature", "key"]

    def __init__(
        self,
        text_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        text_weight: float = 0.2,
        audio_weight: float = 0.8,
        fallback_text_features: int = 384,
    ) -> None:
        self.text_model_name = text_model_name
        self.text_weight = text_weight
        self.audio_weight = audio_weight
        self.fallback_text_features = fallback_text_features
        self.audio_transformer: ColumnTransformer | None = None

    def _song_text(self, songs: pd.DataFrame) -> list[str]:
        artists = songs.get("artist", pd.Series("", index=songs.index)).fillna("")
        tags = songs.get("tags", pd.Series("no_tags", index=songs.index)).fillna(
            "no_tags"
        )
        return ("Artist: " + artists + ". Tags: " + tags).tolist()

    def _text_embeddings(self, texts: list[str]) -> np.ndarray:
        if SentenceTransformer is not None:
            model = SentenceTransformer(self.text_model_name)
            return model.encode(texts, batch_size=64, show_progress_bar=True)

        vectorizer = HashingVectorizer(
            n_features=self.fallback_text_features,
            alternate_sign=False,
            norm=None,
        )
        return vectorizer.transform(texts).toarray()

    def _audio_frame(self, songs: pd.DataFrame) -> pd.DataFrame:
        columns = (
            self.AUDIO_STANDARD_COLUMNS
            + self.AUDIO_MINMAX_COLUMNS
            + self.AUDIO_OHE_COLUMNS
        )
        available = [col for col in columns if col in songs.columns]
        audio = songs.loc[:, available].copy()
        for column in available:
            audio[column] = audio[column].fillna(audio[column].mode().iloc[0])
        return audio

    def fit_transform(self, songs: pd.DataFrame) -> np.ndarray:
        """Create final float32 embeddings from notebook text/audio features."""
        audio = self._audio_frame(songs)
        standard_cols = [c for c in self.AUDIO_STANDARD_COLUMNS if c in audio.columns]
        minmax_cols = [c for c in self.AUDIO_MINMAX_COLUMNS if c in audio.columns]
        ohe_cols = [c for c in self.AUDIO_OHE_COLUMNS if c in audio.columns]

        self.audio_transformer = ColumnTransformer(
            transformers=[
                ("ohe", OneHotEncoder(handle_unknown="ignore"), ohe_cols),
                ("standard_scale", StandardScaler(), standard_cols),
                ("min_max_scale", MinMaxScaler(), minmax_cols),
            ],
            remainder="drop",
        )
        audio_features = self.audio_transformer.fit_transform(audio)
        audio_features = normalize(audio_features, norm="l2")
        text_embeddings = normalize(self._text_embeddings(self._song_text(songs)), norm="l2")
        embeddings = np.concatenate(
            [text_embeddings * self.text_weight, audio_features * self.audio_weight],
            axis=1,
        )
        return embeddings.astype("float32")


class InteractionFeatureEngineer:
    """Convert listening events to a user-item sparse matrix."""

    def fit_transform(self, interactions: pd.DataFrame) -> InteractionMatrix:
        """Build a CSR matrix equivalent to the notebook interaction array."""
        users = pd.Index(interactions["user_id"].astype(str).unique())
        tracks = pd.Index(interactions["track_id"].astype(str).unique())
        user_to_idx = {user: idx for idx, user in enumerate(users)}
        track_to_idx = {track: idx for idx, track in enumerate(tracks)}

        user_indices = interactions["user_id"].astype(str).map(user_to_idx)
        track_indices = interactions["track_id"].astype(str).map(track_to_idx)
        matrix = csr_matrix(
            (
                interactions["playcount"].astype(float),
                (user_indices, track_indices),
            ),
            shape=(len(users), len(tracks)),
        )

        return InteractionMatrix(
            matrix=matrix,
            user_to_idx=user_to_idx,
            idx_to_user={idx: user for user, idx in user_to_idx.items()},
            track_to_idx=track_to_idx,
            idx_to_track={idx: track for track, idx in track_to_idx.items()},
        )
