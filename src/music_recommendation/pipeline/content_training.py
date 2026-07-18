"""Train content embeddings and FAISS artifacts."""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np

from music_recommendation.pipeline.data_ingestion import MusicDataIngestion
from music_recommendation.pipeline.feature_engineering import ContentFeatureEngineer
from music_recommendation.pipeline.preprocessing import SongPreprocessor
from music_recommendation.utils.common import ensure_dir, load_yaml, save_pickle
from music_recommendation.utils.logger import get_logger
from music_recommendation.vector_store.faiss_store import FaissVectorStore

LOGGER = get_logger(__name__)


class ContentTrainingPipeline:
    """Build content embeddings and a vector index from song metadata."""

    def __init__(self, config_path: str | Path = "configs/config.yaml") -> None:
        self.config = load_yaml(config_path)

    def run(self) -> dict[str, Path]:
        """Execute the content training pipeline and log MLflow artifacts."""
        data_cfg = self.config["data"]
        model_cfg = self.config["model"]["content"]
        artifact_cfg = self.config["artifacts"]

        output_dir = ensure_dir(artifact_cfg["content_dir"])
        songs = MusicDataIngestion(data_cfg["dataset_dir"]).load_songs()
        songs = SongPreprocessor(duplicate_subset=["spotify_id"]).transform(songs)

        engineer = ContentFeatureEngineer(
            text_model_name=model_cfg["text_model_name"],
            text_weight=float(model_cfg["text_weight"]),
            audio_weight=float(model_cfg["audio_weight"]),
        )

        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
        with mlflow.start_run(run_name="content_recommender"):
            mlflow.log_params(model_cfg)
            embeddings = engineer.fit_transform(songs)
            store = FaissVectorStore()
            ids = songs["track_id"].astype(str).tolist()
            store.add(embeddings, ids)

            songs_path = output_dir / "songs.pkl"
            embeddings_path = output_dir / "embeddings.npy"
            transformer_path = output_dir / "audio_transformer.pkl"
            ids_path = output_dir / "track_ids.pkl"
            index_path = output_dir / "faiss.index"

            save_pickle(songs, songs_path)
            np.save(embeddings_path, embeddings)
            save_pickle(engineer.audio_transformer, transformer_path)
            save_pickle(ids, ids_path)
            store.save(index_path)

            mlflow.log_metric("num_songs", len(songs))
            mlflow.log_metric("embedding_dim", embeddings.shape[1])
            for path in (songs_path, embeddings_path, transformer_path, ids_path):
                mlflow.log_artifact(str(path), artifact_path="content")
            if index_path.exists():
                mlflow.log_artifact(str(index_path), artifact_path="content")

        LOGGER.info("Content artifacts written to %s", output_dir)
        return {
            "songs": songs_path,
            "embeddings": embeddings_path,
            "index": index_path,
            "ids": ids_path,
        }


if __name__ == "__main__":
    ContentTrainingPipeline().run()
