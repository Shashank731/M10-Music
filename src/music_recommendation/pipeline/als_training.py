"""Train collaborative filtering artifacts."""

from __future__ import annotations

from pathlib import Path

import mlflow

from music_recommendation.pipeline.data_ingestion import MusicDataIngestion
from music_recommendation.pipeline.feature_engineering import InteractionFeatureEngineer
from music_recommendation.pipeline.preprocessing import (
    InteractionPreprocessor,
    SongPreprocessor,
)
from music_recommendation.recommender.collaborative import CollaborativeRecommender
from music_recommendation.utils.common import ensure_dir, load_yaml, save_pickle
from music_recommendation.utils.logger import get_logger

LOGGER = get_logger(__name__)


class ALSTrainingPipeline:
    """Build ALS-style collaborative filtering artifacts."""

    def __init__(self, config_path: str | Path = "configs/config.yaml") -> None:
        self.config = load_yaml(config_path)

    def run(self) -> dict[str, Path]:
        """Execute collaborative training and log MLflow artifacts."""
        data_cfg = self.config["data"]
        model_cfg = self.config["model"]["collaborative"]
        artifact_cfg = self.config["artifacts"]

        output_dir = ensure_dir(artifact_cfg["collaborative_dir"])
        ingestion = MusicDataIngestion(data_cfg["dataset_dir"])
        songs, interactions = ingestion.load()
        songs = SongPreprocessor(duplicate_subset=["spotify_id"]).transform(songs)
        interactions = InteractionPreprocessor().transform(interactions)
        matrix = InteractionFeatureEngineer().fit_transform(interactions)

        recommender = CollaborativeRecommender(
            interaction_matrix=matrix,
            songs=songs,
            factors=int(model_cfg["factors"]),
            regularization=float(model_cfg["regularization"]),
            iterations=int(model_cfg["iterations"]),
            alpha=float(model_cfg["alpha"]),
        ).fit()

        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
        with mlflow.start_run(run_name="collaborative_recommender"):
            mlflow.log_params(model_cfg)
            model_path = output_dir / "collaborative_recommender.pkl"
            matrix_path = output_dir / "interaction_matrix.pkl"
            save_pickle(recommender, model_path)
            save_pickle(matrix, matrix_path)

            mlflow.log_metric("num_users", matrix.matrix.shape[0])
            mlflow.log_metric("num_tracks", matrix.matrix.shape[1])
            mlflow.log_metric("num_interactions", matrix.matrix.nnz)
            mlflow.log_metric(f"precision_at_{model_cfg['top_k']}", 0.0)
            mlflow.log_metric(f"recall_at_{model_cfg['top_k']}", 0.0)
            mlflow.log_metric(f"map_at_{model_cfg['top_k']}", 0.0)
            mlflow.log_metric(f"ndcg_at_{model_cfg['top_k']}", 0.0)
            mlflow.log_artifact(str(model_path), artifact_path="collaborative")
            mlflow.log_artifact(str(matrix_path), artifact_path="collaborative")

        LOGGER.info("Collaborative artifacts written to %s", output_dir)
        return {"model": model_path, "matrix": matrix_path}


if __name__ == "__main__":
    ALSTrainingPipeline().run()
