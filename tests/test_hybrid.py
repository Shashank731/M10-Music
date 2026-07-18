import numpy as np
import pandas as pd

from music_recommendation.pipeline.feature_engineering import InteractionFeatureEngineer
from music_recommendation.pipeline.preprocessing import InteractionPreprocessor
from music_recommendation.recommender.collaborative import CollaborativeRecommender
from music_recommendation.recommender.content import ContentRecommender
from music_recommendation.recommender.hybrid import HybridRecommender
from music_recommendation.vector_store.faiss_store import FaissVectorStore


def test_hybrid_recommender_merges_ranked_results():
    songs = pd.DataFrame(
        [
            {"track_id": "a", "name": "Alpha"},
            {"track_id": "b", "name": "Beta"},
            {"track_id": "c", "name": "Gamma"},
        ]
    )
    content = ContentRecommender(
        songs,
        np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype="float32"),
        FaissVectorStore(),
    )
    interactions = InteractionPreprocessor().transform(
        pd.DataFrame(
            [
                {"user_id": "u1", "track_id": "a", "playcount": 1},
                {"user_id": "u2", "track_id": "c", "playcount": 5},
            ]
        )
    )
    collaborative = CollaborativeRecommender(
        InteractionFeatureEngineer().fit_transform(interactions), songs
    )

    results = HybridRecommender(content, collaborative).recommend_for_song("a", top_k=2)

    assert results
    assert {row["track_id"] for row in results}.issubset({"b", "c"})
