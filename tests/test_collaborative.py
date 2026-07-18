import pandas as pd

from music_recommendation.pipeline.feature_engineering import InteractionFeatureEngineer
from music_recommendation.pipeline.preprocessing import InteractionPreprocessor
from music_recommendation.recommender.collaborative import CollaborativeRecommender


def test_collaborative_recommender_recommends_unseen_popular_track():
    songs = pd.DataFrame(
        [
            {"track_id": "a", "name": "Alpha"},
            {"track_id": "b", "name": "Beta"},
            {"track_id": "c", "name": "Gamma"},
        ]
    )
    interactions = pd.DataFrame(
        [
            {"user_id": "u1", "track_id": "a", "playcount": 1},
            {"user_id": "u2", "track_id": "b", "playcount": 5},
            {"user_id": "u3", "track_id": "b", "playcount": 2},
            {"user_id": "u4", "track_id": "c", "playcount": 1},
        ]
    )
    clean = InteractionPreprocessor().transform(interactions)
    matrix = InteractionFeatureEngineer().fit_transform(clean)
    recommender = CollaborativeRecommender(matrix, songs)

    results = recommender.recommend("u1", top_k=1)

    assert results[0]["track_id"] == "b"
