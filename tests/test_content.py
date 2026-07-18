import numpy as np
import pandas as pd

from music_recommendation.recommender.content import ContentRecommender
from music_recommendation.vector_store.faiss_store import FaissVectorStore


def test_content_recommender_returns_similar_song():
    songs = pd.DataFrame(
        [
            {"track_id": "a", "name": "Alpha", "artist": "One", "tags": "rock"},
            {"track_id": "b", "name": "Beta", "artist": "Two", "tags": "rock"},
            {"track_id": "c", "name": "Gamma", "artist": "Three", "tags": "jazz"},
        ]
    )
    embeddings = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
        dtype="float32",
    )
    recommender = ContentRecommender(songs, embeddings, FaissVectorStore())

    results = recommender.recommend("a", top_k=1)

    assert len(results) == 1
    assert results[0]["track_id"] == "b"


def test_content_search_matches_artist():
    songs = pd.DataFrame(
        [{"track_id": "a", "name": "Alpha", "artist": "The Searchers"}]
    )
    embeddings = np.array([[1.0, 0.0]], dtype="float32")
    recommender = ContentRecommender(songs, embeddings, FaissVectorStore())

    assert recommender.search("searchers")[0]["track_id"] == "a"
