"""Hybrid recommender service loader."""

from __future__ import annotations

from music_recommendation.recommender.hybrid import HybridRecommender

from backend.app.services.collaborative_service import CollaborativeService
from backend.app.services.content_service import ContentService


class HybridService:
    """Create a hybrid recommender from trained base services."""

    def __init__(
        self,
        content_service: ContentService,
        collaborative_service: CollaborativeService,
        content_weight: float = 0.5,
    ) -> None:
        self.content_service = content_service
        self.collaborative_service = collaborative_service
        self.content_weight = content_weight

    def load(self) -> HybridRecommender:
        """Load the hybrid recommender."""
        return HybridRecommender(
            content=self.content_service.load(),
            collaborative=self.collaborative_service.load(),
            content_weight=self.content_weight,
        )
