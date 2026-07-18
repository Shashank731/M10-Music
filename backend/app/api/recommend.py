"""Recommendation API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, Depends, status

from backend.app.schemas import RecommendationResponse
from backend.app.auth import get_current_user

router = APIRouter(prefix="/recommend", tags=["recommendations"])


@router.get("/song/{song_id}", response_model=RecommendationResponse)
def recommend_by_song(
    song_id: str,
    request: Request,
    top_k: int = Query(default=10, ge=1, le=100),
    _current_user: str = Depends(get_current_user),
) -> RecommendationResponse:
    """Return content-based recommendations for a seed song."""
    try:
        recommender = request.app.state.content_service.load()
        recommendations = recommender.recommend(song_id, top_k=top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return RecommendationResponse(query_id=song_id, recommendations=recommendations)


@router.get("/user/{user_id}", response_model=RecommendationResponse)
def recommend_by_user(
    user_id: str,
    request: Request,
    top_k: int = Query(default=10, ge=1, le=100),
    current_user: str = Depends(get_current_user),
) -> RecommendationResponse:
    """Return collaborative recommendations for a user.

    This route requires authentication. The authenticated user's email (subject)
    must match the requested user_id to obtain recommendations for that user.
    """
    if user_id.lower() != current_user.lower():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not allowed to request recommendations for this user",
        )
    try:
        recommender = request.app.state.collaborative_service.load()
        recommendations = recommender.recommend(user_id, top_k=top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return RecommendationResponse(query_id=user_id, recommendations=recommendations)
