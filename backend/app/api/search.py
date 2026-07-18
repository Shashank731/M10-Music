"""Search API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from backend.app.auth import get_current_user
from backend.app.schemas import SearchResponse

router = APIRouter(tags=["search"])


@router.get("/search", response_model=SearchResponse)
def search(
    request: Request,
    q: str = Query(default=""),
    limit: int = Query(default=10, ge=1, le=100),
    _current_user: str = Depends(get_current_user),
) -> SearchResponse:
    """Search indexed song metadata."""
    try:
        results = request.app.state.content_service.load().search(q, limit=limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return SearchResponse(query=q, results=results)
