"""Pydantic response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Song(BaseModel):
    """Song metadata response."""

    track_id: str
    name: str | None = None
    artist: str | None = None
    spotify_preview_url: str | None = None
    spotify_id: str | None = None
    tags: str | None = None
    genre: str | None = None
    year: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    """Recommendation API response."""

    query_id: str
    recommendations: list[dict[str, Any]]


class SearchResponse(BaseModel):
    """Search API response."""

    query: str
    results: list[dict[str, Any]]
