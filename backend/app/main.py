"""FastAPI application for serving trained recommendation artifacts."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

"""
Support both execution contexts:

- When the project root is on PYTHONPATH
  (for example, run from the project root),
  use absolute imports.

- When running inside backend/app or as a module,
  fall back to relative imports.
"""
try:
    # Absolute imports (works when project root is the working directory)
    from backend.app import auth
    from backend.app.api import recommend, search, song
    from backend.app.config import get_settings
    from backend.app.services.collaborative_service import CollaborativeService
    from backend.app.services.content_service import ContentService
except Exception:
    # Relative imports (works when running as a module from backend/app)
    from . import auth
    from .api import recommend, search, song
    from .config import get_settings
    from .services.collaborative_service import CollaborativeService
    from .services.content_service import ContentService


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""
    settings = get_settings()
    app = FastAPI(
        title="Music Recommendation API",
        version="0.1.0",
        description="Serving layer for trained content and ALS recommenders.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.content_service = ContentService(settings.content_dir)
    app.state.collaborative_service = CollaborativeService(
        settings.collaborative_dir
    )

    app.include_router(recommend.router)
    app.include_router(search.router)
    app.include_router(song.router)
    app.include_router(auth.router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
