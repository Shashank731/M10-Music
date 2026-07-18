"""Song metadata API routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse

from backend.app.auth import get_current_user, get_user_from_token
from backend.app.config import get_settings

router = APIRouter(prefix="/songs", tags=["songs"])


def _find_local_audio(song_id: str) -> Path | None:
    """Find a dataset MP3 whose filename includes the track id."""
    dataset_dir = get_settings().dataset_dir
    audio_root = dataset_dir / "MP3-Example"
    if not audio_root.exists():
        return None
    matches = list(audio_root.rglob(f"*{song_id}.mp3"))
    return matches[0] if matches else None


@router.get("/{song_id}")
def get_song(
    song_id: str,
    request: Request,
    _current_user: str = Depends(get_current_user),
) -> dict:
    """Return song metadata by ID."""
    try:
        song = request.app.state.content_service.load().get_song(song_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if song is None:
        raise HTTPException(status_code=404, detail="Song not found")
    if song.get("spotify_preview_url"):
        song["audio_source"] = "spotify_preview"
    elif _find_local_audio(song_id):
        song["audio_source"] = "local_mp3"
    return song


@router.get("/{song_id}/audio")
def get_song_audio(
    song_id: str,
    token: str = Query(...),
) -> FileResponse:
    """Stream a local MP3 preview after token validation.

    The browser audio element cannot attach Authorization headers, so the
    frontend sends the short-lived JWT as a query parameter for this media URL.
    """
    get_user_from_token(token)
    audio_file = _find_local_audio(song_id)
    if audio_file is None:
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(audio_file, media_type="audio/mpeg")
