"""Simple email/password auth with JWT (no external deps).

This module implements registration, login and token helpers using HMAC-SHA256 JWT
encoding implemented with the standard library so no new packages are required.
User records are stored in a JSON file next to this module (users.json).

Note: This is intentionally small and suitable for demo/dev environments. For
production prefer well-tested libraries (Auth backend, PyJWT, passlib, etc.).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from backend.app.config import get_settings


router = APIRouter(prefix="/auth", tags=["auth"])

USERS_PATH = Path(__file__).resolve().parent / "users.json"
DEFAULT_EXP_SECONDS = 60 * 60  # 1 hour


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    padding = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + padding)


def _sign(message: bytes, secret: str) -> str:
    sig = hmac.new(secret.encode(), message, hashlib.sha256).digest()
    return _b64url_encode(sig)


def encode_jwt(payload: dict[str, Any], secret: str, exp: int | None = None) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    data = payload.copy()
    if exp is None:
        exp = int(time.time()) + DEFAULT_EXP_SECONDS
    data["exp"] = exp
    header_b = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_b = _b64url_encode(json.dumps(data, separators=(",", ":")).encode())
    signing_input = f"{header_b}.{payload_b}".encode()
    signature = _sign(signing_input, secret)
    return f"{header_b}.{payload_b}.{signature}"


def decode_jwt(token: str, secret: str) -> dict[str, Any]:
    try:
        header_b, payload_b, signature = token.split(".")
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from err
    signing_input = f"{header_b}.{payload_b}".encode()
    expected_sig = _sign(signing_input, secret)
    if not hmac.compare_digest(expected_sig, signature):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token signature")
    payload_json = _b64url_decode(payload_b)
    data = json.loads(payload_json)
    if data.get("exp", 0) < int(time.time()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    return data


def load_users() -> dict[str, dict[str, str]]:
    if not USERS_PATH.exists():
        return {}
    try:
        return json.loads(USERS_PATH.read_text())
    except Exception:
        return {}


def save_users(users: dict[str, dict[str, str]]) -> None:
    USERS_PATH.write_text(json.dumps(users, indent=2))


def _hash_password(password: str, salt: bytes) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return dk.hex()


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/register", status_code=201)
def register(payload: RegisterRequest):
    users = load_users()
    email = payload.email.lower()
    if email in users:
        raise HTTPException(status_code=400, detail="User already exists")
    salt = os.urandom(16)
    users[email] = {"salt": salt.hex(), "pw": _hash_password(payload.password, salt)}
    save_users(users)
    return {"detail": "user created"}


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest):
    users = load_users()
    email = payload.email.lower()
    user = users.get(email)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    salt = bytes.fromhex(user["salt"])
    if _hash_password(payload.password, salt) != user["pw"]:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    settings = get_settings()
    token = encode_jwt({"sub": email}, settings.jwt_secret)
    return TokenResponse(access_token=token)


def get_current_user(request: Request) -> str:
    """Extract and validate bearer token from Authorization header and return subject (email)."""
    auth = request.headers.get("authorization")
    if not auth or not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization")
    token = auth.split(None, 1)[1]
    return get_user_from_token(token)


def get_user_from_token(token: str) -> str:
    """Validate a raw bearer token and return its subject."""
    settings = get_settings()
    data = decode_jwt(token, settings.jwt_secret)
    sub = data.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Malformed token")
    return sub
