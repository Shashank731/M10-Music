"""Database extension point.

The first production pass serves from trained artifacts. This module keeps a
dedicated database boundary for future metadata persistence without coupling the
backend to storage details.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatabaseConfig:
    """Database connection settings."""

    url: str
