"""Vector search abstraction with a FAISS implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover - optional in lightweight tests
    faiss = None  # type: ignore[assignment]


class VectorStore(Protocol):
    """Minimal vector search API used by recommenders."""

    def add(self, embeddings: np.ndarray, ids: list[str]) -> None:
        """Add embeddings and external IDs."""

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Return top-k IDs and scores for a query vector."""

    def save(self, path: str | Path) -> None:
        """Persist the index."""


@dataclass
class FaissVectorStore:
    """Inner-product FAISS store, replaceable behind the VectorStore protocol."""

    normalize_vectors: bool = True

    def __post_init__(self) -> None:
        self.index = None
        self.ids: list[str] = []
        self._fallback_embeddings: np.ndarray | None = None

    def _prepare(self, embeddings: np.ndarray) -> np.ndarray:
        vectors = np.asarray(embeddings, dtype="float32")
        if self.normalize_vectors:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)
        return vectors

    def add(self, embeddings: np.ndarray, ids: list[str]) -> None:
        """Add vectors to FAISS or a NumPy fallback index."""
        vectors = self._prepare(embeddings)
        self.ids = list(ids)
        if faiss is None:
            self._fallback_embeddings = vectors
            return

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Search by inner product/cosine similarity."""
        if not self.ids:
            return []

        query_vector = self._prepare(np.asarray(query, dtype="float32").reshape(1, -1))
        limit = min(top_k, len(self.ids))
        if faiss is not None and self.index is not None:
            scores, indices = self.index.search(query_vector, limit)
            return [
                (self.ids[int(idx)], float(score))
                for idx, score in zip(indices[0], scores[0])
                if int(idx) >= 0
            ]

        if self._fallback_embeddings is None:
            return []
        scores = self._fallback_embeddings @ query_vector.ravel()
        indices = np.argsort(scores)[::-1][:limit]
        return [(self.ids[int(idx)], float(scores[int(idx)])) for idx in indices]

    def save(self, path: str | Path) -> None:
        """Persist a FAISS index file when FAISS is installed."""
        if faiss is None or self.index is None:
            return
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(output_path))

    @classmethod
    def load(cls, index_path: str | Path, ids: list[str]) -> FaissVectorStore:
        """Load a FAISS index from disk."""
        store = cls()
        store.ids = list(ids)
        if faiss is None:
            raise ImportError("faiss-cpu is required to load a FAISS index")
        store.index = faiss.read_index(str(index_path))
        return store
