"""Local CPU embeddings via fastembed.

The embedding model is heavy to load, so it is constructed once and cached per
model name. After the first download the model runs fully offline.
"""

from __future__ import annotations

from functools import lru_cache

from fastembed import TextEmbedding

from core.config import Settings


class Embedder:
    """Thin wrapper around ``fastembed.TextEmbedding``."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into dense float vectors."""

        return [vector.tolist() for vector in self._model.embed(texts)]


@lru_cache
def _cached_embedder(model_name: str) -> Embedder:
    return Embedder(model_name)


def get_embedder(settings: Settings) -> Embedder:
    """Return a cached Embedder for the configured embedding model."""

    return _cached_embedder(settings.embedding_model)
