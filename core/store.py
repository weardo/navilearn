"""Persistent topic-aware vector store built on ChromaDB.

Chunks are embedded with the shared fastembed embedder and stored with
``source`` and ``topic`` metadata so retrieval can be filtered by topic.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import chromadb
from chromadb.config import Settings as ChromaSettings

from core.config import get_settings
from core.embeddings import Embedder

_COLLECTION = "navilearn"

TopicOf = Union[Callable[[str], str], str, None]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class TopicStore:
    """Wrapper over a persistent ChromaDB collection with topic metadata."""

    def __init__(self, chroma_dir: Optional[str] = None) -> None:
        if chroma_dir is None:
            chroma_dir = get_settings().chroma_dir
        self._client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        doc_id: str,
        chunks: list[str],
        embedder: Embedder,
        topic_of: TopicOf,
        source: str,
    ) -> int:
        """Embed and store chunks, tagging each with a source and topic.

        ``topic_of`` may be a fixed topic string, a callable mapping a chunk to
        its topic, or ``None`` (topic defaults to "general"). Returns the number
        of chunks added.
        """

        if not chunks:
            return 0

        embeddings = embedder.embed(chunks)
        ids: list[str] = []
        metadatas: list[dict] = []
        documents: list[str] = []
        for index, chunk in enumerate(chunks):
            if callable(topic_of):
                topic = topic_of(chunk) or "general"
            elif isinstance(topic_of, str):
                topic = topic_of or "general"
            else:
                topic = "general"
            ids.append(f"{doc_id}:{index}")
            metadatas.append({"source": source, "topic": topic})
            documents.append(chunk)

        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def search(
        self, query: str, embedder: Embedder, top_k: int = 5
    ) -> list[dict]:
        """Return the ``top_k`` most similar chunks with source and topic."""

        if self.count() == 0:
            return []
        query_vec = embedder.embed([query])[0]
        result = self._collection.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, self.count()),
        )
        return self._format(result)

    def search_by_topic(self, topic: str) -> list[dict]:
        """Return all stored chunks whose metadata topic matches ``topic``."""

        result = self._collection.get(where={"topic": topic})
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        out: list[dict] = []
        for text, meta in zip(docs, metas):
            meta = meta or {}
            out.append(
                {
                    "text": text,
                    "source": meta.get("source", ""),
                    "topic": meta.get("topic", ""),
                    "score": 1.0,
                }
            )
        return out

    def list_topics(self) -> list[str]:
        """Return the sorted distinct set of topics in the store."""

        result = self._collection.get()
        metas = result.get("metadatas") or []
        topics = {(m or {}).get("topic", "") for m in metas}
        topics.discard("")
        return sorted(topics)

    def count(self) -> int:
        """Return the number of stored chunks."""

        return self._collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection (drops all stored chunks)."""

        self._client.delete_collection(_COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _format(result: dict) -> list[dict]:
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        out: list[dict] = []
        for text, meta, dist in zip(docs, metas, dists):
            meta = meta or {}
            out.append(
                {
                    "text": text,
                    "source": meta.get("source", ""),
                    "topic": meta.get("topic", ""),
                    "score": _clamp(1.0 - float(dist)),
                }
            )
        return out
