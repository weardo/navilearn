"""Persistent topic-aware vector store built on ChromaDB.

Chunks are embedded with the shared fastembed embedder and stored with
``source`` and ``topic`` metadata so retrieval can be filtered by topic.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import chromadb
from chromadb.config import Settings as ChromaSettings

from core.config import Settings, get_settings
from core.embeddings import Embedder

_COLLECTION = "navilearn"
_SUPABASE_TABLE = "navilearn_chunks"
_SUPABASE_RPC = "navilearn_match_chunks"

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
        owner_id: str = "",
    ) -> int:
        """Embed and store chunks, tagging each with a source and topic.

        ``topic_of`` may be a fixed topic string, a callable mapping a chunk to
        its topic, or ``None`` (topic defaults to "general"). ``owner_id`` is
        stored in each chunk's metadata so per-owner search can filter to a
        single learner's content; the default "" preserves existing callers.
        Returns the number of chunks added.
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
            metadatas.append(
                {"source": source, "topic": topic, "owner_id": owner_id}
            )
            documents.append(chunk)

        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def search(
        self,
        query: str,
        embedder: Embedder,
        top_k: int = 5,
        owner: str = "",
    ) -> list[dict]:
        """Return the ``top_k`` most similar chunks with source and topic.

        When ``owner`` is non-empty the search is filtered to chunks whose
        metadata ``owner_id`` matches, so a learner only searches their own
        content. The default "" leaves the search unfiltered (backward
        compatible).
        """

        if self.count() == 0:
            return []
        query_vec = embedder.embed([query])[0]
        query_kwargs: dict = {
            "query_embeddings": [query_vec],
            "n_results": min(top_k, self.count()),
        }
        if owner:
            query_kwargs["where"] = {"owner_id": owner}
        result = self._collection.query(**query_kwargs)
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


class SupabaseVectorStore:
    """pgvector-backed store exposing the same interface as :class:`TopicStore`.

    Chunks live in the ``chunks`` table (``content`` + ``embedding vector(384)``)
    and similarity search runs through the ``match_chunks`` SQL function. Drop-in
    for :class:`TopicStore` so the pipeline and search UI stay backend-agnostic.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        from supabase import create_client  # local import: optional dependency

        if settings is None:
            settings = get_settings()
        url = settings.supabase_url
        key = settings.supabase_service_role_key or settings.supabase_anon_key
        if not url or not key:
            raise ValueError(
                "SupabaseVectorStore needs supabase_url and a service or anon key"
            )
        self._client = create_client(url, key)

    def _table(self):
        return self._client.table("chunks")

    def add_chunks(
        self,
        doc_id: str,
        chunks: list[str],
        embedder: Embedder,
        topic_of: TopicOf,
        source: str,
        owner_id: str = "",
    ) -> int:
        """Embed and store chunks, tagging each with a source and topic.

        ``owner_id`` is written into each row's ``owner_id`` column so per-owner
        search can filter to a single learner's content; the default "" keeps
        existing callers unchanged.
        """

        if not chunks:
            return 0

        embeddings = embedder.embed(chunks)
        rows: list[dict] = []
        for chunk, embedding in zip(chunks, embeddings):
            if callable(topic_of):
                topic = topic_of(chunk) or "general"
            elif isinstance(topic_of, str):
                topic = topic_of or "general"
            else:
                topic = "general"
            rows.append(
                {
                    "source": source,
                    "topic": topic,
                    "content": chunk,
                    "embedding": embedding,
                    "owner_id": owner_id,
                }
            )
        self._table().insert(rows).execute()
        return len(chunks)

    def search(
        self,
        query: str,
        embedder: Embedder,
        top_k: int = 5,
        owner: str = "",
    ) -> list[dict]:
        """Return the ``top_k`` most similar chunks with source and topic.

        When ``owner`` is non-empty it is forwarded to the ``match_chunks`` RPC,
        which filters results to that owner; "" (the default) means unfiltered
        and backward compatible.
        """

        query_vec = embedder.embed([query])[0]
        res = self._client.rpc(
            "match_chunks",
            {
                "query": query_vec,
                "k": top_k,
                "threshold": 0.0,
                "owner": owner,
            },
        ).execute()
        out: list[dict] = []
        for row in res.data or []:
            out.append(
                {
                    "text": row.get("content", ""),
                    "source": row.get("source", ""),
                    "topic": row.get("topic", ""),
                    "score": _clamp(float(row.get("similarity", 0.0) or 0.0)),
                }
            )
        return out

    def search_by_topic(self, topic: str) -> list[dict]:
        """Return all stored chunks whose topic matches ``topic``."""

        res = (
            self._table()
            .select("content, source, topic")
            .eq("topic", topic)
            .execute()
        )
        out: list[dict] = []
        for row in res.data or []:
            out.append(
                {
                    "text": row.get("content", ""),
                    "source": row.get("source", ""),
                    "topic": row.get("topic", ""),
                    "score": 1.0,
                }
            )
        return out

    def list_topics(self) -> list[str]:
        """Return the sorted distinct set of topics in the store."""

        res = self._table().select("topic").execute()
        topics = {(row or {}).get("topic", "") for row in (res.data or [])}
        topics.discard("")
        return sorted(topics)

    def count(self) -> int:
        """Return the number of stored chunks."""

        res = self._table().select("id", count="exact").execute()
        return int(res.count or 0)

    def reset(self) -> None:
        """Delete all stored chunks (leaves the table in place)."""

        self._table().delete().neq(
            "id", "00000000-0000-0000-0000-000000000000"
        ).execute()
