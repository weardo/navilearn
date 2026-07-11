"""End-to-end ingestion pipeline: source in, study artifacts out.

Ties together parsing, chunking, concept-map extraction, vector storage,
flashcards, summaries and graph rendering behind a single ``process`` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from core.artifacts import Flashcard, Summary, generate_flashcards, summarize
from core.concepts import ConceptMap, extract_concept_map
from core.config import get_settings
from core.embeddings import get_embedder
from core.graph import build_dot, graph_json
from core.llm import get_llm
from core.parsers import chunk_text, extract_text
from core.store import TopicStore


@dataclass
class ProcessResult:
    """The complete set of artifacts produced for one source."""

    source: str
    title: str
    n_chunks: int
    concept_map: ConceptMap
    flashcards: list[Flashcard]
    summary: Summary
    graph_dot: str
    graph_json: dict


@lru_cache
def get_store() -> TopicStore:
    """Return a process-wide persistent TopicStore."""

    return TopicStore()


def _topic_matcher(cmap: ConceptMap):
    """Build a chunk -> best-matching-topic function via keyword overlap.

    Each chunk is assigned to the topic whose name (or child concept names)
    appears most often in the chunk. Falls back to "general".
    """

    topics: list[str] = list(cmap.topics)
    # Map lowercase keyword -> topic, including concept names under each topic.
    keyword_topic: list[tuple[str, str]] = []
    for topic in topics:
        keyword_topic.append((topic.lower(), topic))
    for concept in cmap.concepts:
        if concept.topic:
            keyword_topic.append((concept.name.lower(), concept.topic))

    def match(chunk: str) -> str:
        lowered = chunk.lower()
        best_topic = "general"
        best_score = 0
        # Score each topic by summed keyword occurrences.
        scores: dict[str, int] = {}
        for keyword, topic in keyword_topic:
            if keyword and keyword in lowered:
                scores[topic] = scores.get(topic, 0) + lowered.count(keyword)
        for topic, score in scores.items():
            if score > best_score:
                best_score = score
                best_topic = topic
        return best_topic

    return match


def process(
    source: str,
    store: Optional[TopicStore] = None,
    n_flashcards: int = 12,
) -> ProcessResult:
    """Ingest ``source`` and produce concept map, flashcards, summary, graph.

    When a ``store`` is provided, chunks are embedded and added with each chunk
    tagged to its best-matching topic. Settings, LLM and embedder are built
    internally.
    """

    settings = get_settings()
    llm = get_llm(settings)

    text, title = extract_text(source)
    chunks = chunk_text(text)

    concept_map = extract_concept_map(llm, text)

    if store is not None:
        embedder = get_embedder(settings)
        matcher = _topic_matcher(concept_map)
        store.add_chunks(
            doc_id=title or source,
            chunks=chunks,
            embedder=embedder,
            topic_of=matcher,
            source=source,
        )

    flashcards = generate_flashcards(llm, text, n=n_flashcards)
    summary = summarize(llm, text, concept_map.topics)

    graph_dot = build_dot(concept_map)
    gjson = graph_json(concept_map)

    return ProcessResult(
        source=source,
        title=title,
        n_chunks=len(chunks),
        concept_map=concept_map,
        flashcards=flashcards,
        summary=summary,
        graph_dot=graph_dot,
        graph_json=gjson,
    )
