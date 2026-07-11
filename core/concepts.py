"""Concept-map extraction from learning content via an LLM.

The LLM is asked for structured JSON describing concepts, topics, a topic
hierarchy and typed edges between concepts. Output is parsed robustly and
coerced into typed dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.jsonutil import extract_json
from core.llm import LLMClient

_MAX_CHARS = 8000


@dataclass
class Concept:
    """A single named concept with a short definition and owning topic."""

    name: str
    definition: str
    topic: str


@dataclass
class Edge:
    """A typed relationship between two concepts or topics."""

    source: str
    target: str
    relation: str


@dataclass
class ConceptMap:
    """Structured knowledge map extracted from learning content."""

    concepts: list[Concept] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    hierarchy: dict[str, list[str]] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)


_SYSTEM = (
    "You are an expert curriculum designer who turns raw learning material "
    "into a structured concept map. You reply with JSON only, no prose."
)

_PROMPT = """Analyze the learning content below and produce a concept map.

Return ONLY a single JSON object with exactly these keys:
- "concepts": array of objects, each {{"name": str, "definition": str, "topic": str}}. At most {max_concepts} concepts. Keep definitions to one sentence.
- "topics": array of the distinct topic strings used above (high-level themes).
- "hierarchy": object mapping each topic to an array of its subtopics or concept names.
- "edges": array of objects {{"source": str, "target": str, "relation": str}} where relation is one of "prerequisite-of", "part-of", or "related-to". Sources and targets should be concept or topic names.

Content:
\"\"\"
{content}
\"\"\"
"""


def _coerce_concept_map(data: object, max_concepts: int) -> ConceptMap:
    if not isinstance(data, dict):
        return ConceptMap()

    concepts: list[Concept] = []
    for item in data.get("concepts", []) or []:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            concepts.append(
                Concept(
                    name=name,
                    definition=str(item.get("definition", "")).strip(),
                    topic=str(item.get("topic", "")).strip() or "general",
                )
            )
        elif isinstance(item, str) and item.strip():
            concepts.append(Concept(name=item.strip(), definition="", topic="general"))
    concepts = concepts[:max_concepts]

    topics: list[str] = []
    for topic in data.get("topics", []) or []:
        topic_str = str(topic).strip()
        if topic_str and topic_str not in topics:
            topics.append(topic_str)
    # Backfill topics from concepts if the model omitted the list.
    if not topics:
        for concept in concepts:
            if concept.topic and concept.topic not in topics:
                topics.append(concept.topic)

    hierarchy: dict[str, list[str]] = {}
    raw_hierarchy = data.get("hierarchy", {})
    if isinstance(raw_hierarchy, dict):
        for key, value in raw_hierarchy.items():
            key_str = str(key).strip()
            if not key_str:
                continue
            if isinstance(value, list):
                children = [str(v).strip() for v in value if str(v).strip()]
            elif isinstance(value, str):
                children = [value.strip()] if value.strip() else []
            else:
                children = []
            hierarchy[key_str] = children

    edges: list[Edge] = []
    for item in data.get("edges", []) or []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        if not source or not target:
            continue
        edges.append(
            Edge(
                source=source,
                target=target,
                relation=str(item.get("relation", "related-to")).strip()
                or "related-to",
            )
        )

    return ConceptMap(
        concepts=concepts, topics=topics, hierarchy=hierarchy, edges=edges
    )


def extract_concept_map(
    llm: LLMClient, text: str, max_concepts: int = 15
) -> ConceptMap:
    """Extract a :class:`ConceptMap` from ``text`` using the LLM.

    Very long text is truncated before sending. Malformed model output degrades
    gracefully to a partial or empty map rather than raising.
    """

    content = text.strip()
    if len(content) > _MAX_CHARS:
        content = content[:_MAX_CHARS]

    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": _PROMPT.format(content=content, max_concepts=max_concepts),
        },
    ]
    response = llm.chat(messages)
    data = extract_json(response.text, default={})
    return _coerce_concept_map(data, max_concepts)
