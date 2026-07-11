"""Study artifacts generated from learning content: flashcards, summaries,
and a suggested learning path.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass, field

from core.concepts import ConceptMap
from core.jsonutil import extract_json
from core.llm import LLMClient

_MAX_CHARS = 8000


@dataclass
class Flashcard:
    """A single question/answer study card tagged with its topic."""

    front: str
    back: str
    topic: str


@dataclass
class Summary:
    """An overall summary plus one short summary per topic."""

    overall: str
    per_topic: dict[str, str] = field(default_factory=dict)


def _content_and_topics(text_or_conceptmap: object) -> tuple[str, list[str]]:
    """Normalize the flashcard input into (text, topics)."""

    if isinstance(text_or_conceptmap, ConceptMap):
        cmap = text_or_conceptmap
        lines: list[str] = []
        for concept in cmap.concepts:
            lines.append(f"{concept.name} ({concept.topic}): {concept.definition}")
        return "\n".join(lines), list(cmap.topics)
    return str(text_or_conceptmap), []


_FLASHCARD_SYSTEM = (
    "You are a study-aid generator. You create clear question/answer "
    "flashcards and reply with JSON only, no prose."
)

_FLASHCARD_PROMPT = """Create exactly {n} study flashcards from the content below.

Return ONLY a JSON array. Each element is an object:
{{"front": "a question", "back": "the concise answer", "topic": "the topic it belongs to"}}

The front is a question, the back is the answer. Keep answers concise.

Content:
\"\"\"
{content}
\"\"\"
"""


def generate_flashcards(
    llm: LLMClient, text_or_conceptmap: object, n: int = 12
) -> list[Flashcard]:
    """Generate up to ``n`` flashcards from text or a concept map."""

    content, _topics = _content_and_topics(text_or_conceptmap)
    content = content.strip()
    if len(content) > _MAX_CHARS:
        content = content[:_MAX_CHARS]

    messages = [
        {"role": "system", "content": _FLASHCARD_SYSTEM},
        {"role": "user", "content": _FLASHCARD_PROMPT.format(n=n, content=content)},
    ]
    response = llm.chat(messages)
    data = extract_json(response.text, default=[])

    # Accept either a bare array or an object wrapping the array.
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                data = value
                break

    cards: list[Flashcard] = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            front = str(item.get("front", "")).strip()
            back = str(item.get("back", "")).strip()
            if not front or not back:
                continue
            cards.append(
                Flashcard(
                    front=front,
                    back=back,
                    topic=str(item.get("topic", "")).strip() or "general",
                )
            )
    return cards[:n]


def flashcards_to_csv(cards: list[Flashcard]) -> str:
    """Serialize flashcards to CSV text with a header row."""

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["front", "back", "topic"])
    for card in cards:
        writer.writerow([card.front, card.back, card.topic])
    return buffer.getvalue()


def flashcards_to_json(cards: list[Flashcard]) -> str:
    """Serialize flashcards to a pretty JSON array string."""

    return json.dumps([asdict(card) for card in cards], indent=2, ensure_ascii=False)


_SUMMARY_SYSTEM = (
    "You are a concise study summarizer. You reply with JSON only, no prose."
)

_SUMMARY_PROMPT = """Summarize the learning content below.

Return ONLY a JSON object with these keys:
- "overall": one tight paragraph summarizing the whole content.
- "per_topic": an object mapping each of these topics to a 1-2 sentence summary: {topics}

If a topic is not covered, give a brief best-effort note.

Content:
\"\"\"
{content}
\"\"\"
"""


def summarize(llm: LLMClient, text: str, topics: list[str]) -> Summary:
    """Produce an overall summary and a short per-topic summary."""

    content = text.strip()
    if len(content) > _MAX_CHARS:
        content = content[:_MAX_CHARS]
    topic_list = ", ".join(topics) if topics else "(infer the main topics)"

    messages = [
        {"role": "system", "content": _SUMMARY_SYSTEM},
        {
            "role": "user",
            "content": _SUMMARY_PROMPT.format(content=content, topics=topic_list),
        },
    ]
    response = llm.chat(messages)
    data = extract_json(response.text, default={})

    overall = ""
    per_topic: dict[str, str] = {}
    if isinstance(data, dict):
        overall = str(data.get("overall", "")).strip()
        raw = data.get("per_topic", {})
        if isinstance(raw, dict):
            for key, value in raw.items():
                per_topic[str(key).strip()] = str(value).strip()
    if not overall:
        # Fall back to the raw model text so a summary is always returned.
        overall = response.text.strip()[:500]
    return Summary(overall=overall, per_topic=per_topic)


def learning_path(cmap: ConceptMap) -> list[str]:
    """Order concepts/topics respecting prerequisite edges.

    Builds a dependency graph from "prerequisite-of" edges and returns a
    topological order. Falls back to hierarchy order (then topic order) when
    there are no usable edges or a cycle is detected.
    """

    node_names: list[str] = [c.name for c in cmap.concepts]
    for topic in cmap.topics:
        if topic not in node_names:
            node_names.append(topic)

    prereq_edges = [e for e in cmap.edges if e.relation == "prerequisite-of"]

    def _hierarchy_order() -> list[str]:
        ordered: list[str] = []
        for topic in cmap.topics:
            if topic not in ordered:
                ordered.append(topic)
            for child in cmap.hierarchy.get(topic, []):
                if child not in ordered:
                    ordered.append(child)
        for name in node_names:
            if name not in ordered:
                ordered.append(name)
        return ordered

    if not prereq_edges:
        return _hierarchy_order()

    # Kahn's algorithm. Edge source is a prerequisite of target: source -> target.
    nodes = set(node_names)
    for edge in prereq_edges:
        nodes.add(edge.source)
        nodes.add(edge.target)

    indegree: dict[str, int] = {n: 0 for n in nodes}
    adjacency: dict[str, list[str]] = {n: [] for n in nodes}
    for edge in prereq_edges:
        adjacency[edge.source].append(edge.target)
        indegree[edge.target] += 1

    # Deterministic ordering: process ready nodes in insertion order.
    insertion = {name: i for i, name in enumerate(node_names)}

    def _sort_key(name: str) -> tuple[int, str]:
        return (insertion.get(name, len(insertion)), name)

    ready = sorted([n for n in nodes if indegree[n] == 0], key=_sort_key)
    ordered: list[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for neighbor in adjacency[current]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                ready.append(neighbor)
        ready.sort(key=_sort_key)

    if len(ordered) != len(nodes):
        # Cycle detected: fall back to hierarchy order.
        return _hierarchy_order()
    return ordered
