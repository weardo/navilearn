"""Fast, mostly-offline tests for the NaviLearn core engine."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.artifacts import Flashcard, flashcards_to_csv, flashcards_to_json, learning_path
from core.concepts import Concept, ConceptMap, Edge
from core.graph import build_dot, graph_json
from core.jsonutil import extract_json
from core.parsers import chunk_text, extract_text

_SAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "samples",
    "lesson_photosynthesis.md",
)


def _tiny_map() -> ConceptMap:
    return ConceptMap(
        concepts=[
            Concept(name="Chlorophyll", definition="Green pigment.", topic="Pigments"),
            Concept(name="Glucose", definition="A sugar.", topic="Outputs"),
        ],
        topics=["Pigments", "Outputs"],
        hierarchy={"Pigments": ["Chlorophyll"], "Outputs": ["Glucose"]},
        edges=[Edge(source="Chlorophyll", target="Glucose", relation="prerequisite-of")],
    )


def test_chunk_text_sizing():
    text = "word " * 1000  # 5000 chars.
    chunks = chunk_text(text, size=900, overlap=120)
    assert len(chunks) > 1
    assert all(len(c) <= 900 for c in chunks)
    assert all(c.strip() for c in chunks)


def test_chunk_text_short_returns_single():
    assert chunk_text("short text", size=900) == ["short text"]


def test_chunk_text_empty():
    assert chunk_text("   ", size=900) == []


def test_extract_text_markdown():
    text, title = extract_text(_SAMPLE)
    assert "Photosynthesis" in text
    assert "chlorophyll" in text.lower()
    assert title == "lesson_photosynthesis"


def test_extract_text_unsupported_raises():
    with pytest.raises(ValueError):
        extract_text("something.xyz")


def test_build_dot_is_digraph():
    dot = build_dot(_tiny_map())
    assert dot.startswith("digraph")
    assert "Chlorophyll" in dot
    assert "prerequisite-of" in dot


def test_build_dot_escapes_quotes():
    cmap = ConceptMap(
        concepts=[Concept(name='He said "hi"', definition="", topic="T")],
        topics=["T"],
    )
    dot = build_dot(cmap)
    assert '\\"hi\\"' in dot


def test_graph_json_shape():
    data = graph_json(_tiny_map())
    assert {"nodes", "edges"} <= set(data)
    assert data["nodes"][0]["id"] == "Chlorophyll"
    assert data["edges"][0]["relation"] == "prerequisite-of"


def test_flashcards_to_csv_header_and_rows():
    cards = [
        Flashcard(front="Q1", back="A1", topic="T1"),
        Flashcard(front="Q2", back="A2", topic="T2"),
    ]
    csv_text = flashcards_to_csv(cards)
    lines = [line for line in csv_text.splitlines() if line]
    assert lines[0] == "front,back,topic"
    assert len(lines) == 3
    assert "Q1" in lines[1]


def test_flashcards_to_json_roundtrip():
    import json

    cards = [Flashcard(front="Q", back="A", topic="T")]
    parsed = json.loads(flashcards_to_json(cards))
    assert parsed[0]["front"] == "Q"


def test_learning_path_respects_prereq():
    path = learning_path(_tiny_map())
    assert path.index("Chlorophyll") < path.index("Glucose")


def test_extract_json_from_fenced():
    raw = 'Sure!\n```json\n{"a": 1, "b": [2, 3]}\n```\nDone.'
    assert extract_json(raw) == {"a": 1, "b": [2, 3]}


def test_extract_json_balanced_object_in_prose():
    raw = 'Here is the map: {"name": "x", "nested": {"k": "}"}} thanks'
    assert extract_json(raw) == {"name": "x", "nested": {"k": "}"}}


def test_extract_json_fallback_default():
    assert extract_json("no json here", default={}) == {}
