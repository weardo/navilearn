"""Tests for the shared multimodal pipeline (core/multimodal.py).

``test_analyze_document`` makes one real Groq call to structure a sample
lesson; the rest are network-free.
"""

from __future__ import annotations

from pathlib import Path

from core.concepts import ConceptMap
from core.multimodal import Understanding, analyze, analyze_text, ocr_frames

_SAMPLE = str(
    Path(__file__).resolve().parent.parent
    / "data"
    / "samples"
    / "lesson_photosynthesis.md"
)


def test_analyze_document() -> None:
    """A markdown lesson is classified as a document and structured (1 call)."""

    understanding = analyze(_SAMPLE)
    assert isinstance(understanding, Understanding)
    assert understanding.modality == "document"
    assert understanding.text.strip()
    assert isinstance(understanding.concept_map, ConceptMap)
    assert understanding.concept_map.concepts, "expected a non-empty concept map"


def test_analyze_text_in_memory() -> None:
    """A raw string is analyzed as modality 'text' without any parsing."""

    understanding = analyze_text(
        "Newton's second law relates force, mass and acceleration.",
        title="note",
        structure=False,
    )
    assert understanding.modality == "text"
    assert understanding.meta["title"] == "note"
    assert understanding.text.startswith("Newton")


def test_ocr_frames_stub_returns_empty() -> None:
    """The OCR extension hook exists and returns '' until a backend is wired."""

    assert ocr_frames(["frame1.png", "frame2.png"], settings=None) == ""
