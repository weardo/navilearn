"""The shared multimodal analysis pipeline: NaviLearn's moat.

Every feature (Study Studio, AI Interview, Analytics) consumes a single
:class:`Understanding` produced here instead of re-implementing parsing or
structuring. A source of any modality (document, transcript, video, audio,
YouTube, screen frames, or a raw in-memory string) is normalized to text and
then structured into a :class:`ConceptMap` by the LLM.

Flow: source -> ``to_text`` (parsers + STT classify + extract) -> optional
``extract_concept_map`` (LLM) -> ``Understanding``. Frames feed in through the
``ocr_frames`` extension hook so the live interviewer can add screen OCR later
without touching any consumer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.concepts import ConceptMap, extract_concept_map
from core.config import Settings, get_settings
from core.llm import get_llm
from core.parsers import extract_text
from core.stt import AUDIO_EXTS, VIDEO_EXTS

_TRANSCRIPT_EXTS = (".srt", ".vtt")
_DOCUMENT_EXTS = (".pdf", ".docx", ".txt", ".md", ".markdown")


@dataclass
class Understanding:
    """Normalized, structured view of one source, shared by every feature.

    ``text`` is the extracted plain text, ``modality`` records where it came
    from (document, transcript, video, audio, text, youtube) and
    ``concept_map`` is the LLM-structured knowledge map (possibly empty when
    structuring is skipped).
    """

    source: str
    modality: str
    text: str
    concept_map: ConceptMap
    meta: dict = field(default_factory=dict)


def _is_youtube(source: str) -> bool:
    """True when the source looks like a YouTube URL."""

    lowered = source.lower()
    return "youtube.com" in lowered or "youtu.be" in lowered


def _classify_modality(source: str) -> str:
    """Classify a source into a modality label by URL shape or extension.

    Anything that is not a recognized URL or file extension is treated as a
    raw in-memory string (modality ``"text"``).
    """

    if _is_youtube(source):
        return "youtube"
    lowered = source.lower()
    if lowered.endswith(_TRANSCRIPT_EXTS):
        return "transcript"
    if lowered.endswith(_DOCUMENT_EXTS):
        return "document"
    if lowered.endswith(tuple(VIDEO_EXTS)):
        return "video"
    if lowered.endswith(tuple(AUDIO_EXTS)):
        return "audio"
    return "text"


def to_text(source: str, settings: Settings) -> tuple[str, str, str]:
    """Return ``(text, title, modality)`` for a source of any modality.

    Documents, transcripts and YouTube URLs are dispatched through
    :func:`core.parsers.extract_text`; video and audio are transcribed inside
    ``extract_text`` via :mod:`core.stt`. A source with no recognized URL shape
    or file extension is treated as raw text and returned as-is.
    """

    modality = _classify_modality(source)
    if modality == "text":
        return source, "pasted", "text"
    text, title = extract_text(source)
    return text, title, modality


def analyze(
    source: str, settings: Settings | None = None, structure: bool = True
) -> Understanding:
    """Analyze a source into an :class:`Understanding`.

    Extracts text and modality via :func:`to_text`, then (when ``structure`` is
    True and there is text) builds a :class:`ConceptMap` with the configured
    LLM. Set ``structure=False`` to skip the LLM call and get an empty map.
    """

    settings = settings or get_settings()
    text, title, modality = to_text(source, settings)
    if structure and text.strip():
        concept_map = extract_concept_map(get_llm(settings), text)
    else:
        concept_map = ConceptMap()
    return Understanding(
        source=source,
        modality=modality,
        text=text,
        concept_map=concept_map,
        meta={"title": title},
    )


def analyze_text(
    text: str,
    title: str = "pasted",
    settings: Settings | None = None,
    structure: bool = True,
) -> Understanding:
    """Analyze an in-memory string (modality ``"text"``).

    Used by the interviewer for live speech and screen text where there is no
    file to parse. Behaves like :func:`analyze` but skips extraction.
    """

    settings = settings or get_settings()
    if structure and text.strip():
        concept_map = extract_concept_map(get_llm(settings), text)
    else:
        concept_map = ConceptMap()
    return Understanding(
        source=title,
        modality="text",
        text=text,
        concept_map=concept_map,
        meta={"title": title},
    )


def ocr_frames(image_paths: list[str], settings: Settings) -> str:
    """Frames -> text extension hook for screen OCR (documented stub).

    This is the interface the AI Interviewer fills in to read a candidate's
    shared screen: a list of captured frame image paths is turned into text by
    a vision model, which then flows into :func:`analyze_text` like any other
    source. No OCR backend is wired here yet, so this returns an empty string
    rather than raising, keeping the pipeline callable end to end.

    To implement: send each frame to a vision-capable model (for example a
    Groq or Sarvam multimodal endpoint) and concatenate the recognized text in
    frame order. Deduplicate near-identical consecutive frames to save tokens.
    """

    # NotImplemented-style no-op: the contract exists, the backend does not.
    _ = (image_paths, settings)
    return ""
