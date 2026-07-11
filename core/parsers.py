"""Multi-source text extraction and chunking for NaviLearn.

Supports PDF, DOCX, plain text, Markdown, subtitle files (SRT/VTT) and
YouTube URLs. Each extractor returns the raw text plus a best-effort title.
"""

from __future__ import annotations

import os
import re
from urllib.parse import parse_qs, urlparse

_YOUTUBE_HOSTS = ("youtube.com", "youtu.be", "www.youtube.com", "m.youtube.com")


def _is_youtube(source: str) -> bool:
    """Return True when the source looks like a YouTube watch URL."""

    lowered = source.lower()
    return "youtube.com" in lowered or "youtu.be" in lowered


def _youtube_video_id(url: str) -> str:
    """Extract the 11-character video id from common YouTube URL shapes."""

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if "youtu.be" in host:
        vid = parsed.path.lstrip("/").split("/")[0]
        return vid or url
    query = parse_qs(parsed.query)
    if "v" in query and query["v"]:
        return query["v"][0]
    # Fallbacks for /embed/<id>, /shorts/<id>, /live/<id>.
    parts = [p for p in parsed.path.split("/") if p]
    if parts:
        return parts[-1]
    return url


def _extract_pdf(source: str) -> tuple[str, str]:
    from pypdf import PdfReader

    reader = PdfReader(source)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages)
    title = os.path.splitext(os.path.basename(source))[0]
    return text, title


def _extract_docx(source: str) -> tuple[str, str]:
    import docx

    document = docx.Document(source)
    paragraphs = [p.text for p in document.paragraphs]
    text = "\n".join(paragraphs)
    title = os.path.splitext(os.path.basename(source))[0]
    return text, title


def _extract_plain(source: str) -> tuple[str, str]:
    with open(source, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    title = os.path.splitext(os.path.basename(source))[0]
    return text, title


def _strip_subtitles(raw: str, is_vtt: bool) -> str:
    """Drop indices, timestamps and cue settings, keep spoken lines."""

    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if is_vtt and (stripped == "WEBVTT" or stripped.startswith(("NOTE", "STYLE"))):
            continue
        if stripped.isdigit():  # SRT sequence index.
            continue
        if "-->" in stripped:  # Timestamp line.
            continue
        # Remove inline tags like <00:00:01.000> or <c> styling.
        cleaned = re.sub(r"<[^>]+>", "", stripped)
        cleaned = cleaned.strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def _extract_subtitles(source: str) -> tuple[str, str]:
    is_vtt = source.lower().endswith(".vtt")
    with open(source, "r", encoding="utf-8", errors="replace") as handle:
        raw = handle.read()
    text = _strip_subtitles(raw, is_vtt)
    title = os.path.splitext(os.path.basename(source))[0]
    return text, title


def _extract_youtube(source: str) -> tuple[str, str]:
    from youtube_transcript_api import YouTubeTranscriptApi

    video_id = _youtube_video_id(source)
    segments: list[dict]
    try:
        # Newer API exposes an instance ``fetch`` method.
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id)
        segments = [
            {"text": snippet.text}
            for snippet in fetched
        ]
    except (AttributeError, TypeError):
        # Older API exposes a classmethod ``get_transcript``.
        segments = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join(
        (seg["text"] if isinstance(seg, dict) else getattr(seg, "text", "")).strip()
        for seg in segments
    )
    return text, video_id


def extract_text(source: str) -> tuple[str, str]:
    """Extract ``(text, title)`` from a file path or a YouTube URL.

    Dispatch is by extension for local files and by host for URLs. Raises
    ``ValueError`` for anything unsupported.
    """

    if _is_youtube(source):
        return _extract_youtube(source)

    lowered = source.lower()
    if lowered.endswith(".pdf"):
        return _extract_pdf(source)
    if lowered.endswith(".docx"):
        return _extract_docx(source)
    if lowered.endswith((".txt", ".md", ".markdown")):
        return _extract_plain(source)
    if lowered.endswith((".srt", ".vtt")):
        return _extract_subtitles(source)

    raise ValueError(
        f"Unsupported source: {source!r}. Supported: .pdf, .docx, .txt, .md, "
        ".srt, .vtt files or a YouTube URL."
    )


def chunk_text(text: str, size: int = 900, overlap: int = 120) -> list[str]:
    """Split text into overlapping character windows on word boundaries.

    Returns a list of non-empty chunks of roughly ``size`` characters each,
    with ``overlap`` characters shared between consecutive chunks so context is
    not severed at boundaries.
    """

    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    normalized = text.strip()
    if not normalized:
        return []
    if len(normalized) <= size:
        return [normalized]

    chunks: list[str] = []
    length = len(normalized)
    start = 0
    while start < length:
        end = min(start + size, length)
        window = normalized[start:end]
        # Prefer to break on the last whitespace so words stay intact.
        if end < length:
            cut = window.rfind(" ")
            if cut > size // 2:
                end = start + cut
                window = normalized[start:end]
        chunk = window.strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        # Advance, keeping ``overlap`` characters of context. Always make at
        # least one character of progress to avoid infinite loops.
        start = max(end - overlap, start + 1)
    return chunks
