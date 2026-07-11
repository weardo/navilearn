"""Screen vision for the AI Interviewer: read an image with a vision LLM.

Challenge 1 needs the interviewer to see the candidate's screen (their editor,
running app, diagrams, slides) and turn it into text the shared pipeline can
reason over. This module sends a captured frame to a Groq vision model through
LiteLLM and returns a plain-text description.

The rest of the pipeline stays modality-agnostic: :func:`describe_screen` is the
only place that knows about image bytes. :func:`core.multimodal.ocr_frames`
calls it per frame and joins the results, so every consumer (interview, study
studio, analytics) receives ordinary text.

Failure is non-fatal by contract: a missing key, an unreachable endpoint, or a
malformed image yields ``""`` rather than raising, so a live capture loop never
crashes on a bad frame.
"""

from __future__ import annotations

import base64
from pathlib import Path

from litellm import completion

from core.config import Settings, get_settings

# Groq multimodal model reachable via LiteLLM (verified to accept image_url).
VISION_MODEL = "groq/meta-llama/llama-4-scout-17b-16e-instruct"

# Default instruction: pull out everything an interviewer would want to see.
_DEFAULT_PROMPT = (
    "You are assisting a technical interviewer by reading the candidate's "
    "shared screen. Describe concisely what is visible: the UI or application, "
    "any code snippets (name languages, functions, and notable logic), "
    "diagrams, architecture, and on-screen text. Infer what the project appears "
    "to do. Report only what is actually shown, do not invent details."
)

# Map common image suffixes to the mime type LiteLLM expects in the data URI.
_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def _mime_for(path: str) -> str:
    """Return the image mime type for ``path``, defaulting to PNG."""

    return _MIME_BY_SUFFIX.get(Path(path).suffix.lower(), "image/png")


def describe_screen(
    image_path: str,
    settings: Settings | None = None,
    prompt: str | None = None,
) -> str:
    """Describe one screen image with the Groq vision model, as plain text.

    Reads the image at ``image_path``, base64-encodes it into a data URI, and
    asks the vision model to extract what is on screen for a technical
    interview. ``prompt`` overrides the default instruction when given.

    Returns the description text, or ``""`` on any error (missing file, missing
    API key, network failure, malformed response) so a live capture loop is
    never broken by a single bad frame.
    """

    settings = settings or get_settings()
    instruction = prompt or _DEFAULT_PROMPT
    try:
        raw = Path(image_path).read_bytes()
        if not raw:
            return ""
        b64 = base64.b64encode(raw).decode("ascii")
        data_uri = f"data:{_mime_for(image_path)};base64,{b64}"
        response = completion(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            api_key=settings.groq_api_key,
        )
        content = response.choices[0].message.content
        return (content or "").strip()
    except Exception:  # noqa: BLE001 - vision is best-effort; never crash a frame.
        return ""
