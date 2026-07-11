"""Hands-free voice + screen capture as a Streamlit custom component.

This is the perception layer for the conversational interview. Where
:mod:`core.screen_share` only grabs the candidate's screen on a timer,
this component listens continuously to the candidate's microphone in the
browser and, using client-side voice-activity detection, hands Python one
utterance at a time WHEN THE CANDIDATE GOES QUIET. Each returned value pairs
the recorded speech (a ``data:audio/webm`` data URL) with a fresh screen frame
(a ``data:image/jpeg`` data URL) grabbed at the same instant, so the
interviewer always reasons over the answer and the screen that produced it.

The candidate clicks "Enable mic and screen" once; from then on they just talk.
The component increments ``seq`` per utterance so callers can tell a new answer
from a persisting one across reruns, and returns ``seq == -1`` on error. Python
passes ``allow=False`` to pause the mic while the interviewer's own voice plays,
so the interviewer is never captured as an answer.

Returns ``None`` until the first utterance arrives. The decode helpers translate
the returned data URLs into raw bytes, degrading to ``None`` on any bad, empty,
or error value so a single bad turn never breaks the interview.
"""

from __future__ import annotations

import base64
import binascii
import os
from typing import Optional

import streamlit.components.v1 as components

from core.screen_share import data_url_to_jpeg

_COMPONENT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "components", "voice_screen"
)

# Declared once at import time. ``path`` points at the static frontend directory.
_voice_screen_component = components.declare_component(
    "navilearn_voice_screen", path=_COMPONENT_DIR
)


def voice_screen_widget(
    allow: bool = True,
    silence_ms: int = 1100,
    speech_rms: float = 0.02,
    key: Optional[str] = None,
) -> Optional[dict]:
    """Render the hands-free voice + screen widget and return the latest utterance.

    ``allow`` is passed to the frontend: set it ``False`` to pause the mic while
    the interviewer is speaking (so the interviewer's voice is not captured as an
    answer) and ``True`` to resume listening for the next answer. ``silence_ms``
    is how long the candidate must stay quiet after speech before a turn is
    ended, and ``speech_rms`` is the microphone energy above which audio counts
    as speech.

    Returns the component's dict for the most recent utterance
    (``{"seq": int, "audio": data-url, "screen": data-url, "error"?: str}``),
    or ``None`` before the first utterance has arrived.
    """

    result = _voice_screen_component(
        allow=allow,
        silence_ms=silence_ms,
        speech_rms=speech_rms,
        default=None,
        key=key,
    )
    return result if isinstance(result, dict) else None


def decode_audio(data_url: Optional[str]) -> Optional[bytes]:
    """Decode a ``data:audio/webm;base64,...`` string into raw audio bytes.

    Returns ``None`` for a missing value, an ``"ERROR:"`` sentinel, a data URL
    with no payload, or malformed base64, so a bad recording never breaks the
    interview flow.
    """

    if not data_url or not isinstance(data_url, str) or data_url.startswith("ERROR:"):
        return None
    if "," not in data_url:
        return None
    b64 = data_url.split(",", 1)[1].strip()
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64)
    except (binascii.Error, ValueError):
        return None
    return raw or None


def decode_screen(data_url: Optional[str]) -> Optional[bytes]:
    """Decode a ``data:image/jpeg;base64,...`` string into raw JPEG bytes.

    Reuses :func:`core.screen_share.data_url_to_jpeg`, returning ``None`` for a
    missing value, an error sentinel, or malformed data.
    """

    return data_url_to_jpeg(data_url)
