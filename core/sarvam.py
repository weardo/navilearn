"""Indic voice for NaviLearn via Sarvam AI (a NavGurukul-style differentiator).

Two thin clients over Sarvam's HTTP API, shaped like the rest of the kit
(settings-driven, httpx, best-effort errors):

- SarvamTTS  (Bulbul): text -> spoken wav bytes in an Indian language. Long text
  is chunked at sentence boundaries so every request stays under the API cap.
- SarvamSTT  (Saarika): an audio file -> transcript text, for Indic speech that
  Groq Whisper handles less well.

Both reuse settings.sarvam_api_key. TTS raises a clear error on failure so a
caller can fall back to another voice, STT returns "" so it never crashes a
learner's flow. The key/endpoint may rate-limit, so tests stay fully offline.
"""
from __future__ import annotations

import base64
import io
import re
import wave

import httpx

from core.config import Settings, get_settings

TTS_URL = "https://api.sarvam.ai/text-to-speech"
STT_URL = "https://api.sarvam.ai/speech-to-text"

# Bulbul caps text per request; chunk defensively below the documented limit.
MAX_TTS_CHARS = 2400
DEFAULT_TTS_MODEL = "bulbul:v2"
DEFAULT_TTS_SPEAKER = "anushka"
DEFAULT_TTS_LANG = "hi-IN"
DEFAULT_STT_MODEL = "saarika:v2.5"

_TIMEOUT = 120.0


def split_sentences(text: str, limit: int = MAX_TTS_CHARS) -> list[str]:
    """Split text into <=limit-char chunks at sentence boundaries, greedily.

    Boundaries are the Devanagari danda and the Latin sentence marks (. ? !).
    A single sentence longer than limit is emitted on its own (Sarvam then
    trims it) so the splitter never loops or drops text.
    """
    sents = re.split(r"(?<=[।.?!])\s+", text.strip())
    chunks: list[str] = []
    cur = ""
    for sentence in sents:
        if not sentence:
            continue
        if cur and len(cur) + len(sentence) + 1 > limit:
            chunks.append(cur)
            cur = sentence
        elif cur:
            cur = f"{cur} {sentence}"
        else:
            cur = sentence
    if cur:
        chunks.append(cur)
    return chunks or [text.strip()]


def _concat_wav(parts: list[bytes]) -> bytes:
    """Concatenate several wav byte blobs into one wav using stdlib wave.

    Assumes a shared format (Bulbul returns a consistent codec per call). Falls
    back to the first blob if a part cannot be parsed.
    """
    if len(parts) == 1:
        return parts[0]
    frames: list[bytes] = []
    params = None
    for blob in parts:
        try:
            with wave.open(io.BytesIO(blob), "rb") as w:
                params = params or w.getparams()
                frames.append(w.readframes(w.getnframes()))
        except (wave.Error, EOFError):
            continue
    if params is None:
        return parts[0]
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setparams(params)
        for fr in frames:
            writer.writeframes(fr)
    return out.getvalue()


class SarvamTTS:
    """Bulbul text-to-speech: Indian-language text -> wav bytes."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def _synth_chunk(
        self, text: str, speaker: str, lang: str, model: str
    ) -> bytes:
        """POST one chunk and return its decoded wav bytes."""
        resp = httpx.post(
            TTS_URL,
            headers={
                "api-subscription-key": self._settings.sarvam_api_key,
                "Content-Type": "application/json",
            },
            json={
                "text": text,
                "target_language_code": lang,
                "model": model,
                "speaker": speaker,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        audios = resp.json().get("audios") or []
        if not audios:
            raise RuntimeError("Sarvam TTS returned no audio")
        return base64.b64decode(audios[0])

    def synthesize(
        self,
        text: str,
        speaker: str = DEFAULT_TTS_SPEAKER,
        lang: str = DEFAULT_TTS_LANG,
        model: str = DEFAULT_TTS_MODEL,
    ) -> bytes:
        """Synthesize text to wav bytes, chunking long text at sentences.

        Raises a clear RuntimeError on any failure (missing key, network,
        empty response) so a caller can catch it and pick another voice,
        rather than crashing.
        """
        if not text or not text.strip():
            raise RuntimeError("Sarvam TTS: empty text")
        if not self._settings.sarvam_api_key:
            raise RuntimeError("Sarvam TTS: SARVAM_API_KEY is not set")
        try:
            parts = [
                self._synth_chunk(chunk, speaker, lang, model)
                for chunk in split_sentences(text)
            ]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Sarvam TTS failed: {exc}") from exc
        return _concat_wav(parts)


class SarvamSTT:
    """Saarika speech-to-text: an audio file -> transcript text."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def transcribe(
        self, audio_path: str, language_code: str = "unknown"
    ) -> str:
        """Transcribe an audio file via Saarika, returning the transcript.

        Returns "" on any failure (missing key, network, bad response) so it
        never crashes a caller. Pass language_code like "hi-IN" to hint the
        language, or leave "unknown" for auto-detection.
        """
        if not self._settings.sarvam_api_key:
            return ""
        data: dict[str, str] = {"model": DEFAULT_STT_MODEL}
        if language_code and language_code != "unknown":
            data["language_code"] = language_code
        try:
            with open(audio_path, "rb") as fh:
                resp = httpx.post(
                    STT_URL,
                    headers={
                        "api-subscription-key": self._settings.sarvam_api_key,
                    },
                    files={"file": (audio_path, fh, "audio/wav")},
                    data=data,
                    timeout=_TIMEOUT,
                )
            resp.raise_for_status()
            return str(resp.json().get("transcript", "") or "").strip()
        except Exception:  # noqa: BLE001
            return ""


def synthesize(
    text: str,
    speaker: str = DEFAULT_TTS_SPEAKER,
    lang: str = DEFAULT_TTS_LANG,
    model: str = DEFAULT_TTS_MODEL,
    settings: Settings | None = None,
) -> bytes:
    """Module-level convenience wrapper over SarvamTTS.synthesize."""
    return SarvamTTS(settings).synthesize(text, speaker=speaker, lang=lang, model=model)


def transcribe(
    audio_path: str,
    language_code: str = "unknown",
    settings: Settings | None = None,
) -> str:
    """Module-level convenience wrapper over SarvamSTT.transcribe."""
    return SarvamSTT(settings).transcribe(audio_path, language_code=language_code)
