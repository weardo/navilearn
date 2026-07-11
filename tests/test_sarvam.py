"""Offline tests for core.sarvam.

No network is touched: the Sarvam key/endpoint may rate-limit, so these cover
only pure helpers (the sentence splitter, the wav concatenator) and the graceful
no-key paths (STT returns "", TTS raises). Live Bulbul/Saarika calls are
verified separately by the main thread.
"""
from __future__ import annotations

import io
import wave

import pytest

from core.config import Settings
from core.sarvam import (
    MAX_TTS_CHARS,
    SarvamSTT,
    SarvamTTS,
    _concat_wav,
    split_sentences,
    synthesize,
    transcribe,
)


def test_module_functions_are_callable() -> None:
    assert callable(split_sentences)
    assert callable(synthesize)
    assert callable(transcribe)
    assert callable(SarvamTTS)
    assert callable(SarvamSTT)


def test_split_short_text_single_chunk() -> None:
    chunks = split_sentences("Hello world.")
    assert chunks == ["Hello world."]


def test_split_respects_char_limit() -> None:
    sentence = "A" * 100 + ". "
    text = sentence * 60  # ~6000 chars
    chunks = split_sentences(text)
    assert len(chunks) > 1
    assert all(len(c) <= MAX_TTS_CHARS for c in chunks)


def test_split_devanagari_boundary() -> None:
    chunks = split_sentences("नमस्ते दुनिया। यह एक परीक्षा है।", limit=20)
    assert len(chunks) == 2


def test_split_empty_returns_empty_string_chunk() -> None:
    assert split_sentences("   ") == [""]


def test_split_single_oversized_sentence_not_dropped() -> None:
    text = "B" * (MAX_TTS_CHARS + 500)  # no boundary, one giant sentence
    chunks = split_sentences(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def _make_wav(nframes: int = 8, value: bytes = b"\x00\x01") -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(value * nframes)
    return buf.getvalue()


def test_concat_single_returns_input() -> None:
    blob = _make_wav()
    assert _concat_wav([blob]) is blob


def test_concat_multiple_wavs_sums_frames() -> None:
    a = _make_wav(nframes=8)
    b = _make_wav(nframes=8)
    merged = _concat_wav([a, b])
    with wave.open(io.BytesIO(merged), "rb") as w:
        assert w.getnframes() == 16
        assert w.getnchannels() == 1


def test_tts_raises_without_key() -> None:
    tts = SarvamTTS(Settings(sarvam_api_key=""))
    with pytest.raises(RuntimeError):
        tts.synthesize("नमस्ते")


def test_tts_raises_on_empty_text() -> None:
    tts = SarvamTTS(Settings(sarvam_api_key="dummy"))
    with pytest.raises(RuntimeError):
        tts.synthesize("   ")


def test_stt_returns_empty_without_key() -> None:
    stt = SarvamSTT(Settings(sarvam_api_key=""))
    assert stt.transcribe("/nonexistent/path.wav") == ""


def test_stt_returns_empty_on_missing_file() -> None:
    # Key present but the file does not exist: caught, returns "".
    stt = SarvamSTT(Settings(sarvam_api_key="dummy"))
    assert stt.transcribe("/nonexistent/path.wav", language_code="hi-IN") == ""
