"""Speech-to-text for arbitrary video/audio from any source.

Real STT, not a transcript shortcut: extract audio with ffmpeg, transcribe via
Groq's Whisper API (whisper-large-v3-turbo). Handles any uploaded or downloaded
video (mp4, mkv, webm, mov, avi, ...) and audio (mp3, wav, m4a, ...). Long media
is split into time chunks so it stays under the API file-size limit.

Whisper STT is a documented extension point for a fully local pipeline
(faster-whisper on CPU), but Groq keeps it fast and light on this machine.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import httpx

from core.config import Settings, get_settings

GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
STT_MODEL = "whisper-large-v3-turbo"
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".flv", ".wmv"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus"}
# Chunk length in seconds. 16kHz mono wav is ~32 KB/s, so 600s ~= 19 MB, safely
# under Groq's free-tier upload limit.
CHUNK_SECONDS = 600


def is_media(path: str) -> bool:
    """True if the path looks like a video or audio file we can transcribe."""
    return Path(path).suffix.lower() in (VIDEO_EXTS | AUDIO_EXTS)


def extract_audio(media_path: str, out_wav: str) -> str:
    """Extract mono 16kHz wav from any video/audio via ffmpeg. Returns out_wav."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", media_path, "-vn", "-ac", "1", "-ar", "16000",
         "-f", "wav", out_wav],
        check=True, capture_output=True,
    )
    return out_wav


def _duration_seconds(path: str) -> float:
    """Media duration via ffprobe, 0.0 if it cannot be determined."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def _transcribe_file(wav_path: str, settings: Settings) -> str:
    """Send one audio file to Groq Whisper and return the transcript text."""
    with open(wav_path, "rb") as fh:
        resp = httpx.post(
            GROQ_TRANSCRIBE_URL,
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            files={"file": (Path(wav_path).name, fh, "audio/wav")},
            data={"model": STT_MODEL},
            timeout=180.0,
        )
    resp.raise_for_status()
    return resp.json().get("text", "").strip()


def transcribe(media_path: str, settings: Settings | None = None) -> str:
    """Transcribe a video or audio file end to end.

    Extracts audio, and for long media splits it into CHUNK_SECONDS segments,
    transcribes each, and joins the results in order.
    """
    settings = settings or get_settings()
    with tempfile.TemporaryDirectory() as tmp:
        wav = extract_audio(media_path, str(Path(tmp) / "audio.wav"))
        duration = _duration_seconds(wav)

        if duration <= CHUNK_SECONDS:
            return _transcribe_file(wav, settings)

        # Split into fixed-length chunks with ffmpeg segmenter.
        seg_dir = Path(tmp) / "segs"
        seg_dir.mkdir()
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav, "-f", "segment",
             "-segment_time", str(CHUNK_SECONDS), "-c", "copy",
             str(seg_dir / "chunk_%03d.wav")],
            check=True, capture_output=True,
        )
        parts = sorted(seg_dir.glob("chunk_*.wav"))
        return " ".join(_transcribe_file(str(p), settings) for p in parts).strip()
