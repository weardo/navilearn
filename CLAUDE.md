# NaviLearn — agent working notes

Read `GOAL.md` first. This file is the how-we-build conventions.

## What this is
A Python edtech platform. Self-contained repo (no imports from sibling projects). Ships separately.

## Architecture: one pipeline, many thin features
```
sources (txt/pdf/docx/srt/vtt/youtube/video/audio/screen-frame)
        │
   core/multimodal.py  analyze(source) -> Understanding   ← the moat, single source of truth
        │  (parsers + STT + OCR feed it; LLM structures it)
        ├── Study Studio  (core/artifacts.py, core/graph.py)  flashcards / summaries / concept graph
        ├── AI Interview  (core/interview.py)                 adaptive Q + scoring from live screen+speech
        └── Analytics     (dashboard)                         progress + recommendations
```
Rule: **features never re-implement parsing or analysis.** They call the pipeline and shape its output.

## Layout
- `core/` — shared services (the reuse layer):
  - `config.py` Settings (.env) · `llm.py` Groq/Sarvam/Ollama/mock adapter · `embeddings.py` fastembed
  - `parsers.py` document/transcript extract · `stt.py` video/audio -> text (Groq Whisper / Sarvam Saarika)
  - `concepts.py` `artifacts.py` `graph.py` `store.py` (Chroma) `jsonutil.py`
  - `multimodal.py` the unified `analyze()` entry (build this next) · `interview.py` (Ch1) · `sarvam.py` (Indic voice)
- `app.py` Study Studio web tool · `cli.py` batch CLI · `tests/` · `data/samples/`

## Providers
- LLM: Groq `llama-3.1-8b-instant` (default, key in `.env`). Sarvam-M optional for Indic.
- STT: Groq `whisper-large-v3-turbo` (English) · Sarvam `saarika:v2` (Indic). TTS: Sarvam `bulbul` (Indic voice).
- Embeddings: fastembed `bge-small-en-v1.5`. Vector store: Chroma (persistent, local).

## Commands (use the repo venv)
```bash
.venv/bin/python -m pytest -q                 # tests
.venv/bin/streamlit run app.py                # Study Studio web tool
.venv/bin/python cli.py <source> --out OUTDIR # batch artifacts
```

## Conventions
- Typed, `from __future__ import annotations`, small modules, docstrings. NO em dashes in any text.
- Robust LLM-JSON parsing via `core/jsonutil.extract_json` (never trust raw model output).
- Side-effects best-effort (log + continue). Secrets only in `.env` (gitignored).
- Finish one challenge before the next; keep every increment demoable.
