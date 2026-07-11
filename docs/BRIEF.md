# NaviLearn — Hackathon Brief

**One unified, Indic-first learning platform for students, mentors, and teachers, built on a single multimodal understanding pipeline.**

- **Live demo:** https://navi.soupup.ai
- **Code (public):** https://github.com/weardo/navilearn
- **Try instantly:** one-click demo accounts on the sign-in page: **Sam Student**, **Maya Mentor**, **Tara Teacher**.

## The idea
Instead of shipping separate tools per challenge, NaviLearn puts **one shared pipeline** in the middle: any input (document, PDF, transcript, video, audio, YouTube link, a live screen, or voice) is turned into a structured `Understanding` (text + an LLM-built concept map). Every feature is a thin consumer of that pipeline, so outputs stay consistent for the three people the platform serves.

## What it does
- **Student dashboard:** progress per course, time spent, completed lessons, a 30-day trend, completion donut, adaptive "recommended next", CSV export, and a documented REST API (Swagger at `/docs`).
- **Learn (Coursera-style):** course -> module -> lesson with text, **video**, and **documents**; "Mark complete" writes real progress.
- **AI Interviewer:** reads the candidate's **shared screen** (Groq vision) and **microphone** (browser mic), runs a screen+mic pre-flight, keeps a live conversation log, scores against a rubric, and **saves the report** for the candidate, mentor, and teacher.
- **Study Studio:** any source -> concept graph, flashcards, layered summaries, owner-scoped semantic search (RAG on pgvector).
- **On-device summarizer:** a quantized T5-small ONNX model runs client-side (server stays Groq-only).
- **Live Classroom + Co-solve:** shared notes, live polls, chat, and a runnable collaborative code workspace.
- **Notes:** Evernote-style, "Save to Notes" from anywhere, plus shareable read-only links.
- **Messaging:** 1:1 and group chat with full-text search.
- **Mentor + Teacher:** two-way mentor<->student feedback threads, mentee assignment, and a cohort dashboard.

## Architecture
```
sources (text / pdf / video / audio / youtube / screen / voice)
     -> core/multimodal.py  ->  Understanding (text + concept map)
         -> Study Studio, AI Interview, Learn/Dashboard, Notes, Classroom, Messaging, Mentor/Teacher
```
**Swappable AI:** LiteLLM gateway with a Groq -> Groq multi-model fallback chain, plus Sarvam for Indic speech. Vendor or offline models swap by config.
**One data plane:** Supabase (Postgres + pgvector + Auth + Storage).

## Tech stack
Python (Streamlit + FastAPI), Supabase, Groq LLM + vision, Sarvam STT/TTS, fastembed, ONNX (client-side). Deployed on Hetzner (systemd + Cloudflare tunnel).

## Demo in 3 steps
1. Open https://navi.soupup.ai and click a demo account.
2. As **Sam Student**: open Learn (complete a lesson), Study Studio (paste text -> flashcards), AI Interview (share screen + speak).
3. As **Maya Mentor / Tara Teacher**: see the same student's progress, interview reports, and send two-way feedback.

## Honest notes
Demo-grade by design: the co-solve code runner is disabled on the public URL for safety; the Streamlit login is intentionally one-click for judging (the REST API has real password + bearer auth). Full docs: `docs/ARCHITECTURE.md`, `docs/API_DOCS.md`, `docs/DEPLOY.md`.
