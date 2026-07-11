# NaviLearn: Hackathon Submission

**One unified, Indic-first learning platform for students, mentors, and teachers,
built on one shared multimodal understanding pipeline.**

- **Live demo:** https://navi.soupup.ai
- **Repository:** https://github.com/weardo/navilearn
- **Stack:** Python (Streamlit + FastAPI), Supabase (Postgres + pgvector + Auth + Storage), Groq LLM, Sarvam Indic speech, fastembed, ONNX (client-side).

## Try it in 60 seconds
1. Open https://navi.soupup.ai
2. On the sign-in page pick a one-click demo account:
   - **Sam Student** (student dashboard, Learn, Study Studio, AI Interview, Notes, Messages)
   - **Maya Mentor** (mentee roster, two-way feedback, assign students)
   - **Tara Teacher** (cohort dashboard, course authoring)
3. Login persists across reloads; log out returns you to sign-in.

## Challenges covered (one integrated product, not separate apps)

| Challenge | What NaviLearn ships |
|---|---|
| **Student Dashboard (full-stack)** | Progress per course, time spent, completed lessons, a 30-day activity trend, completion donut, adaptive "recommended next", CSV export, and a documented FastAPI backend (auth + aggregates + time-series + lesson details + activity, Swagger at `/docs`). |
| **Learn / course management** | Coursera-style course outline (course -> module -> lesson), lessons carry text, **video**, and **documents**; **Mark complete** writes real progress that drives the dashboard. Mentors/teachers author courses and upload media. |
| **AI Interviewer** | A conversational interviewer that reads the candidate's **shared screen** (Groq vision, `llama-4-scout`) and **microphone** (browser mic), runs a screen+mic pre-flight, keeps a continuous conversation log, scores against a rubric (LLM-as-judge), and **saves the report** so the candidate, their mentor, and the teacher all see it. |
| **Study Studio (multimodal ingestion)** | Any document, PDF, transcript, video, audio, or YouTube link -> concept graph, flashcards, layered summaries, and an owner-scoped searchable topic index (RAG over pgvector). |
| **On-device summarizer** | A quantized T5-small ONNX summarizer that runs client-side (via onnxruntime), keeping the server LLM-only (Groq). |
| **Live Classroom + Co-solve** | A shared room with notes, live polls, chat, and a separate **Co-solve** page: a runnable collaborative code workspace with a problem statement, optimistic-concurrency saves, and shared run output. |
| **Notes** | Evernote-style personal notes with **Save to Notes** from anywhere (summaries, flashcards, concept graphs, interview reports) and **shareable read-only links**. |
| **Messaging** | 1:1 and group chats with full-text search, WhatsApp-style bubbles, and live updates. |
| **Mentor + Teacher** | Two-way mentor<->student feedback threads, mentee assignment, and a cohort dashboard (coverage, per-student progress, activity). |

## Architecture (the moat: one pipeline, many thin features)
```
sources (text / pdf / docx / srt / vtt / youtube / video / audio / screen frame / voice)
        |
   core/multimodal.py  ->  Understanding  (extracted text + LLM-structured concept map)
        |-- Study Studio      (flashcards / summaries / concept graph, owner-scoped RAG)
        |-- AI Interview       (screen + mic -> adaptive Q + rubric scoring)
        |-- Learn / Dashboard  (progress, recommendations, analytics)
        |-- Notes / Messaging / Classroom / Mentor / Teacher
```
- **Swappable AI layer:** a LiteLLM gateway with a Groq -> Groq multi-model fallback chain (each Groq model has its own token budget), plus Sarvam for Indic speech. Vendor or offline models swap by config, no code change.
- **One data plane on Supabase:** relational tables + pgvector embeddings + Auth + Storage.
- See `docs/ARCHITECTURE.md` for the module map and data model, `docs/API_DOCS.md` for the REST API.

## Deployment
Running on a Hetzner host: a systemd-managed Streamlit service behind a Cloudflare
tunnel at `navi.soupup.ai`, talking to hosted Supabase. Full runbook (Docker
option, tunnel, secrets) in `docs/DEPLOY.md`.

## Honest limitations (demo-grade by design)
- The **co-solve code runner** is disabled on the hosted demo (`NAVI_ENABLE_CODE_RUN=false`) because it executes on the host.
- **Row Level Security is off** (the app uses the Supabase service key server-side); documented in `docs/CH5_PRIVACY.md`.
- The **REST API** now has real password + bearer auth with per-student authorization; the interactive Streamlit demo login is intentionally frictionless (one-click accounts) for judging.
- The **on-device ONNX summarizer** ships client-side and is not installed on the server (the server uses Groq).
