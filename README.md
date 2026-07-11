# NaviLearn

**A holistic, Indic-first learning platform where one multimodal understanding pipeline turns any input into structured, scored learning intelligence for every stakeholder.**

NaviLearn is a Python (Streamlit + FastAPI) and Supabase edtech platform built for the NavGurukul-style mission of reaching underserved learners at scale. Instead of shipping a pile of disconnected tools, it stands one shared analysis pipeline in the middle: documents, PDFs, transcripts, videos, audio, YouTube links, and even a live screen plus voice are all normalized into a single `Understanding` (extracted text plus an LLM-structured concept map). Every feature is a thin consumer of that pipeline, so outputs stay consistent across the three people the platform serves: **students** who learn and get assessed, **mentors** who coach and give feedback, and **teachers** who run and analyze a cohort. Indic language support (Sarvam speech-to-text and text-to-speech) is first-class, and the whole stack is designed to run cheaply and offline-tolerantly on a laptop.

---

## Features by stakeholder and challenge

NaviLearn is a single deliverable that covers multiple hackathon challenges. The pages live under `pages/` in the Streamlit multipage app; the home dashboard is `Home.py`.

### Student

| Feature | Page | What it does |
| --- | --- | --- |
| Progress dashboard | `Home.py` | Sign in (one click for demo accounts), then a student sees top metrics (lessons completed, hours, courses in progress, latest interview score), progress per course, a 30-day activity trend, and a "recommended next" rule. Rendered entirely from the repository. |
| Learn (course player) | `pages/8_Learn.py` | A Coursera-style two-pane player: a persistent module/lesson outline on the left, lesson content (markdown, embedded video, attached docs) on the right, and Mark Complete to close the progress loop and log activity. |
| Study Studio | `pages/1_Study_Studio.py` | The Challenge 6 multimodal ingestion surface. Add one source (upload, paste, or a YouTube URL) and get flashcards, a summary, and a concept graph back. Save the result as a study set. |
| AI Interview | `pages/2_AI_Interview.py` | The flagship Challenge 1 autonomous interviewer, driveable in the browser. It reads the candidate's shared screen (OCR of code, slides, diagrams) and spoken answers (speech-to-text), asks adaptive context-aware questions, then scores the session into a report that lands in the dashboard. |
| Notes | `pages/Notes.py` | An Evernote-style personal notebook (Supabase-backed): title, markdown body, tags, and an optional public share link. |

### Mentor

| Feature | Page | What it does |
| --- | --- | --- |
| Mentor Dashboard | `pages/3_Mentor.py` | A two-way workspace (Challenge 4 mentor role): claim unassigned students, scope the roster to only your own mentees, review their progress, and leave written feedback that persists per student. |
| Training Modules | `pages/4_Training_Modules.py` | Challenge 7 as a mentor/teacher tool: paste or upload the meeting notes you keep repeating and turn them into reusable LMS module content. |
| Author | `pages/9_Author.py` | Create course content: courses, modules, and lessons with markdown bodies and uploaded media (stored in Supabase Storage). |

### Teacher

| Feature | Page | What it does |
| --- | --- | --- |
| Teacher Dashboard | `pages/7_Teacher.py` | A school/admin roll-up of the whole learner cohort into one class-wide overview (completion, activity, interview outcomes). |

### Cross-cutting and shared surfaces

| Feature | Page / Module | What it does |
| --- | --- | --- |
| Live Classroom | `pages/5_Classroom.py` | One shared, Supabase-backed room (Challenge 5): a co-edited notes document, live polls with per-user voting, and running chat. State lives in Postgres, so multiple browsers share it via cheap polling (no websockets). |
| Co-solve workspace | `pages/Cosolve.py` | A shared multi-stakeholder problem-solving room: a problem statement, a co-edited code editor, and a runnable Python workspace whose latest output is shared with the whole room. |
| Messaging | `pages/6_Messages.py` | A Telegram-style chat surface alongside the classroom: a main room, one-to-one direct messages, ad-hoc named group rooms, and full-text search across every room you belong to. |
| Shared study set viewer | `pages/Shared.py` | A public, login-free read-only view of a shared study set via an unlisted link. |
| On-device summarizer | `core/summarizer.py` | The Challenge 2 ultra-light summarizer: a quantized T5-small run entirely through onnxruntime. No LLM, no network, no rate limit. Degrades to an empty summary if the model is absent. |

---

## Architecture summary

NaviLearn is **one pipeline, many thin features**. A single `analyze(source) -> Understanding` layer (`core/multimodal.py`) ingests any modality, normalizes it to text (parsers + speech-to-text + screen OCR), and structures it into a concept map with the configured LLM. Study Studio, the AI Interviewer, and the analytics dashboard are all thin consumers: they call the pipeline and shape its output. They never re-implement parsing or analysis.

Three layers underneath are swappable by a single environment variable each:

- **LLM** via a LiteLLM gateway (`core/llm.py`): Groq by default, with a Groq-to-Groq (or offline Ollama) fallback chain, plus OpenAI, Sarvam (Indic), and a deterministic offline mock.
- **Vector store** (`core/store.py`): local Chroma by default, or Supabase pgvector, behind one interface.
- **Data layer** (`core/repo.py`): stdlib SQLite by default, or Supabase Postgres, behind a `Repository` protocol.

Full details, the module map, and the data model are in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

---

## Tech stack

- **Frontend / app**: Streamlit multipage app (`Home.py` + `pages/`), with a small no-build custom component for browser screen sharing (`components/screen_share/`).
- **Backend API**: FastAPI (`api.py`) served by Uvicorn, with auto-generated Swagger docs.
- **LLM**: Groq `llama-3.1-8b-instant` by default via LiteLLM, with a fallback chain; Sarvam-M optional for Indic.
- **Speech**: Sarvam Saarika (Indic speech-to-text) and Sarvam Bulbul (Indic text-to-speech); Groq Whisper for English.
- **Embeddings**: fastembed `BAAI/bge-small-en-v1.5` (local CPU, offline after first download).
- **Vector store**: Chroma (local, default) or Supabase pgvector.
- **Database**: SQLite (default) or Supabase Postgres.
- **Storage**: Supabase Storage (public `course-media` bucket) for lesson media.
- **On-device summarizer**: quantized T5-small on onnxruntime via optimum.

---

## Setup and run

### Prerequisites

- **Python 3.11+** (the bundled `.venv` uses Python 3.12).
- The repository ships a preconfigured virtual environment at `.venv/`. Use it directly. All commands below call `.venv/bin/...`.

### Create the environment (if `.venv` is missing)

The project is a standard pyproject package. Its declared dependencies are in `pyproject.toml` (there is no `requirements.txt`). A few runtime extras used by the platform layer (FastAPI, Uvicorn, LiteLLM, supabase-py, onnxruntime, optimum) are provisioned in the shipped `.venv` on top of the pyproject core.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
```

### Configure environment variables

Secrets and backend selection live in `.env` (gitignored). Provide your own `.env` with these variable **names** (values not shown here):

| Variable | Purpose |
| --- | --- |
| `LLM_PROVIDER`, `LLM_MODEL` | Active LLM provider and model. |
| `GROQ_API_KEY`, `GROQ_MODEL` | Groq LLM credentials and model. |
| `LLM_FALLBACK_MODELS` | Comma-separated LiteLLM fallback models (the Groq-to-Groq or offline backup chain). Empty means retries only. |
| `SARVAM_API_KEY` | Sarvam Indic speech-to-text and text-to-speech (and optional Indic LLM). |
| `EMBEDDING_MODEL`, `CHROMA_DIR`, `TOP_K` | Embeddings model, local Chroma directory, retrieval depth. |
| `VECTOR_BACKEND` | `chroma` (default) or `supabase`. |
| `DB_BACKEND` | `sqlite` (default) or `supabase`. |
| `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_DB_PASSWORD` | Supabase backend (relational, pgvector, storage). |

### Launch the Streamlit app

**Important:** launch through `run.sh`, which exports `ARROW_DEFAULT_MEMORY_POOL=system` before starting Streamlit. This build's pyarrow can segfault when Arrow serialization runs on a thread other than the one that first imported it (Streamlit spins a fresh thread per rerun); the env var forces pyarrow onto the system allocator as a guard.

```bash
./run.sh
# equivalent to:
#   export ARROW_DEFAULT_MEMORY_POOL=system
#   .venv/bin/streamlit run Home.py
```

### Run the FastAPI backend

The REST API exposes the same dashboard data (profiles, courses, lessons, progress, activity time series, interview reports) as HTTP/JSON.

```bash
.venv/bin/uvicorn api:app --reload --port 8000
```

Interactive Swagger UI is at `http://localhost:8000/docs` (and ReDoc at `/redoc`). The full endpoint reference is in **[docs/API_DOCS.md](docs/API_DOCS.md)**.

---

## Seed data and demo accounts

You do not need to create data by hand. On first use, if the backend has no profiles, the app seeds a small but realistic demo dataset (`seed_demo` in `core/repo.py`): three roles, two courses (Python Foundations and Intro to Machine Learning) with lessons, roughly 30 days of activity, and matching progress rows, so every dashboard has real donut and time-series data immediately.

The seeded users are available as **one-click demo logins** on the home page:

| Name | Role | Demo email |
| --- | --- | --- |
| Sam Student | student | `student@navilearn.dev` |
| Maya Mentor | mentor | `mentor@navilearn.dev` |
| Tara Teacher | teacher | `teacher@navilearn.dev` |

Sam is assigned to Maya, so the Mentor Dashboard has a real mentee to coach.

---

## Testing

Run the test suite with the repo venv:

```bash
.venv/bin/python -m pytest -q
```

Tests stay fully offline: the LLM gateway has a deterministic `MockClient`, and Sarvam and other network paths are not exercised.

---

## Honest limitations and security

This is a hackathon deliverable. Treat the following as blockers before any untrusted or production deployment:

- **Demo-grade auth.** Login is a one-click or email-and-role form with an HMAC-signed cookie. There is no password, email verification, or session hardening. The unauthenticated API `login` endpoint deliberately never trusts a client-supplied role and never mutates an existing profile, but this is not a real identity system.
- **Row Level Security is disabled.** The Supabase tables that back messaging, notes, classroom, and mentoring are created with **no RLS** and are accessed with the service-role key from the server. Owner scoping is enforced in application code, not in the database. See **[docs/CH5_PRIVACY.md](docs/CH5_PRIVACY.md)** for the privacy model and what must change.
- **The co-solve code runner executes on the host.** `run_python` in `core/classroom.py` runs submitted code in a separate Python subprocess with a wall-clock timeout and no shell, but on the same machine with filesystem and network access. It **must be sandboxed** (container, seccomp, network egress block) before accepting untrusted code in a shared or public deployment.
- **API auth caveats.** The FastAPI service has no authentication or authorization on its data routes. See **[docs/API_DOCS.md](docs/API_DOCS.md)** for the specifics.
