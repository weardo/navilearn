# NaviLearn Architecture

This document is the component-level companion to the top-level `README.md`. It describes the one-pipeline design, what each module does, the data model, and why the platform is built as one shared pipeline with many thin features. All paths are relative to the repo root.

---

## One pipeline, many features

The heart of NaviLearn is a single analysis pipeline. Every feature consumes its output instead of re-implementing parsing or structuring.

```
sources (txt / md / pdf / docx / srt / vtt / youtube / video / audio / screen frames / raw string)
        |
   core/multimodal.py  analyze(source) -> Understanding      <- the moat, single source of truth
        |   parsers + speech-to-text + screen OCR feed it; the LLM structures it
        |
        +-- Study Studio    flashcards / summaries / concept graph   (core/pipeline.py, artifacts, graph)
        +-- AI Interview     adaptive questions + scored report       (core/interview.py)
        +-- Analytics        progress + recommendations               (Home.py, core/repo.py)
```

`analyze(source)` returns an `Understanding`: the extracted plain text, a `modality` label (document, transcript, video, audio, text, youtube), an LLM-built `ConceptMap`, and a small `meta` dict. `analyze_text(...)` is the in-memory variant used by the interviewer for live speech and screen text, and `ocr_frames(...)` is the extension hook that turns captured screen frames into text (via a Groq vision model) before they flow back through the same path.

**The rule:** features never re-implement parsing or analysis. They call the pipeline and shape its output. This keeps code overlap minimal and outputs consistent across every stakeholder view.

---

## Core module map

`core/` is the shared services (reuse) layer. Streamlit pages and the FastAPI app are thin consumers of it.

### Pipeline and AI

| Module | Responsibility |
| --- | --- |
| `multimodal.py` | The unified `analyze()` entry: classify modality, normalize to text, structure into a concept map. The moat. |
| `pipeline.py` | End-to-end Study Studio ingestion: parse, chunk, extract concept map, embed into the vector store, generate flashcards and a summary, render the graph. Returns a `ProcessResult`. |
| `parsers.py` | Document and transcript text extraction and chunking (pdf, docx, txt, md, srt, vtt, youtube). |
| `stt.py` | Video/audio to text. Groq Whisper (English) and Sarvam Saarika (Indic). Owns the video/audio extension sets. |
| `vision.py` | `describe_screen(...)`: read a screen frame with a Groq vision model (OCR of code, slides, diagrams). |
| `concepts.py` | LLM extraction of a `ConceptMap` (topics, concepts, relationships). |
| `artifacts.py` | LLM generation of flashcards and summaries. |
| `graph.py` | Render a concept map to Graphviz DOT and JSON. |
| `interview.py` | The autonomous interviewer engine (Challenge 1): opening question, adaptive follow-ups grounded in screen and speech text, and a rubric-scored report. Consumes text the pipeline produced; never parses itself. |
| `summarizer.py` | Offline on-device summarizer (Challenge 2): a quantized T5-small on onnxruntime via optimum. No LLM, no network. Degrades to `""`. |
| `sarvam.py` | Indic voice: `SarvamTTS` (Bulbul) and `SarvamSTT` (Saarika) over Sarvam's HTTP API. |

### Provider abstractions (swappable layers)

| Module | Responsibility |
| --- | --- |
| `llm.py` | The LiteLLM gateway: one `get_llm(settings).chat(messages) -> LLMResponse` interface for Groq, OpenAI, Sarvam, Ollama, or a deterministic `MockClient`. Automatic fallbacks (Groq-to-Groq or offline backup); zero primary retries so a rate limit fails over immediately instead of blocking the Streamlit thread. |
| `embeddings.py` | Local CPU embeddings via fastembed (`bge-small-en-v1.5`), cached per model. |
| `store.py` | Topic-aware vector store: `TopicStore` (Chroma, default) or `SupabaseVectorStore` (pgvector). Same interface; both tag chunks with `source`, `topic`, and `owner_id`. |
| `repo.py` | The `Repository` protocol plus `SqliteRepo` (default, stdlib) and `SupabaseRepo` (Postgres). Typed dataclass entities. `get_repo(settings)` picks the backend; `seed_demo(repo)` populates the demo dataset. |
| `config.py` | `Settings` (pydantic-settings) read from `.env`. Holds provider, model, backend, and Supabase keys. |

### Stakeholder feature data layers (Supabase-backed, thin, best-effort)

| Module | Responsibility |
| --- | --- |
| `messaging.py` | Direct-message, group, and searchable messaging. Main room, deterministic DM room ids, full-text search over rooms the user belongs to. |
| `notes.py` | Owner-scoped personal notes with an optional public share flag. |
| `classroom.py` | The Live Classroom (shared notes, polls, chat) plus the shared co-solve surface and `run_python(...)`, the subprocess code runner. |
| `mentoring.py` | Mentor write verbs: claim a student, scope to own mentees, persist per-student feedback (`mentor_notes`). Bridges readable ids to Postgres uuids via `uuid5`. |
| `training_modules.py` | Turn recurring meeting notes into reusable LMS module content (Challenge 7). |
| `storage.py` | Upload lesson media into the public `course-media` Supabase Storage bucket, returning a public URL. |
| `session.py` | Streamlit session helpers: auth, the cached repository, demo accounts, first-use seeding. |
| `screen_share.py` | Browser `getDisplayMedia` capture as a no-build Streamlit component: the candidate's own screen as base64 JPEG frames. |
| `capture.py` | Server-display capture (correct only for a locally self-hosted run; `screen_share.py` is the honest path for a hosted deployment). |
| `exporters.py`, `jsonutil.py` | CSV exporters for progress/activity; robust extraction of JSON from messy LLM output. |

The FastAPI app (`api.py`) is a thin consumer of `core/repo.py`: it shapes repository output into typed Pydantic responses and holds no business logic. The CLI (`cli.py`) runs the Study Studio pipeline in batch.

---

## Data model

Two backends model the same entities so application code is shared: the portable SQLite schema is in `db/schema.sql`, and the Supabase/Postgres counterpart is in `supabase/migrations/`. All ids are text so opaque SQLite strings and Postgres uuid text interoperate.

### Core relational tables (`db/schema.sql` and the platform migration)

| Table | Key columns | Purpose |
| --- | --- | --- |
| `profiles` | `id`, `email`, `full_name`, `role` (student / mentor / teacher), `mentor_id` | A platform user; `mentor_id` links a student to a mentor. |
| `courses` | `id`, `title`, `description` | A course grouping ordered lessons. |
| `lessons` | `id`, `course_id`, `title`, `order_index`, `content` (markdown), `module`, `video_url`, `doc_url` | Real teaching material: a markdown body, a grouping module name, optional embedded video and attached doc. |
| `progress` | `id`, `student_id`, `lesson_id`, `status`, `time_spent_seconds`, `completed_at`, unique `(student_id, lesson_id)` | A student's progress against one lesson. |
| `activity_events` | `id`, `student_id`, `type`, `payload` (JSON), `created_at` | A timestamped learner action feeding the analytics time series (a `seconds` payload key feeds study-minutes). |
| `study_sets` | `id`, `owner_id`, `title`, `source`, `content` (JSON), `created_at` | A saved Study Studio result (a learner's "notes"), shareable read-only. |
| `interview_reports` | `id`, `student_id`, `project_title`, `scores` (JSON), `feedback`, `created_at` | A scored AI Interview report that surfaces on the dashboard. |
| `chunks` (Supabase) | `content`, `embedding vector(384)`, `source`, `topic`, `owner_id` | pgvector store for embeddings; searched via the `match_chunks` SQL function. Local runs use Chroma instead. |

### Feature tables (Supabase, text ids, no RLS)

| Table(s) | Purpose |
| --- | --- |
| `rooms`, `room_members`, `room_messages` | Messaging: rooms typed `main` / `dm` / `group`, membership `(room_id, user_id)`, and messages with a generated `fts` tsvector for full-text search. |
| `notes` | Personal notes (`owner_id`, `title`, `body`, `tags`, `source`, `is_public`, timestamps). |
| `classroom_sessions`, `classroom_notes`, `classroom_polls`, `poll_votes`, `chat_messages`, `classroom_solve` | The Live Classroom (shared notes, polls with per-user votes, chat) and the co-solve surface (shared code, language, problem statement, latest output). |
| `mentor_notes` | Per-student written mentor feedback (`student_id`, `mentor_id` as text). |

---

## Swappable provider story

Every external dependency sits behind an interface chosen by one setting, so swapping a vendor or going offline is a config change, not a code change.

| Concern | Setting | Options | Default |
| --- | --- | --- | --- |
| LLM | `LLM_PROVIDER` (+ `LLM_FALLBACK_MODELS`) | groq, openai, sarvam, ollama, mock | groq `llama-3.1-8b-instant` |
| Vector store | `VECTOR_BACKEND` | chroma (local), supabase (pgvector) | chroma |
| Data layer | `DB_BACKEND` | sqlite (stdlib), supabase (Postgres) | sqlite |
| Speech-to-text | (provider) | Groq Whisper (English), Sarvam Saarika (Indic) | Groq Whisper |
| Text-to-speech | (Sarvam) | Sarvam Bulbul (Indic voice) | Sarvam Bulbul |
| Summarizer | on-device | quantized T5-small on onnxruntime | onnxruntime |

Design invariants that make this hold together:

- **The LLM gateway is the single AI entry point.** Every feature calls `get_llm(settings).chat(...)`; no feature binds to a vendor SDK.
- **Side-effects never fail a user flow.** Storage, telemetry, speech, and the Supabase feature layers are best-effort: they log and continue (reads degrade to empty, writes to no-ops) so a backend hiccup never crashes a page.
- **The pipeline is the single source of truth.** Features consume `Understanding`; they do not re-parse or re-structure.
- **Self-contained repo.** No imports from sibling projects; the platform ships on its own.

For the REST surface see [API_DOCS.md](API_DOCS.md); for the privacy and RLS model see [CH5_PRIVACY.md](CH5_PRIVACY.md).
