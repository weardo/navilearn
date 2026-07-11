# NaviLearn — GOAL

## One line
A holistic, Indic-first learning platform where **one multimodal understanding pipeline** turns any
input (documents, videos, or a live screen+voice presentation) into structured, scored learning
intelligence, served to students, mentors, and teachers.

## The problem
Edtech at scale (NavGurukul's mission: 1 crore underserved learners) needs to (a) turn messy source
material into study-ready artifacts, (b) assess real understanding, not just completion, and (c) do
both cheaply, offline-tolerant, and in Indian languages. Today these are separate tools with separate
models and inconsistent outputs.

## The moat (proprietary core)
**The shared multimodal analysis pipeline.** A single `analyze(source) -> Understanding` layer that
ingests text / PDF / DOCX / transcripts / video (STT) / live screen (OCR) / audio, and emits ONE
consistent structured representation (concepts, topics, hierarchy, relationships, evidence). Every
feature is a thin consumer of it:
- **Study Studio** turns a document's Understanding into flashcards + summaries + a concept graph.
- **AI Interview** turns a live screen+speech Understanding into adaptive questions + a scored report.
- **Analytics** turns accumulated Understanding into progress and recommendations.

Because there is exactly one pipeline, code overlap is minimal and outputs are consistent across every
stakeholder view. Indic support (Sarvam STT/TTS) is first-class, not bolted on.

## Stakeholders and their modules (mapped to the hackathon challenges)
- **Student**: Study Studio (Ch6) · AI Interview (Ch1) · progress dashboard (Ch4)
- **Mentor**: mentee progress + review (Ch4 role) · meeting-notes -> training modules (Ch7)
- **Teacher**: content/class management + analytics · live classroom collaboration (Ch5)
- **Cross-cutting**: ultra-light client-side summaries (Ch2) inside the voice/interview loop

## Flagship: the autonomous AI Interviewer (Ch1)
Runs **on the candidate's own machine**. It captures the **screen** (OCR of UI, code, slides, diagrams)
and the candidate's **speech** (STT), feeds both into the shared pipeline, and **autonomously** conducts
an adaptive interview: context-aware questions, follow-ups driven by what is on screen and what was said,
then scores technical depth, clarity, originality, and implementation understanding into a feedback
report that lands in the dashboard.

## Build discipline (D-day)
Finish one challenge fully before starting the next. Order: Ch6 Study Studio (done) -> shared multimodal
pipeline -> Ch4 dashboard + roles -> Ch6 video -> Ch1 interview -> stretch (Ch7, Ch2, Ch5).

## Non-negotiables
- Side-effects (storage, telemetry, TTS) never fail a user flow: log and continue.
- Secrets live only in `.env` (gitignored) / host env, never in the repo.
- The multimodal pipeline is the single source of truth: features consume it, they do not re-implement parsing or analysis.
- Ships as a self-contained repo: no imports from sibling projects; copy, do not reference.
