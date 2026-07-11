# AGENTS

Start with `GOAL.md` (what and why) then `CLAUDE.md` (how we build). Those are the source of truth.

Quick orientation:
- One shared pipeline (`core/multimodal.py` -> `analyze(source)`) is the moat; every feature consumes it.
- Self-contained Python repo, own `.venv`, secrets in `.env` (gitignored).
- Run tests with `.venv/bin/python -m pytest -q`; the Study Studio UI with `.venv/bin/streamlit run app.py`.
- Finish one challenge fully before starting the next; keep each step demoable.
