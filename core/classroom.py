"""Supabase-backed data layer for the Live Classroom Collaboration page (Ch5).

A single shared classroom session holds three collaborative surfaces that any
signed-in user can see and change: one shared notes document, a set of live
polls with per-user voting, and a running chat. Because every write lands in
Postgres via the service-role key, multiple browsers pointed at the same session
share state, and a "Refresh" in the UI pulls everyone else's latest edits. That
is what makes the page real-time-ish without websockets: cheap, free-tier
friendly polling over a shared table.

This module is intentionally thin and typed. It owns exactly one Supabase client
(built from ``settings.supabase_url`` + ``settings.supabase_service_role_key``)
and maps the classroom operations onto plain table calls. Every side-effect is
best-effort: a backend failure is logged and swallowed so the Streamlit UI keeps
rendering instead of crashing. Reads degrade to empty results; writes degrade to
no-ops.

The tables (public schema, no RLS, text ids) are created by the
``navilearn_classroom`` migration:

- ``classroom_sessions(id, title, created_by, created_at)``
- ``classroom_notes(id, session_id unique, content, updated_by, updated_at)``
- ``classroom_polls(id, session_id, question, options jsonb, created_at)``
- ``poll_votes(id, poll_id, voter_id, option_index, created_at)`` unique(poll_id, voter_id)
- ``chat_messages(id, session_id, author_id, author_name, text, created_at)``

The ``navilearn_classroom_solve`` migration adds the shared co-solving surface,
later extended with a shared problem statement and the latest run output so the
whole room (mentor + teacher + students) works one problem together:

- ``classroom_solve(session_id text primary key, content, language, updated_by,
  updated_at, problem, last_output, last_run_by)``
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from core.config import get_settings
from core.repo import Profile

_LOG = logging.getLogger(__name__)

# The demo runs against one shared classroom so every visitor lands in the same
# room without any setup. The id is fixed and human-readable on purpose.
DEFAULT_SESSION_ID = "main-classroom"
DEFAULT_SESSION_TITLE = "Main Classroom"


# --------------------------------------------------------------------------- #
# Typed rows
# --------------------------------------------------------------------------- #
@dataclass
class ClassroomSession:
    """A shared classroom room that scopes notes, polls, and chat."""

    id: str
    title: str = DEFAULT_SESSION_TITLE
    created_by: str = ""
    created_at: str = ""


@dataclass
class Poll:
    """A live poll: a question plus an ordered list of option labels."""

    id: str
    session_id: str
    question: str
    options: list[str] = field(default_factory=list)
    created_at: str = ""


@dataclass
class ChatMessage:
    """A single chat line posted to a session."""

    id: str
    session_id: str
    author_id: str
    author_name: str
    text: str
    created_at: str = ""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _new_id() -> str:
    """Return a fresh opaque string id."""

    return str(uuid.uuid4())


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""

    return datetime.now(timezone.utc).isoformat()


def _display_name(user: Optional[Profile]) -> str:
    """Return a human-friendly name for a user, or an empty string."""

    if user is None:
        return ""
    return user.full_name or user.email or user.id


@lru_cache(maxsize=1)
def _client():
    """Return a cached Supabase client built from settings, or ``None``.

    Cached because ``create_client`` opens a session we want to reuse across
    Streamlit reruns. Returns ``None`` (rather than raising) when the project is
    not configured or the client cannot be built, so every caller can degrade to
    a no-op instead of crashing the page.
    """

    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        _LOG.warning("Classroom disabled: Supabase URL/service key not configured.")
        return None
    try:
        from supabase import create_client  # local import: optional dependency

        return create_client(
            settings.supabase_url, settings.supabase_service_role_key
        )
    except Exception as exc:  # noqa: BLE001 - a bad client must not crash the UI.
        _LOG.warning("Classroom Supabase client unavailable: %s", exc)
        return None


def _table(name: str):
    """Return a table handle, or ``None`` if the client is unavailable."""

    client = _client()
    return client.table(name) if client is not None else None


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
def create_session(name: str, user: Optional[Profile]) -> str:
    """Create a co-solve classroom session and return its id (best-effort).

    Any role (mentor, teacher, or student) may open a room. The creator's
    display name is recorded on ``created_by`` so the session picker can show
    who started each room. A fresh id is always returned even if the insert
    fails, so the caller can proceed; the row simply may not exist server-side
    in that degraded case.
    """

    session_id = _new_id()
    table = _table("classroom_sessions")
    if table is None:
        return session_id
    try:
        table.insert(
            {
                "id": session_id,
                "title": (name or DEFAULT_SESSION_TITLE).strip() or DEFAULT_SESSION_TITLE,
                "created_by": _display_name(user) or "Anonymous",
                "created_at": _now_iso(),
            }
        ).execute()
    except Exception as exc:  # noqa: BLE001 - creation is best-effort.
        _LOG.warning("create_session failed: %s", exc)
    return session_id


def list_sessions() -> list[dict[str, str]]:
    """Return every co-solve session as ``{id, name, created_by, created_at}``.

    Ensures the shared "Main Classroom" exists first, then lists all rooms
    newest-first. Reads degrade to just the default room (or an empty list when
    even that lookup fails) so the picker always has something to show.
    """

    default_id = get_or_create_default_session()
    default_row = {
        "id": default_id,
        "name": DEFAULT_SESSION_TITLE,
        "created_by": "system",
        "created_at": "",
    }
    table = _table("classroom_sessions")
    if table is None:
        return [default_row]
    try:
        res = (
            table.select("*").order("created_at", desc=True).execute()
        )
    except Exception as exc:  # noqa: BLE001 - reads degrade to the default room.
        _LOG.warning("list_sessions failed: %s", exc)
        return [default_row]

    sessions: list[dict[str, str]] = []
    seen_default = False
    for row in res.data or []:
        sid = row.get("id") or ""
        if not sid:
            continue
        if sid == default_id:
            seen_default = True
        sessions.append(
            {
                "id": sid,
                "name": row.get("title") or DEFAULT_SESSION_TITLE,
                "created_by": row.get("created_by") or "",
                "created_at": row.get("created_at") or "",
            }
        )
    if not seen_default:
        sessions.append(default_row)
    return sessions or [default_row]


def get_or_create_default_session() -> str:
    """Return the id of the single shared "Main Classroom", creating it once.

    Idempotent: uses the fixed :data:`DEFAULT_SESSION_ID` and upserts on it, so
    concurrent first-time visitors all converge on the same room. Falls back to
    returning the fixed id even when the backend is unreachable so the page can
    still render (reads/writes then no-op).
    """

    table = _table("classroom_sessions")
    if table is None:
        return DEFAULT_SESSION_ID
    try:
        existing = (
            table.select("id").eq("id", DEFAULT_SESSION_ID).limit(1).execute()
        )
        if not (existing.data or []):
            table.upsert(
                {
                    "id": DEFAULT_SESSION_ID,
                    "title": DEFAULT_SESSION_TITLE,
                    "created_by": "system",
                    "created_at": _now_iso(),
                }
            ).execute()
    except Exception as exc:  # noqa: BLE001 - default room is best-effort.
        _LOG.warning("get_or_create_default_session failed: %s", exc)
    return DEFAULT_SESSION_ID


# --------------------------------------------------------------------------- #
# Shared notes (one document per session)
# --------------------------------------------------------------------------- #
def get_notes(session_id: str) -> dict[str, str]:
    """Return the shared notes doc for a session.

    Always returns a dict with ``content``, ``updated_by``, and ``updated_at``
    keys (empty strings when there is no note yet or the backend is down).
    """

    empty = {"content": "", "updated_by": "", "updated_at": ""}
    table = _table("classroom_notes")
    if table is None:
        return empty
    try:
        res = (
            table.select("*").eq("session_id", session_id).limit(1).execute()
        )
        rows = res.data or []
        if not rows:
            return empty
        row = rows[0]
        return {
            "content": row.get("content") or "",
            "updated_by": row.get("updated_by") or "",
            "updated_at": row.get("updated_at") or "",
        }
    except Exception as exc:  # noqa: BLE001 - reads degrade to empty.
        _LOG.warning("get_notes failed: %s", exc)
        return empty


def save_notes(session_id: str, content: str, user: Optional[Profile]) -> bool:
    """Upsert the shared notes doc for a session. Returns success (best-effort).

    There is at most one note row per session (unique on ``session_id``), so
    this upserts on that column and records who last edited it.
    """

    table = _table("classroom_notes")
    if table is None:
        return False
    editor = ""
    if user is not None:
        editor = user.full_name or user.email or user.id
    try:
        table.upsert(
            {
                "id": f"notes-{session_id}",
                "session_id": session_id,
                "content": content or "",
                "updated_by": editor,
                "updated_at": _now_iso(),
            },
            on_conflict="session_id",
        ).execute()
        return True
    except Exception as exc:  # noqa: BLE001 - writes degrade to no-op.
        _LOG.warning("save_notes failed: %s", exc)
        return False


# --------------------------------------------------------------------------- #
# Polls and votes
# --------------------------------------------------------------------------- #
def create_poll(session_id: str, question: str, options: list[str]) -> Optional[str]:
    """Create a poll with the given question and option labels. Returns its id.

    Blank option labels are dropped. Returns ``None`` when the question or
    options are empty, or the write fails.
    """

    question = (question or "").strip()
    clean_options = [o.strip() for o in (options or []) if o and o.strip()]
    if not question or len(clean_options) < 2:
        return None
    table = _table("classroom_polls")
    if table is None:
        return None
    poll_id = _new_id()
    try:
        table.insert(
            {
                "id": poll_id,
                "session_id": session_id,
                "question": question,
                "options": clean_options,
                "created_at": _now_iso(),
            }
        ).execute()
        return poll_id
    except Exception as exc:  # noqa: BLE001 - poll creation is best-effort.
        _LOG.warning("create_poll failed: %s", exc)
        return None


def list_polls(session_id: str) -> list[Poll]:
    """Return the polls for a session, newest first (best-effort)."""

    table = _table("classroom_polls")
    if table is None:
        return []
    try:
        res = (
            table.select("*")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001 - reads degrade to empty.
        _LOG.warning("list_polls failed: %s", exc)
        return []
    polls: list[Poll] = []
    for row in res.data or []:
        options = row.get("options") or []
        if isinstance(options, str):
            # jsonb normally decodes to a list; guard the string edge case.
            import json

            try:
                options = json.loads(options)
            except (ValueError, TypeError):
                options = []
        if not isinstance(options, list):
            options = []
        polls.append(
            Poll(
                id=row["id"],
                session_id=row.get("session_id", session_id),
                question=row.get("question", ""),
                options=[str(o) for o in options],
                created_at=row.get("created_at", ""),
            )
        )
    return polls


def vote(poll_id: str, voter_id: str, option_index: int) -> bool:
    """Record (or change) a user's vote on a poll. Returns success.

    Upserts on ``(poll_id, voter_id)`` so re-voting moves the ballot instead of
    stuffing the box: one vote per voter per poll.
    """

    table = _table("poll_votes")
    if table is None:
        return False
    try:
        table.upsert(
            {
                "id": f"{poll_id}:{voter_id}",
                "poll_id": poll_id,
                "voter_id": voter_id,
                "option_index": int(option_index),
                "created_at": _now_iso(),
            },
            on_conflict="poll_id,voter_id",
        ).execute()
        return True
    except Exception as exc:  # noqa: BLE001 - voting is best-effort.
        _LOG.warning("vote failed: %s", exc)
        return False


def poll_results(poll_id: str) -> list[dict[str, Any]]:
    """Return per-option vote counts for a poll as ``[{option, count}, ...]``.

    Options are listed in their poll order (even those with zero votes). Votes
    whose ``option_index`` is out of range are ignored. Returns an empty list
    when the poll is missing or the backend is down.
    """

    polls_table = _table("classroom_polls")
    votes_table = _table("poll_votes")
    if polls_table is None or votes_table is None:
        return []

    # Fetch the poll to recover its option labels and their order.
    try:
        poll_res = (
            polls_table.select("options").eq("id", poll_id).limit(1).execute()
        )
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("poll_results poll lookup failed: %s", exc)
        return []
    poll_rows = poll_res.data or []
    if not poll_rows:
        return []
    options = poll_rows[0].get("options") or []
    if isinstance(options, str):
        import json

        try:
            options = json.loads(options)
        except (ValueError, TypeError):
            options = []
    if not isinstance(options, list):
        options = []
    labels = [str(o) for o in options]

    counts = [0] * len(labels)
    try:
        vote_res = (
            votes_table.select("option_index").eq("poll_id", poll_id).execute()
        )
        for row in vote_res.data or []:
            idx = row.get("option_index")
            if isinstance(idx, int) and 0 <= idx < len(counts):
                counts[idx] += 1
    except Exception as exc:  # noqa: BLE001 - counts degrade to zeros.
        _LOG.warning("poll_results vote count failed: %s", exc)

    return [{"option": label, "count": counts[i]} for i, label in enumerate(labels)]


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
def post_message(session_id: str, user: Optional[Profile], text: str) -> bool:
    """Append a chat message to a session. Returns success (best-effort).

    Empty messages are dropped. The author name is snapshotted onto the row so
    the chat log reads well even if profiles change later.
    """

    text = (text or "").strip()
    if not text:
        return False
    table = _table("chat_messages")
    if table is None:
        return False
    author_id = user.id if user else ""
    author_name = ""
    if user is not None:
        author_name = user.full_name or user.email or user.id
    try:
        table.insert(
            {
                "id": _new_id(),
                "session_id": session_id,
                "author_id": author_id,
                "author_name": author_name or "Anonymous",
                "text": text,
                "created_at": _now_iso(),
            }
        ).execute()
        return True
    except Exception as exc:  # noqa: BLE001 - posting is best-effort.
        _LOG.warning("post_message failed: %s", exc)
        return False


def list_messages(session_id: str, limit: int = 50) -> list[ChatMessage]:
    """Return the most recent messages for a session in chronological order.

    Fetches the newest ``limit`` rows and returns them oldest-first so the UI can
    render top-to-bottom like a normal chat transcript.
    """

    table = _table("chat_messages")
    if table is None:
        return []
    try:
        res = (
            table.select("*")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(max(1, int(limit)))
            .execute()
        )
    except Exception as exc:  # noqa: BLE001 - reads degrade to empty.
        _LOG.warning("list_messages failed: %s", exc)
        return []
    rows = list(res.data or [])
    rows.reverse()  # newest-first fetch -> oldest-first for display
    return [
        ChatMessage(
            id=row["id"],
            session_id=row.get("session_id", session_id),
            author_id=row.get("author_id", ""),
            author_name=row.get("author_name", "") or "Anonymous",
            text=row.get("text", ""),
            created_at=row.get("created_at", ""),
        )
        for row in rows
    ]


# --------------------------------------------------------------------------- #
# Co-solving (one shared code/text workspace per session)
# --------------------------------------------------------------------------- #
# The default language shown when a room has never been co-solved in yet.
DEFAULT_SOLVE_LANGUAGE = "python"


def _empty_solve() -> dict[str, str]:
    """Return a blank co-solve workspace dict with every key present."""

    return {
        "content": "",
        "language": DEFAULT_SOLVE_LANGUAGE,
        "updated_by": "",
        "updated_at": "",
        "problem": "",
        "last_output": "",
        "last_run_by": "",
    }


def _solve_from_row(row: dict[str, Any]) -> dict[str, str]:
    """Normalise a ``classroom_solve`` DB row into the workspace dict shape."""

    return {
        "content": row.get("content") or "",
        "language": row.get("language") or DEFAULT_SOLVE_LANGUAGE,
        "updated_by": row.get("updated_by") or "",
        "updated_at": row.get("updated_at") or "",
        "problem": row.get("problem") or "",
        "last_output": row.get("last_output") or "",
        "last_run_by": row.get("last_run_by") or "",
    }


def get_solve(session_id: str) -> dict[str, str]:
    """Return the shared co-solve workspace for a session.

    Always returns a dict with ``content``, ``language``, ``updated_by``,
    ``updated_at``, ``problem``, ``last_output``, and ``last_run_by`` keys
    (falling back to empty content and :data:`DEFAULT_SOLVE_LANGUAGE` when there
    is nothing saved yet or the backend is down). This is the collaborative
    surface: one shared problem statement, one shared editor, and the most
    recent run output, all co-edited by mentor, teacher, and students.
    """

    table = _table("classroom_solve")
    if table is None:
        return _empty_solve()
    try:
        res = table.select("*").eq("session_id", session_id).limit(1).execute()
        rows = res.data or []
        if not rows:
            return _empty_solve()
        return _solve_from_row(rows[0])
    except Exception as exc:  # noqa: BLE001 - reads degrade to empty.
        _LOG.warning("get_solve failed: %s", exc)
        return _empty_solve()


def save_solve(
    session_id: str,
    content: str,
    language: str,
    user: Optional[Profile],
    problem: Optional[str] = None,
    base_updated_at: Optional[str] = None,
) -> tuple[str, dict[str, str]]:
    """Upsert the shared co-solve workspace, with optimistic concurrency.

    There is at most one workspace row per session (``session_id`` is the
    primary key), so this upserts on that column and records who last edited it.

    Concurrency: when ``base_updated_at`` is provided it is the ``updated_at``
    the caller loaded. Before writing, the stored ``updated_at`` is compared; if
    the stored value is newer (someone else saved in the meantime), no write
    happens and ``("conflict", current_row)`` is returned so the caller can show
    the other person's latest content instead of silently overwriting it. When
    ``base_updated_at`` is ``None`` the write is unconditional (backward
    compatible).

    When ``problem`` is not ``None`` the shared problem statement is persisted
    alongside the content; when it is ``None`` the stored problem is preserved.

    Returns ``("ok", row)`` on a successful write, ``("conflict", current_row)``
    when a newer save is detected, and ``("error", empty_row)`` when the backend
    is unavailable or the write fails (best-effort: it logs, never raises).
    """

    table = _table("classroom_solve")
    if table is None:
        return ("error", _empty_solve())

    current = get_solve(session_id) if base_updated_at is not None else None
    if base_updated_at is not None and current is not None:
        stored = current.get("updated_at") or ""
        # A stored timestamp strictly newer than the caller's base means someone
        # saved since the caller loaded: refuse and hand back the current row.
        if stored and stored > base_updated_at:
            return ("conflict", current)

    editor = _display_name(user)
    payload: dict[str, Any] = {
        "session_id": session_id,
        "content": content or "",
        "language": (language or DEFAULT_SOLVE_LANGUAGE).strip()
        or DEFAULT_SOLVE_LANGUAGE,
        "updated_by": editor,
        "updated_at": _now_iso(),
    }
    if problem is not None:
        payload["problem"] = problem
    try:
        res = table.upsert(payload, on_conflict="session_id").execute()
        rows = res.data or []
        row = _solve_from_row(rows[0]) if rows else get_solve(session_id)
        return ("ok", row)
    except Exception as exc:  # noqa: BLE001 - writes degrade to no-op.
        _LOG.warning("save_solve failed: %s", exc)
        return ("error", _empty_solve())


def save_run_output(session_id: str, output: str, user: Optional[Profile]) -> bool:
    """Persist the most recent run output so the whole room sees it live.

    Updates only ``last_output`` and ``last_run_by`` (it deliberately leaves
    ``updated_at`` untouched so publishing a run does not trip the editor's
    optimistic-concurrency check). Best-effort: logs and returns ``False`` on
    failure rather than crashing the room.
    """

    table = _table("classroom_solve")
    if table is None:
        return False
    try:
        table.upsert(
            {
                "session_id": session_id,
                "last_output": output or "",
                "last_run_by": _display_name(user) or "Anonymous",
            },
            on_conflict="session_id",
        ).execute()
        return True
    except Exception as exc:  # noqa: BLE001 - writes degrade to no-op.
        _LOG.warning("save_run_output failed: %s", exc)
        return False


# The Python interpreter used by :func:`run_python`. Runs in a child process so
# a crash, infinite loop, or exit() in the shared code cannot take down Streamlit.
_RUN_PYTHON_BIN = "/mnt/data/astra/projects/jobprep/navilearn/.venv/bin/python"

# Cap on captured stdout/stderr so a runaway print loop cannot balloon the row
# or the UI. Output past this is truncated with a visible marker.
_RUN_OUTPUT_LIMIT = 20_000


def _truncate(text: str, limit: int = _RUN_OUTPUT_LIMIT) -> str:
    """Return ``text`` clipped to ``limit`` chars with a truncation marker."""

    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more chars]"


def run_python(code: str, timeout: int = 5) -> dict[str, Any]:
    """Execute ``code`` in a separate Python subprocess and capture its output.

    DANGER / DEMO ONLY: this runs the supplied code ON THE HOST with the repo
    virtualenv. The child gets a MINIMAL environment (no app secrets, no
    PYTHONPATH) and runs from /tmp, so it cannot read the Supabase key or import
    app modules, but it still has no real sandbox: no resource limits beyond a
    wall-clock ``timeout``, and filesystem and network access. It is safe enough
    for a single-tenant local classroom demo where participants are trusted, but
    it MUST be gated behind a real sandbox (container / seccomp / nsjail / an
    isolated runner service) before any untrusted or multi-tenant deployment.

    The code is written to a :class:`~tempfile.NamedTemporaryFile` and run with
    ``subprocess.run([python, file], capture_output=True, text=True,
    timeout=timeout)`` and no shell. Returns ``{"ok", "stdout", "stderr"}``:
    ``ok`` is ``True`` only on a clean (exit code 0) run within the timeout. On
    :class:`subprocess.TimeoutExpired` it returns ``ok=False`` with a
    "timed out after Ns" stderr. Best-effort: any unexpected error is caught and
    surfaced in ``stderr`` rather than raised. Very long output is truncated.
    """

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(code or "")
            tmp_path = fh.name

        proc = subprocess.run(
            [_RUN_PYTHON_BIN, tmp_path],
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout)),
            cwd="/tmp",
            # NEVER inherit the app's process environment: it holds the Supabase
            # service key and other secrets that executed code could otherwise
            # read via os.environ. A minimal env (no PYTHONPATH, so app modules
            # and their secrets are not importable either) is all Python needs.
            env={"PATH": "/usr/bin:/bin", "HOME": "/tmp", "LANG": "C.UTF-8"},
        )
        return {
            "ok": proc.returncode == 0,
            "stdout": _truncate(proc.stdout or ""),
            "stderr": _truncate(proc.stderr or ""),
        }
    except subprocess.TimeoutExpired as exc:
        partial = ""
        if exc.stdout:
            partial = exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode("utf-8", "replace")
        return {
            "ok": False,
            "stdout": _truncate(partial),
            "stderr": f"timed out after {max(1, int(timeout))}s",
        }
    except Exception as exc:  # noqa: BLE001 - the runner must never raise.
        _LOG.warning("run_python failed: %s", exc)
        return {"ok": False, "stdout": "", "stderr": f"runner error: {exc}"}
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:  # noqa: BLE001 - cleanup is best-effort.
                pass


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export_summary(session_id: str) -> str:
    """Render the whole session (notes + polls with results + chat) as Markdown.

    Everything is a best-effort read, so a partially-available backend still
    produces a usable summary rather than an error.
    """

    lines: list[str] = ["# Classroom session summary", ""]

    notes = get_notes(session_id)
    lines.append("## Shared notes")
    content = (notes.get("content") or "").strip()
    if content:
        lines.append(content)
        editor = notes.get("updated_by") or ""
        stamp = notes.get("updated_at") or ""
        meta = " · ".join(p for p in (f"last edited by {editor}" if editor else "", stamp) if p)
        if meta:
            lines.append("")
            lines.append(f"_{meta}_")
    else:
        lines.append("_No notes yet._")
    lines.append("")

    polls = list_polls(session_id)
    lines.append("## Polls")
    if not polls:
        lines.append("_No polls yet._")
    else:
        for poll in polls:
            lines.append(f"### {poll.question}")
            results = poll_results(poll.id)
            total = sum(r["count"] for r in results) or 0
            for row in results:
                count = row["count"]
                pct = round(100.0 * count / total, 1) if total else 0.0
                lines.append(f"- {row['option']}: {count} vote(s) ({pct}%)")
            lines.append("")
    lines.append("")

    messages = list_messages(session_id, limit=50)
    lines.append("## Recent chat")
    if not messages:
        lines.append("_No messages yet._")
    else:
        for msg in messages:
            stamp = (msg.created_at or "")[:19].replace("T", " ")
            prefix = f"**{msg.author_name}**"
            if stamp:
                prefix += f" ({stamp})"
            lines.append(f"- {prefix}: {msg.text}")
    lines.append("")

    return "\n".join(lines)
