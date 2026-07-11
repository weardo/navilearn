"""Co-solve (Ch5): a shared multi-stakeholder problem-solving workspace.

Mentor, teacher, and student(s) share one room that holds a problem statement, a
co-edited code editor, and the most recent run output, all in Supabase so every
browser converges. In-progress drafts live in ``st.session_state`` and are never
rebound to the DB on an auto-refresh; the DB copy is loaded on a room switch or
an explicit "Pull latest". Saving uses optimistic concurrency so a stale write
surfaces a conflict instead of silently clobbering a teammate's edit.

This page is a thin UI over :mod:`core.classroom`; all backend calls there are
best-effort so a hiccup logs and continues instead of crashing the room. Sync is
near-real-time: a shared table plus an explicit (or auto) refresh, the free-tier
friendly, low-latency way to collaborate without websockets.
"""

from __future__ import annotations

import logging

import streamlit as st

from core.classroom import (
    create_session,
    get_or_create_default_session,
    get_solve,
    list_sessions,
    run_python,
    save_run_output,
    save_solve,
)
from core.config import get_settings
from core.session import require_user

_LOG = logging.getLogger(__name__)

# Languages offered in the co-solve editor. "text" covers plain scratch work.
_SOLVE_LANGUAGES: list[str] = [
    "python",
    "javascript",
    "typescript",
    "java",
    "c",
    "cpp",
    "go",
    "rust",
    "sql",
    "bash",
    "text",
]

# How long the auto-refresh toggle waits between reruns, in seconds.
_AUTO_REFRESH_SECONDS = 3

# Wall-clock cap on the co-solve "Run (Python)" host subprocess, in seconds.
_RUN_TIMEOUT_SECONDS = 5

# Session-state keys owned by the co-solve workspace.
_SOLVE_SESSION_KEY = "cosolve_session_id"
_SOLVE_CONTENT_KEY = "solve_content_input"
_SOLVE_LANG_KEY = "solve_language_input"
_SOLVE_PROBLEM_KEY = "solve_problem_input"
_SOLVE_DRAFT_OWNER_KEY = "solve_draft_session"
_SOLVE_BASE_KEY = "solve_base_updated_at"
_SOLVE_CONFLICT_KEY = "solve_conflict"


def _seed_solve_drafts(solve: dict, *, session_id: str) -> None:
    """Overwrite the local drafts with a room's saved workspace.

    Called on a room switch, first load, or an explicit "Pull latest". Records
    which room the drafts belong to and the ``updated_at`` they were loaded at
    so the next Save can detect a concurrent edit (optimistic concurrency).
    """

    language = solve.get("language") or _SOLVE_LANGUAGES[0]
    if language not in _SOLVE_LANGUAGES:
        language = _SOLVE_LANGUAGES[0]
    st.session_state[_SOLVE_CONTENT_KEY] = solve.get("content") or ""
    st.session_state[_SOLVE_LANG_KEY] = language
    st.session_state[_SOLVE_PROBLEM_KEY] = solve.get("problem") or ""
    st.session_state[_SOLVE_BASE_KEY] = solve.get("updated_at") or ""
    st.session_state[_SOLVE_DRAFT_OWNER_KEY] = session_id
    st.session_state.pop(_SOLVE_CONFLICT_KEY, None)


def _render_solve_session_picker(default_session_id: str, user) -> str:
    """Room picker plus a "new room" control. Returns the selected session id.

    Any role can spin up a co-solve room. The chosen id is held in
    ``st.session_state`` so the workspace stays put across reruns and
    auto-refreshes.
    """

    sessions = list_sessions()
    ids = [s["id"] for s in sessions]
    name_by_id = {s["id"]: s.get("name") or s["id"] for s in sessions}
    creator_by_id = {s["id"]: s.get("created_by") or "" for s in sessions}

    current = st.session_state.get(_SOLVE_SESSION_KEY) or default_session_id
    if current not in ids:
        current = ids[0] if ids else default_session_id
    index = ids.index(current) if current in ids else 0

    picker_col, new_col = st.columns([2, 3])
    with picker_col:
        selected = st.selectbox(
            "Co-solve session",
            options=ids,
            index=index,
            format_func=lambda sid: name_by_id.get(sid, sid),
            help="Every room is a separate shared problem + solution + run output.",
        )
    st.session_state[_SOLVE_SESSION_KEY] = selected

    with new_col:
        with st.form("new_cosolve_session_form", clear_on_submit=True):
            new_name = st.text_input(
                "New co-solve session",
                placeholder="e.g. Recursion drill",
            )
            created = st.form_submit_button("Create session", width="stretch")
        if created:
            name = (new_name or "").strip()
            if not name:
                st.warning("Give the new session a name.")
            else:
                new_id = create_session(name, user)
                st.session_state[_SOLVE_SESSION_KEY] = new_id
                st.rerun()

    creator = creator_by_id.get(selected) or ""
    if creator:
        st.caption(f"Room: {name_by_id.get(selected, selected)} · started by {creator}")
    return selected


def _render_solve(default_session_id: str, user) -> None:
    """Co-solve: a shared multi-stakeholder problem-solving workspace.

    Mentor, teacher, and student(s) share one room that holds a problem
    statement, a co-edited code editor, and the most recent run output, all in
    Supabase so every browser converges. In-progress drafts live in
    ``st.session_state`` and are never rebound to the DB on an auto-refresh; the
    DB copy is loaded on a room switch or an explicit "Pull latest". Saving uses
    optimistic concurrency so a stale write surfaces a conflict instead of
    silently clobbering a teammate's edit.
    """

    st.caption(
        "A shared workspace for mentor, teacher, and students: one problem, one "
        "solution, one run output. Everyone edits the same room and sees the "
        "latest within seconds."
    )

    session_id = _render_solve_session_picker(default_session_id, user)

    solve = get_solve(session_id)

    # Seed drafts from the DB when we switch rooms or land here the first time.
    # After that the widgets own their state via ``key`` so reruns (including an
    # auto-refresh) never overwrite in-progress typing.
    if st.session_state.get(_SOLVE_DRAFT_OWNER_KEY) != session_id:
        _seed_solve_drafts(solve, session_id=session_id)

    # Presence: who last edited and who last ran the code.
    editor = solve.get("updated_by") or ""
    runner = solve.get("last_run_by") or ""
    stamp = (solve.get("updated_at") or "")[:19].replace("T", " ")
    presence_bits = []
    presence_bits.append(f"Last edited by {editor} ({stamp})" if editor else "No edits yet")
    if runner:
        presence_bits.append(f"last run by {runner}")
    st.caption(" · ".join(presence_bits))

    if st.button("Pull latest", width="stretch", key="pull_solve_btn"):
        _seed_solve_drafts(get_solve(session_id), session_id=session_id)
        st.rerun()

    # Shared problem statement.
    st.text_area(
        "Problem statement (shared by the room)",
        height=120,
        key=_SOLVE_PROBLEM_KEY,
        placeholder="State the problem everyone is solving together.",
    )

    # Shared editor + language.
    language = st.selectbox(
        "Language",
        options=_SOLVE_LANGUAGES,
        key=_SOLVE_LANG_KEY,
    )
    content = st.text_area(
        "Shared solution (edited by everyone)",
        height=260,
        key=_SOLVE_CONTENT_KEY,
    )
    problem = st.session_state.get(_SOLVE_PROBLEM_KEY, "")

    save_col, run_col = st.columns(2)
    with save_col:
        if st.button("Save", type="primary", width="stretch", key="save_solve_btn"):
            status, row = save_solve(
                session_id,
                content,
                language,
                user,
                problem=problem,
                base_updated_at=st.session_state.get(_SOLVE_BASE_KEY),
            )
            if status == "ok":
                st.session_state[_SOLVE_BASE_KEY] = row.get("updated_at") or ""
                st.session_state.pop(_SOLVE_CONFLICT_KEY, None)
                st.success("Saved for the whole room.")
            elif status == "conflict":
                st.session_state[_SOLVE_CONFLICT_KEY] = row
                st.warning(
                    "Someone else saved since you loaded this room. Your draft was "
                    "not written so it would not overwrite theirs. Review their "
                    "latest below, then Pull latest (adopts theirs) or Save again."
                )
            else:
                st.warning("Could not save right now. Try again after a refresh.")
    code_run_enabled = get_settings().enable_code_run
    with run_col:
        if code_run_enabled:
            if st.button("Run (Python)", width="stretch", key="run_solve_btn"):
                result = run_python(content, timeout=_RUN_TIMEOUT_SECONDS)
                parts = []
                if result.get("stdout"):
                    parts.append(result["stdout"])
                if result.get("stderr"):
                    parts.append(f"[stderr]\n{result['stderr']}")
                combined = "\n".join(parts) or "(no output)"
                save_run_output(session_id, combined, user)
                st.rerun()
        else:
            # Public/hosted deploy: never execute code on the host. Show a
            # disabled control so the layout and affordance stay intact.
            st.button(
                "Run (Python)",
                width="stretch",
                key="run_solve_btn",
                disabled=True,
            )
    if code_run_enabled:
        st.caption(
            f"Run executes on the host in a subprocess with a {_RUN_TIMEOUT_SECONDS}s "
            "timeout (local demo only). Everyone in the room sees the result below."
        )
    else:
        st.caption("Code execution is disabled on the hosted demo for safety.")

    # A stale-write conflict: show the other person's latest so nothing is lost.
    conflict = st.session_state.get(_SOLVE_CONFLICT_KEY)
    if conflict:
        their_editor = conflict.get("updated_by") or "someone"
        st.error(f"Conflict: {their_editor} has the newer saved version:")
        st.code(
            conflict.get("content") or "",
            language=None if language == "text" else language,
        )

    # Persisted run output: the whole room sees the latest run, live.
    st.markdown("**Latest run output**")
    last_output = solve.get("last_output") or ""
    last_run_by = solve.get("last_run_by") or ""
    if last_output:
        st.code(last_output, language="text")
        if last_run_by:
            st.caption(f"Run by {last_run_by}")
    else:
        st.caption("No run yet. Write Python above and press Run.")

    st.markdown("**Preview**")
    if content.strip():
        # st.code language is a hint; "text" renders as plain.
        st.code(content, language=None if language == "text" else language)
    else:
        st.caption("Nothing to preview yet.")


def main() -> None:
    """Entry point: gate on login, then render the co-solve workspace."""

    st.set_page_config(page_title="Co-solve | NaviLearn", page_icon="🧩")
    user = require_user()
    st.title("Co-solve")
    st.caption(
        "Near-real-time collaboration on a shared coding workspace. State is "
        "saved in Supabase, so everyone sees the same problem, solution, and run "
        "output. Hit Refresh to pull the latest. Low latency and free-tier friendly."
    )

    default_session_id = get_or_create_default_session()

    top_left, top_mid = st.columns([1, 1])
    with top_left:
        if st.button("🔄 Refresh", width="stretch", key="refresh_btn"):
            st.rerun()
    with top_mid:
        auto = st.toggle(
            f"Auto-refresh ({_AUTO_REFRESH_SECONDS}s)",
            value=True,
            key="auto_refresh_toggle",
            help="Poll Supabase every few seconds so shared edits appear on their own.",
        )
    st.caption("Shared state syncs through Supabase. Every save lands in Postgres.")

    st.divider()
    _render_solve(default_session_id, user)  # co-solve manages its own room selection

    # Near-real-time sync via the non-blocking ``streamlit-autorefresh``
    # component: it schedules a rerun from the browser without ever sleeping on
    # the single Streamlit script thread. A blocking sleep here would freeze the
    # whole UI (dead buttons) for the interval, so we never block the thread.
    # If the component is genuinely unavailable, we degrade to manual refresh and
    # tell the user, rather than blocking.
    if auto:
        try:
            from streamlit_autorefresh import st_autorefresh  # type: ignore

            st_autorefresh(
                interval=_AUTO_REFRESH_SECONDS * 1000, key="cosolve_autorefresh"
            )
        except Exception as exc:  # noqa: BLE001 - degrade to manual refresh.
            _LOG.warning("Auto-refresh component unavailable: %s", exc)
            st.caption(
                "Auto-refresh is unavailable in this environment. "
                "Click the Refresh button above to pull the latest."
            )


main()
