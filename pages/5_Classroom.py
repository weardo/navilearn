"""Live Classroom Collaboration (Ch5): one shared room, Supabase-backed.

Every signed-in user who opens this page joins the same "Main Classroom" session
and sees the same three collaborative surfaces backed by Postgres:

- Shared notes: a single document the whole room edits, with a last-editor stamp.
- Live poll: create a poll, cast one vote (changeable), and watch a live bar
  chart of the tally.
- Chat: a running message log with a send box.

State lives in Supabase (service-role writes, no RLS), so opening the page in two
browsers and hitting "Refresh" shows each other's edits. This is the free-tier
friendly, low-latency way to get near-real-time collaboration without websockets:
a shared table plus an explicit refresh. The page is a thin UI over
:mod:`core.classroom`; all backend calls there are best-effort so a hiccup logs
and continues instead of crashing the room.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from core.classroom import (
    create_poll,
    export_summary,
    get_notes,
    get_or_create_default_session,
    list_messages,
    list_polls,
    poll_results,
    post_message,
    save_notes,
    vote,
)
from core.session import require_user

_LOG = logging.getLogger(__name__)

# How long the auto-refresh toggle waits between reruns, in seconds.
_AUTO_REFRESH_SECONDS = 3


def _render_notes(session_id: str, user) -> None:
    """Shared notes: a text area + save button, showing the last editor.

    The editable text is owned by ``st.session_state`` (seeded once from the DB),
    never rebound to the DB copy on rerun. That keeps an auto-refresh, or any
    other rerun, from clobbering text the user is in the middle of typing. A
    "Pull latest" button is the explicit way to overwrite the local draft with the
    room's saved copy.
    """

    _NOTES_KEY = "classroom_notes_input"

    st.subheader("Shared notes")
    notes = get_notes(session_id)
    db_content = notes.get("content") or ""
    # Seed the local draft from the DB exactly once. After that the widget owns
    # its own state via ``key`` so reruns never overwrite in-progress typing.
    if _NOTES_KEY not in st.session_state:
        st.session_state[_NOTES_KEY] = db_content

    editor = notes.get("updated_by") or ""
    stamp = (notes.get("updated_at") or "")[:19].replace("T", " ")
    if editor or stamp:
        meta = " · ".join(p for p in (f"Last edited by {editor}" if editor else "", stamp) if p)
        st.caption(meta)
    else:
        st.caption("No edits yet. Be the first to write.")

    if st.button("Pull latest", width="stretch", key="pull_notes_btn"):
        st.session_state[_NOTES_KEY] = db_content
        st.rerun()

    st.text_area(
        "Room notes (shared by everyone)",
        height=260,
        key=_NOTES_KEY,
    )
    if st.button("Save notes", type="primary", width="stretch", key="save_notes_btn"):
        if save_notes(session_id, st.session_state.get(_NOTES_KEY, ""), user):
            st.success("Notes saved for the whole room.")
        else:
            st.warning("Could not save notes right now. Try again after a refresh.")


def _render_poll_creator(session_id: str) -> None:
    """A form to create a poll from a question + comma-separated options."""

    with st.form("create_poll_form", clear_on_submit=True):
        question = st.text_input("New poll question", placeholder="Which topic next?")
        options_raw = st.text_input(
            "Options (comma-separated)", placeholder="Recursion, Big-O, Graphs"
        )
        submitted = st.form_submit_button("Create poll", width="stretch")
    if submitted:
        options = [o.strip() for o in options_raw.split(",") if o.strip()]
        if not question.strip() or len(options) < 2:
            st.warning("Give a question and at least two comma-separated options.")
        elif create_poll(session_id, question, options):
            st.success("Poll created.")
        else:
            st.warning("Could not create the poll right now.")


def _render_poll(poll, user) -> None:
    """Render one poll: a vote widget and a live results bar chart."""

    st.markdown(f"**{poll.question}**")
    if not poll.options:
        st.caption("This poll has no options.")
        return

    choice = st.radio(
        "Your vote",
        options=list(range(len(poll.options))),
        format_func=lambda i: poll.options[i],
        key=f"vote_choice_{poll.id}",
    )
    if st.button("Vote", key=f"vote_btn_{poll.id}"):
        if vote(poll.id, user.id, int(choice)):
            st.success("Vote recorded. You can change it any time.")
        else:
            st.warning("Could not record your vote right now.")

    results = poll_results(poll.id)
    total = sum(r["count"] for r in results)
    if results:
        chart_df = pd.DataFrame(
            {"votes": [r["count"] for r in results]},
            index=[r["option"] for r in results],
        )
        st.bar_chart(chart_df)
        st.caption(f"{total} vote(s) total")
    st.divider()


def _render_polls(session_id: str, user) -> None:
    """Live poll section: creator form plus every poll in the room."""

    st.subheader("Live polls")
    _render_poll_creator(session_id)
    polls = list_polls(session_id)
    if not polls:
        st.info("No polls yet. Create the first one above.")
        return
    for poll in polls:
        _render_poll(poll, user)


def _render_chat(session_id: str, user) -> None:
    """Chat section: recent messages plus a send box."""

    st.subheader("Chat")
    messages = list_messages(session_id, limit=50)
    if not messages:
        st.caption("No messages yet. Say hello.")
    else:
        for msg in messages:
            stamp = (msg.created_at or "")[:19].replace("T", " ")
            with st.chat_message("user"):
                who = msg.author_name or "Anonymous"
                st.markdown(f"**{who}**  \n{msg.text}")
                if stamp:
                    st.caption(stamp)

    with st.form("chat_form", clear_on_submit=True):
        text = st.text_input("Message", placeholder="Type a message for the room")
        sent = st.form_submit_button("Send", width="stretch")
    if sent:
        if post_message(session_id, user, text):
            st.rerun()
        else:
            st.warning("Message not sent. Make sure it is not empty.")


def main() -> None:
    """Entry point: gate on login, join the shared room, render the surfaces."""

    st.set_page_config(page_title="Classroom | NaviLearn", page_icon="🧑‍🏫")
    user = require_user()
    st.title("Live Classroom")
    st.caption(
        "Near-real-time collaboration on one shared room. State is saved in "
        "Supabase, so everyone sees the same notes, polls, and chat. Hit Refresh "
        "to pull the latest. Low latency and free-tier friendly. Looking for the "
        "shared coding workspace? It now lives on the Co-solve page."
    )

    session_id = get_or_create_default_session()

    top_left, top_mid, top_right = st.columns([1, 1, 1])
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
    with top_right:
        st.download_button(
            "Download session summary",
            data=export_summary(session_id),
            file_name="classroom_summary.md",
            mime="text/markdown",
            width="stretch",
            key="download_summary_btn",
        )
    st.caption("Shared state syncs through Supabase. Every save lands in Postgres.")

    notes_col, polls_col, chat_col = st.columns(3)
    with notes_col:
        _render_notes(session_id, user)
    with polls_col:
        _render_polls(session_id, user)
    with chat_col:
        _render_chat(session_id, user)

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
                interval=_AUTO_REFRESH_SECONDS * 1000, key="classroom_autorefresh"
            )
        except Exception as exc:  # noqa: BLE001 - degrade to manual refresh.
            _LOG.warning("Auto-refresh component unavailable: %s", exc)
            st.caption(
                "Auto-refresh is unavailable in this environment. "
                "Click the Refresh button above to pull the latest."
            )


main()
