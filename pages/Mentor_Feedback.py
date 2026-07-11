"""Mentor Feedback: the student's own page for their two-way mentor thread.

This extracts the mentor conversation out of the crowded Home dashboard and
gives it room to breathe on its own page. The student sees the full thread
(mentor messages and their own replies, labelled by author via
``st.chat_message``) and can reply inline. Messages are read via
:func:`core.mentoring.list_notes` keyed by ``user.id`` (the runtime id the
student carries, which the mentoring layer canonicalizes to match how a mentor
stored them); a reply is saved via :func:`core.mentoring.save_note` with the
``student`` author role.

The mentoring module is imported defensively so a momentary import or backend
failure degrades to a caption rather than crashing the page, and sending is
best-effort.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import streamlit as st

from core.session import require_user

_LOG = logging.getLogger(__name__)


def _fmt_date(value: str) -> str:
    """Return a short, human-friendly date from an ISO timestamp.

    Falls back to the raw string (or a dash) when the value cannot be parsed, so
    a malformed timestamp never breaks a message.
    """

    if not value:
        return "n/a"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return value[:10] if len(value) >= 10 else value
    return parsed.strftime("%d %b %Y")


def _render_thread(user: Any) -> None:
    """Render the two-way mentor thread and the student's reply box.

    Reads the thread via :func:`core.mentoring.list_notes` and shows each message
    as a chat bubble labelled by author (student vs mentor). Below the thread a
    reply box saves the student's message via :func:`core.mentoring.save_note`
    and reruns so the new reply appears. All backend calls are best-effort.
    """

    try:
        from core.mentoring import list_notes, save_note
    except Exception as exc:  # noqa: BLE001 - never crash the page over mentoring.
        _LOG.warning("mentor feedback unavailable: %s", exc)
        st.info("Mentor feedback is momentarily unavailable. Try again shortly.")
        return

    try:
        notes = list_notes(user.id)
    except Exception as exc:  # noqa: BLE001 - reads degrade to empty.
        _LOG.warning("mentor feedback thread unavailable: %s", exc)
        st.info("Mentor feedback is momentarily unavailable. Try again shortly.")
        notes = []

    if not notes:
        st.info("No mentor feedback yet.")
        st.caption(
            "Once a mentor reviews your work, their notes will appear here and "
            "you can reply to them."
        )
    else:
        for note in notes:
            role = (note.get("author_role") or "mentor").strip() or "mentor"
            is_student = role == "student"
            avatar = "🎓" if is_student else "🧑‍🏫"
            chat_role = "user" if is_student else "assistant"
            author = (note.get("author_name") or "").strip() or (
                "You" if is_student else "Mentor"
            )
            text = (note.get("text") or "").strip()
            with st.chat_message(chat_role, avatar=avatar):
                if text:
                    st.write(text)
                st.caption(f"{author} · {_fmt_date(note.get('created_at', ''))}")

    with st.form("mentor-feedback-reply-form", clear_on_submit=True):
        reply_text = st.text_area(
            "Reply to your mentor",
            key="mentor-feedback-reply-text",
            placeholder="Ask a question or share how it is going.",
            height=100,
        )
        submitted = st.form_submit_button("Send reply")
    if submitted:
        author_name = (user.full_name or "").strip() or user.email or "Student"
        try:
            ok = save_note(user.id, user.id, author_name, reply_text, "student")
        except Exception as exc:  # noqa: BLE001 - never crash over a send.
            _LOG.warning("mentor reply failed: %s", exc)
            ok = False
        if ok:
            st.success("Reply sent.")
            st.rerun()
        else:
            st.warning("Could not send your reply. Enter some text and try again.")


def main() -> None:
    """Gate on login, then render the student's mentor conversation."""

    st.set_page_config(page_title="Mentor Feedback | NaviLearn", page_icon="🤝")
    user = require_user()
    st.title("Mentor Feedback")
    _render_thread(user)


main()
