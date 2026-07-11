"""Public read-only viewer for a shared NaviLearn study set.

A study set is a learner's "notes". This page renders one from an unlisted
share link of the form ``/Shared?s=<id>`` without requiring a login: anyone
holding the link can view the notes, but nothing here can be edited. The set
id is read from ``st.query_params`` and loaded through the same repository the
Study Studio writes to, so the read-only view mirrors the saved-set layout
(overall summary, topics, flashcards, per-topic summaries and the concept
graph).

The page is deliberately defensive: a missing id, a missing set, or a partial
``content`` dict each degrade to a friendly message rather than an error, and a
signed-in session is welcome but never required.
"""

from __future__ import annotations

import logging
from typing import Optional

import streamlit as st

from core.config import get_settings
from core.notes import Note, get_public_note
from core.repo import StudySet, get_repo
from core.session import current_user

_LOG = logging.getLogger(__name__)


def _read_share_id() -> str:
    """Return the ``s`` query parameter (the shared set id), or an empty string.

    Best-effort: any failure to read the query params yields an empty string so
    the page falls back to its "how sharing works" explanation.
    """

    try:
        raw = st.query_params.get("s", "")
    except Exception as exc:  # noqa: BLE001 - query params are best-effort.
        _LOG.warning("Could not read share query params: %s", exc)
        return ""
    return (raw or "").strip()


def _read_note_id() -> str:
    """Return the ``n`` query parameter (the shared note id), or an empty string.

    Best-effort: any failure to read the query params yields an empty string so
    the page falls back to its "how sharing works" explanation.
    """

    try:
        raw = st.query_params.get("n", "")
    except Exception as exc:  # noqa: BLE001 - query params are best-effort.
        _LOG.warning("Could not read shared note query params: %s", exc)
        return ""
    return (raw or "").strip()


def _load_shared_note(note_id: str) -> Optional[Note]:
    """Load a public note by id, or ``None`` when missing, private or failing.

    Delegates to :func:`core.notes.get_public_note`, which only returns rows
    flagged ``is_public``. A backend failure surfaces as "not found" rather than
    crashing the public page.
    """

    try:
        return get_public_note(note_id)
    except Exception as exc:  # noqa: BLE001 - the public viewer never crashes.
        _LOG.warning("Could not load shared note %s: %s", note_id, exc)
        return None


def _load_shared_set(set_id: str) -> Optional[StudySet]:
    """Load a study set by id through the repository, or ``None`` on any error.

    Never requires or checks a user: a share link is intentionally unlisted but
    public. A repository failure is logged and surfaces as "not found" rather
    than crashing the public page.
    """

    try:
        repo = get_repo(get_settings())
        return repo.get_study_set(set_id)
    except Exception as exc:  # noqa: BLE001 - the public viewer never crashes.
        _LOG.warning("Could not load shared study set %s: %s", set_id, exc)
        return None


def _render_not_found() -> None:
    """Explain that the requested share link resolved to nothing."""

    st.warning(
        "This shared note was not found or is no longer available. "
        "Ask whoever sent the link to share it again."
    )


def _render_note_not_found() -> None:
    """Explain that the requested note id resolved to nothing public."""

    st.warning("This note was not found or is not shared.")


def _render_shared_note(note: Note) -> None:
    """Render a single public note read-only: title, meta caption, body.

    Strictly read-only: the title, a "Shared note" caption carrying the last
    updated date and any tags, then the Markdown body. Missing fields degrade to
    friendly defaults so a sparse note still renders.
    """

    st.title(note.title or "Untitled note")

    updated = note.updated_at or "an earlier session"
    caption = f"Shared note  |  updated {updated}"
    tags = (note.tags or "").strip()
    if tags:
        caption += f"  |  tags: {tags}"
    st.caption(caption)

    body = note.body or ""
    if body.strip():
        st.markdown(body)
    else:
        st.info("This note has no content yet.")


def _render_no_id() -> None:
    """Explain the share-link format when no set id is present."""

    st.info(
        "A share link looks like `/Shared?s=<id>`. Open **Study Studio** to "
        "build a study set from any document, video or article, then use its "
        "Share section to create a link like this one."
    )


def _render_shared_set(study_set: StudySet) -> None:
    """Render a study set read-only from its persisted ``content`` dict.

    Mirrors the Study Studio saved-set layout but strictly read-only: no edit,
    open or download controls. Missing ``content`` keys degrade gracefully to
    empty defaults so a partially populated set still renders.
    """

    content = study_set.content or {}
    title = content.get("title") or study_set.title or "Untitled notes"

    st.title(title)
    created = study_set.created_at or "an earlier session"
    st.caption(f"Shared study notes  |  created {created}")

    st.markdown("### Summary")
    overall = content.get("summary_overall") or ""
    if overall:
        st.write(overall)
    else:
        st.info("No summary was saved for these notes.")

    st.markdown("### Topics")
    topics = content.get("topics", []) or []
    if topics:
        for topic in topics:
            st.markdown(f"- {topic}")
    else:
        st.caption("No topics were saved.")

    st.markdown("### Flashcards")
    cards = content.get("flashcards", []) or []
    if cards:
        for index, card in enumerate(cards, start=1):
            with st.container(border=True):
                st.markdown(f"**Q{index}. {card.get('front', '')}**")
                if card.get("topic"):
                    st.caption(f"Topic: {card.get('topic')}")
                with st.expander("Show answer"):
                    st.write(card.get("back", ""))
    else:
        st.caption("No flashcards were saved.")

    st.markdown("### Summaries by topic")
    per_topic = content.get("summary_per_topic", {}) or {}
    if per_topic:
        for topic, text in per_topic.items():
            with st.expander(topic):
                st.write(text or "(no summary)")
    else:
        st.caption("No per-topic summaries were saved.")

    st.markdown("### Concept graph")
    graph_dot = content.get("graph_dot") or ""
    if graph_dot.strip():
        st.graphviz_chart(graph_dot, width="stretch")
    else:
        st.caption("No concept graph was saved for these notes.")


def main() -> None:
    """Render the public shared-notes viewer."""

    st.set_page_config(page_title="Shared notes | NaviLearn", page_icon="🔗")

    # A signed-in session is welcome but never required for a public link.
    current_user()

    note_id = _read_note_id()
    if note_id:
        note = _load_shared_note(note_id)
        if note is None:
            st.title("Shared note")
            _render_note_not_found()
            return
        _render_shared_note(note)
        return

    set_id = _read_share_id()
    if not set_id:
        st.title("Shared notes")
        _render_no_id()
        return

    study_set = _load_shared_set(set_id)
    if study_set is None:
        st.title("Shared notes")
        _render_not_found()
        return

    _render_shared_set(study_set)


main()
