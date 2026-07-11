"""Notes (Evernote-style): a Supabase-backed personal notebook.

A two-pane note-taking app that lives alongside the rest of NaviLearn:

- Left pane: a "+ New note" button and the signed-in user's notes, newest edit
  first, each rendered as a clickable row showing the title, a short snippet of
  the body, and when it was last updated. Clicking a row opens it in the editor.
- Right pane: a Markdown editor for the selected (or brand-new) note: a title
  box, a Markdown body area, a tags box, a Save button (create for a new note,
  update for an existing one) and a Delete button, plus a small live Markdown
  preview of the body.
- Share section (existing notes only): two clearly-separated models. "Send to a
  person" delivers the note straight to another user's dashboard (a targeted,
  recipient-private share recorded in ``note_shares``). "Anyone with the link" is
  the separate public model: a button flips the note public and builds an unlisted
  ``/Shared?n=<id>`` link in a copyable code block, with a "Make private" control
  to revoke it.

The open note id lives in ``st.session_state['notes_selected_id']``; the sentinel
:data:`_NEW` means an unsaved new note. All backend work is delegated to
:mod:`core.notes` and :mod:`core.messaging`, whose calls are best-effort: a
backend hiccup logs and continues instead of crashing the page.
"""

from __future__ import annotations

import logging
from typing import Optional

import streamlit as st

from core.messaging import list_directory
from core.notes import (
    Note,
    create_note,
    delete_note,
    get_note,
    list_notes,
    set_public,
    share_note_with,
    update_note,
)
from core.session import require_user

_LOG = logging.getLogger(__name__)

# Session-state key holding the id of the note currently open in the editor.
_SELECTED_KEY = "notes_selected_id"

# Sentinel stored in :data:`_SELECTED_KEY` for an unsaved, brand-new note.
_NEW = "__new__"


# --------------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------------- #
def _short_time(iso: str) -> str:
    """Return a short ``YYYY-MM-DD HH:MM`` stamp from an ISO-8601 string.

    Best-effort: falls back to a leading slice or an empty string so a malformed
    or missing timestamp never breaks a row.
    """

    if not iso:
        return ""
    if "T" in iso and len(iso) >= 16:
        return f"{iso[:10]} {iso[11:16]}"
    return iso[:16]


def _snippet(body: str, limit: int = 60) -> str:
    """Return a one-line snippet of a note body, collapsing whitespace."""

    text = " ".join((body or "").split())
    if not text:
        return "(empty note)"
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _base_url() -> str:
    """Return the deployment's ``scheme://host`` for building share links.

    The host is read from the ``Host`` request header (falling back to the dev
    server address) and the scheme from ``X-Forwarded-Proto`` (falling back to
    ``http``). Best-effort: any failure to read headers yields the local default.
    """

    host = "localhost:8600"
    scheme = "http"
    try:
        headers = st.context.headers or {}
        host = headers.get("host", "localhost:8600") or "localhost:8600"
        scheme = headers.get("x-forwarded-proto", "") or "http"
    except Exception as exc:  # noqa: BLE001 - header read must never crash the page.
        _LOG.warning("Could not read request headers for share link: %s", exc)
    return f"{scheme}://{host}"


def _share_link(note_id: str) -> str:
    """Return the public share URL for a note id."""

    return f"{_base_url()}/Shared?n={note_id}"


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def _selected_id() -> Optional[str]:
    """Return the open note id (or :data:`_NEW`), or ``None`` if nothing is open."""

    return st.session_state.get(_SELECTED_KEY)


def _select(note_id: Optional[str]) -> None:
    """Open ``note_id`` (or :data:`_NEW`) in the editor and rerun both panes."""

    st.session_state[_SELECTED_KEY] = note_id
    st.rerun()


# --------------------------------------------------------------------------- #
# Left pane: the note list
# --------------------------------------------------------------------------- #
def _render_list(user, current_id: Optional[str]) -> None:
    """Render the "+ New note" button and the selectable list of the user's notes."""

    if st.button("+ New note", key="notes_new_btn", width="stretch", type="primary"):
        _select(_NEW)

    st.markdown("**Your notes**")
    notes = list_notes(user.id)
    if not notes:
        st.caption("No notes yet. Click **+ New note** to start.")
        return

    for note in notes:
        is_current = note.id == current_id
        with st.container(border=True):
            label = note.title.strip() or "Untitled note"
            if st.button(
                label,
                key=f"notes_row_{note.id}",
                width="stretch",
                type="primary" if is_current else "secondary",
                disabled=is_current,
            ):
                _select(note.id)
            stamp = _short_time(note.updated_at)
            snippet = _snippet(note.body)
            st.caption(f"{snippet}  ·  {stamp}" if stamp else snippet)


# --------------------------------------------------------------------------- #
# Right pane: the editor
# --------------------------------------------------------------------------- #
def _render_editor(user, current_id: Optional[str]) -> None:
    """Render the editor for the selected (or new) note, plus preview and sharing."""

    if current_id is None:
        st.info("Select a note on the left, or click **+ New note** to begin.")
        return

    is_new = current_id == _NEW
    note: Optional[Note] = None
    if not is_new:
        note = get_note(current_id)
        if note is None:
            st.warning("This note could not be loaded. It may have been deleted.")
            if st.button("Start a new note", key="notes_missing_new"):
                _select(_NEW)
            return
        if note.owner_id and note.owner_id != user.id:
            st.warning("This note belongs to someone else.")
            return

    st.markdown("### New note" if is_new else "### Edit note")

    title = st.text_input(
        "Title",
        value="" if note is None else note.title,
        key=f"notes_title_{current_id}",
        placeholder="Give your note a title",
    )
    body = st.text_area(
        "Body (Markdown)",
        value="" if note is None else note.body,
        key=f"notes_body_{current_id}",
        height=280,
        placeholder="Write in Markdown. Headings, **bold**, lists and more.",
    )
    tags = st.text_input(
        "Tags",
        value="" if note is None else note.tags,
        key=f"notes_tags_{current_id}",
        placeholder="comma, separated, tags",
    )

    save_col, delete_col = st.columns([1, 1])
    with save_col:
        if st.button("Save", key=f"notes_save_{current_id}", width="stretch", type="primary"):
            _save(user, note, title, body, tags)
    with delete_col:
        if not is_new and st.button(
            "Delete", key=f"notes_delete_{current_id}", width="stretch"
        ):
            _delete(note)

    st.divider()
    st.markdown("#### Preview")
    if (body or "").strip():
        st.markdown(body)
    else:
        st.caption("Nothing to preview yet.")

    if not is_new and note is not None:
        st.divider()
        _render_share(user, note)


def _save(
    user, note: Optional[Note], title: str, body: str, tags: str
) -> None:
    """Persist the editor contents: create a new note or update the current one."""

    if note is None:
        created = create_note(user.id, title, body, tags, source="notes")
        st.toast("Note created.")
        _select(created.id)
        return
    if update_note(note.id, title, body, tags):
        st.toast("Note saved.")
    else:
        st.warning("Could not save the note. Please try again.")
    st.rerun()


def _delete(note: Optional[Note]) -> None:
    """Delete the current note and return the editor to the empty state."""

    if note is None:
        return
    if delete_note(note.id):
        st.toast("Note deleted.")
    else:
        st.warning("Could not delete the note. Please try again.")
    _select(None)


# --------------------------------------------------------------------------- #
# Sharing
# --------------------------------------------------------------------------- #
def _render_share(user, note: Note) -> None:
    """Render the Share section: two clearly-separated sharing models.

    - **Send to a person** delivers the note directly to another user's
      dashboard (targeted, private to the recipient). This is the default,
      primary way to share.
    - **Anyone with the link** is the separate public-link model: publish the
      note and hand out an unlisted URL that any holder can open.
    """

    st.markdown("#### Share")

    _render_send_to_person(user, note)

    st.divider()
    _render_public_link(note)


def _render_send_to_person(user, note: Note) -> None:
    """Deliver the note straight to another user's dashboard (targeted share)."""

    st.markdown("**Send to a person**")
    st.caption(
        "Deliver this note straight to someone's dashboard. Only they can see it."
    )

    people = list_directory(user.id)
    if not people:
        st.caption("No other people are available to send this to yet.")
        return

    labels = {
        p["id"]: (p["name"] + (f" ({p['role']})" if p.get("role") else ""))
        for p in people
    }
    choice = st.selectbox(
        "Send to a person",
        options=[p["id"] for p in people],
        format_func=lambda pid: labels.get(pid, pid),
        key=f"notes_send_person_{note.id}",
        label_visibility="collapsed",
    )
    if st.button("Send to person", key=f"notes_send_btn_{note.id}", width="stretch", type="primary"):
        other_name = choice
        for person in people:
            if person["id"] == choice:
                other_name = person["name"]
                break
        if share_note_with(note.id, user.id, user.full_name or user.id, choice):
            st.success(
                f"Shared with {other_name}. It now appears on their dashboard."
            )
        else:
            st.warning("Could not share the note. Please try again.")


def _render_public_link(note: Note) -> None:
    """Render the separate "anyone with the link" public-share model."""

    st.markdown("**Anyone with the link**")

    if not note.is_public:
        st.caption(
            "This note is private. Publish it to create an unlisted link anyone "
            "can open."
        )
        if st.button(
            "Create public link", key=f"notes_share_{note.id}", width="stretch"
        ):
            if set_public(note.id, True):
                st.toast("Note published. Anyone with the link can now view it.")
            else:
                st.warning("Could not publish the note. Please try again.")
            st.rerun()
        return

    link = _share_link(note.id)
    st.code(link, language="text")
    st.caption("Anyone with this link can view this note.")

    if st.button("Make private", key=f"notes_private_{note.id}", width="stretch"):
        if set_public(note.id, False):
            st.toast("Note is private again. The share link no longer works.")
        else:
            st.warning("Could not make the note private. Please try again.")
        st.rerun()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    """Gate on login, then render the two-pane notebook."""

    st.set_page_config(page_title="Notes | NaviLearn", page_icon="\U0001f4dd")
    user = require_user()
    st.title("Notes")

    current_id = _selected_id()

    left, right = st.columns([1, 2], gap="medium")
    with left:
        _render_list(user, current_id)
    with right:
        _render_editor(user, current_id)


main()
