"""Supabase-backed data layer for personal study notes.

A note is a small, owner-scoped Markdown document: a title, a Markdown body,
optional comma-separated tags, and an optional free-text source label. Each note
belongs to exactly one user (``owner_id``) and can optionally be flipped public
(``is_public``) so it can be read by anyone holding its id (a lightweight
"share link" without any new tables or auth surface).

This module mirrors the thin, typed, best-effort style of :mod:`core.classroom`.
It owns exactly one Supabase client (built from ``settings.supabase_url`` +
``settings.supabase_service_role_key``) and maps note operations onto plain
table calls against the pre-existing ``notes`` table:

- ``notes(id uuid, owner_id text, title text, body text, tags text,
  source text, is_public bool, created_at timestamptz, updated_at timestamptz)``

Every side-effect is best-effort: a backend failure is logged and swallowed so
the Streamlit UI keeps rendering instead of crashing. Reads degrade to empty
results (``[]`` / ``None``); writes degrade to ``False`` (or, for
:func:`create_note`, a local-only :class:`Note` whose row may not exist
server-side).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Optional

from core.config import get_settings

_LOG = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Typed row
# --------------------------------------------------------------------------- #
@dataclass
class Note:
    """A single owner-scoped Markdown note.

    ``tags`` is stored as a raw comma-separated string (the UI splits it); an
    empty string means untagged. ``source`` is a free-text provenance label
    (for example ``"manual"`` or a document name). ``is_public`` gates whether
    :func:`get_public_note` will return the row to an unauthenticated reader.
    """

    id: str
    owner_id: str
    title: str
    body: str
    tags: str = ""
    source: str = ""
    is_public: bool = False
    created_at: str = ""
    updated_at: str = ""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _new_id() -> str:
    """Return a fresh opaque uuid string (accepted by the ``uuid`` column)."""

    return str(uuid.uuid4())


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""

    return datetime.now(timezone.utc).isoformat()


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
        _LOG.warning("Notes disabled: Supabase URL/service key not configured.")
        return None
    try:
        from supabase import create_client  # local import: optional dependency

        return create_client(
            settings.supabase_url, settings.supabase_service_role_key
        )
    except Exception as exc:  # noqa: BLE001 - a bad client must not crash the UI.
        _LOG.warning("Notes Supabase client unavailable: %s", exc)
        return None


def _table():
    """Return the ``notes`` table handle, or ``None`` if the client is down."""

    client = _client()
    return client.table("notes") if client is not None else None


def _note_from_row(row: dict[str, Any]) -> Note:
    """Normalise a ``notes`` DB row into a typed :class:`Note`."""

    return Note(
        id=str(row.get("id") or ""),
        owner_id=str(row.get("owner_id") or ""),
        title=row.get("title") or "",
        body=row.get("body") or "",
        tags=row.get("tags") or "",
        source=row.get("source") or "",
        is_public=bool(row.get("is_public")),
        created_at=row.get("created_at") or "",
        updated_at=row.get("updated_at") or "",
    )


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #
def list_notes(owner_id: str) -> list[Note]:
    """Return a user's notes, most recently updated first (best-effort).

    Reads degrade to an empty list when the backend is unavailable or the query
    fails.
    """

    table = _table()
    if table is None:
        return []
    try:
        res = (
            table.select("*")
            .eq("owner_id", owner_id)
            .order("updated_at", desc=True)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001 - reads degrade to empty.
        _LOG.warning("list_notes failed: %s", exc)
        return []
    return [_note_from_row(row) for row in (res.data or [])]


def get_note(note_id: str) -> Optional[Note]:
    """Return a single note by id, or ``None`` if missing or the backend is down."""

    table = _table()
    if table is None:
        return None
    try:
        res = table.select("*").eq("id", note_id).limit(1).execute()
        rows = res.data or []
        return _note_from_row(rows[0]) if rows else None
    except Exception as exc:  # noqa: BLE001 - reads degrade to None.
        _LOG.warning("get_note failed: %s", exc)
        return None


def get_public_note(note_id: str) -> Optional[Note]:
    """Return a note by id ONLY when it is public, else ``None``.

    This is the read path for share links: it filters on ``is_public = true`` so
    a private note is never disclosed even to a caller who knows its id. Returns
    ``None`` when the note is missing, private, or the backend is unavailable.
    """

    table = _table()
    if table is None:
        return None
    try:
        res = (
            table.select("*")
            .eq("id", note_id)
            .eq("is_public", True)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        return _note_from_row(rows[0]) if rows else None
    except Exception as exc:  # noqa: BLE001 - reads degrade to None.
        _LOG.warning("get_public_note failed: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #
def create_note(
    owner_id: str,
    title: str,
    body: str,
    tags: str = "",
    source: str = "",
) -> Note:
    """Create a note for ``owner_id`` and return it (best-effort).

    A fresh uuid and matching created/updated timestamps are generated locally
    so the returned :class:`Note` is always usable by the caller, even if the
    insert fails (in that degraded case the row simply may not exist
    server-side). New notes start private (``is_public = False``).
    """

    now = _now_iso()
    note = Note(
        id=_new_id(),
        owner_id=owner_id,
        title=title or "",
        body=body or "",
        tags=tags or "",
        source=source or "",
        is_public=False,
        created_at=now,
        updated_at=now,
    )
    table = _table()
    if table is None:
        return note
    try:
        res = table.insert(
            {
                "id": note.id,
                "owner_id": note.owner_id,
                "title": note.title,
                "body": note.body,
                "tags": note.tags,
                "source": note.source,
                "is_public": note.is_public,
                "created_at": note.created_at,
                "updated_at": note.updated_at,
            }
        ).execute()
        rows = res.data or []
        if rows:
            return _note_from_row(rows[0])
    except Exception as exc:  # noqa: BLE001 - creation is best-effort.
        _LOG.warning("create_note failed: %s", exc)
    return note


def update_note(note_id: str, title: str, body: str, tags: str) -> bool:
    """Update a note's title, body, and tags, bumping ``updated_at``.

    Returns ``True`` on a successful write, ``False`` when the backend is
    unavailable or the update fails (best-effort: logs, never raises).
    """

    table = _table()
    if table is None:
        return False
    try:
        table.update(
            {
                "title": title or "",
                "body": body or "",
                "tags": tags or "",
                "updated_at": _now_iso(),
            }
        ).eq("id", note_id).execute()
        return True
    except Exception as exc:  # noqa: BLE001 - writes degrade to no-op.
        _LOG.warning("update_note failed: %s", exc)
        return False


def delete_note(note_id: str) -> bool:
    """Delete a note by id. Returns success (best-effort).

    Returns ``False`` when the backend is unavailable or the delete fails.
    """

    table = _table()
    if table is None:
        return False
    try:
        table.delete().eq("id", note_id).execute()
        return True
    except Exception as exc:  # noqa: BLE001 - writes degrade to no-op.
        _LOG.warning("delete_note failed: %s", exc)
        return False


def set_public(note_id: str, public: bool) -> bool:
    """Flip a note's ``is_public`` flag, bumping ``updated_at``. Returns success.

    Used to publish (``public=True``) or unpublish (``public=False``) a note for
    share-by-link. Best-effort: returns ``False`` when the backend is down or the
    write fails.
    """

    table = _table()
    if table is None:
        return False
    try:
        table.update(
            {
                "is_public": bool(public),
                "updated_at": _now_iso(),
            }
        ).eq("id", note_id).execute()
        return True
    except Exception as exc:  # noqa: BLE001 - writes degrade to no-op.
        _LOG.warning("set_public failed: %s", exc)
        return False
