"""Supabase-backed write layer for the two-way Mentor Dashboard (Challenge 4).

The read-only Mentor page could only look. This module gives a mentor real
verbs: claim an unassigned student, scope the roster to just their own mentees,
and leave written feedback that persists per student. It is intentionally thin
and typed, mirroring :mod:`core.classroom`: it owns exactly one Supabase client
(built from ``settings.supabase_url`` + ``settings.supabase_service_role_key``)
and maps each operation onto plain table calls.

Every side-effect is best-effort. A backend failure is logged and swallowed so
the Streamlit page keeps rendering instead of crashing: reads degrade to empty
results, writes degrade to ``False`` no-ops.

Two id conventions live side by side, and this module bridges them:

- ``profiles.id`` and ``profiles.mentor_id`` are Postgres ``uuid`` columns. The
  seed layer stores readable ids such as ``"student-demo"`` as a deterministic
  ``uuid5`` (see :func:`_uid`), so every profiles query coerces its id/foreign
  key the same way. Returned :class:`~core.repo.Profile` ids are mapped back to
  their readable form when known (see :data:`_READABLE_BY_UID`) so callers that
  passed ``"student-demo"`` see ``"student-demo"`` come back.
- ``mentor_notes.student_id`` and ``mentor_notes.mentor_id`` are plain ``text``
  columns. The ``student_id`` is canonicalized (see :func:`_canonical_id`) on
  both write and read so a note a mentor keys by a readable id such as
  ``"student-demo"`` is still found when the student reads with the uuid their
  runtime :class:`Profile` carries (what :meth:`core.repo.SupabaseRepo.get_profile`
  returns). Without this, mentor feedback would silently never reach the student.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

from core.config import get_settings
from core.repo import Profile

_LOG = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Client and id helpers
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _client():
    """Return a cached Supabase client built from settings, or ``None``.

    Cached because ``create_client`` opens a session worth reusing across
    Streamlit reruns. Returns ``None`` (rather than raising) when the project is
    not configured or the client cannot be built, so every caller can degrade to
    a no-op instead of crashing the page.
    """

    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        _LOG.warning("Mentoring disabled: Supabase URL/service key not configured.")
        return None
    try:
        from supabase import create_client  # local import: optional dependency

        return create_client(
            settings.supabase_url, settings.supabase_service_role_key
        )
    except Exception as exc:  # noqa: BLE001 - a bad client must not crash the UI.
        _LOG.warning("Mentoring Supabase client unavailable: %s", exc)
        return None


def _table(name: str):
    """Return a table handle, or ``None`` if the client is unavailable."""

    client = _client()
    return client.table(name) if client is not None else None


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""

    return datetime.now(timezone.utc).isoformat()


def _uid(value: Optional[str]) -> Optional[str]:
    """Coerce an id/foreign-key to a valid UUID string, matching the repo.

    ``profiles`` id columns are ``uuid``, but callers use readable ids such as
    ``"student-demo"``. Values that already parse as UUIDs pass through
    unchanged; anything else maps to a deterministic ``uuid5`` so the same
    string always yields the same UUID and foreign keys stay consistent with
    :class:`core.repo.SupabaseRepo`.
    """

    if value is None:
        return None
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"navilearn:{value}"))


# Reverse map from coerced uuid back to the readable id it came from. Seeded
# with the well-known demo ids and extended whenever a readable id flows in as a
# function argument, so a caller that passes ``"student-demo"`` gets it back on
# the returned :class:`Profile` even though Postgres stores the uuid form.
_READABLE_BY_UID: dict[str, str] = {}


def _remember_readable(value: Optional[str]) -> None:
    """Record ``value`` so its uuid form can be mapped back to it later."""

    if not value:
        return
    coerced = _uid(value)
    if coerced is not None and coerced != str(value):
        _READABLE_BY_UID[coerced] = str(value)


for _seed_id in ("student-demo", "mentor-demo", "teacher-demo"):
    _remember_readable(_seed_id)


def _display_id(stored: Optional[str]) -> str:
    """Map a stored uuid back to its readable id when known, else pass through."""

    key = str(stored or "")
    return _READABLE_BY_UID.get(key, key)


def _canonical_id(value: Optional[str]) -> Optional[str]:
    """Map any id (readable or uuid) to the canonical repo id, or ``None``.

    This is the join key that closes the mentor-feedback loop. A mentor keys a
    note by whatever id it holds for a student (often a readable id such as
    ``"student-demo"`` restored by :func:`_display_id`), while the student reads
    with the id their runtime :class:`Profile` carries, which
    :meth:`core.repo.SupabaseRepo.get_profile` returns as the stored uuid. Both
    are collapsed onto the same deterministic ``uuid5`` (see :func:`_uid`, which
    mirrors :meth:`core.repo.SupabaseRepo._uid`), the value Postgres actually
    stores for a profile, so a note saved by either side lines up on read.
    Values that are already a canonical uuid pass through unchanged.
    """

    return _uid(value)


def _row_to_profile(row: dict) -> Profile:
    """Build a :class:`Profile` from a profiles row, restoring readable ids."""

    return Profile(
        id=_display_id(row.get("id")),
        email=row.get("email") or "",
        full_name=row.get("full_name") or "",
        role=row.get("role", "student"),
        mentor_id=_display_id(row.get("mentor_id")) if row.get("mentor_id") else None,
    )


# --------------------------------------------------------------------------- #
# Roster
# --------------------------------------------------------------------------- #
def list_students_for_mentor(mentor_id: str) -> list[Profile]:
    """Return the students assigned to ``mentor_id`` (best-effort, name order).

    Scopes to ``role = 'student'`` and ``mentor_id`` equal to this mentor, so a
    mentor only ever sees their own mentees. Degrades to an empty list when the
    backend is down.
    """

    _remember_readable(mentor_id)
    table = _table("profiles")
    if table is None:
        return []
    try:
        res = (
            table.select("*")
            .eq("role", "student")
            .eq("mentor_id", _uid(mentor_id))
            .order("full_name")
            .execute()
        )
    except Exception as exc:  # noqa: BLE001 - reads degrade to empty.
        _LOG.warning("list_students_for_mentor failed: %s", exc)
        return []
    return [_row_to_profile(row) for row in (res.data or [])]


def list_unassigned_students() -> list[Profile]:
    """Return students with no mentor yet (best-effort, name order).

    These are the candidates a mentor can claim via :func:`assign_mentor`.
    Degrades to an empty list when the backend is down.
    """

    table = _table("profiles")
    if table is None:
        return []
    try:
        res = (
            table.select("*")
            .eq("role", "student")
            .is_("mentor_id", "null")
            .order("full_name")
            .execute()
        )
    except Exception as exc:  # noqa: BLE001 - reads degrade to empty.
        _LOG.warning("list_unassigned_students failed: %s", exc)
        return []
    return [_row_to_profile(row) for row in (res.data or [])]


def assign_mentor(student_id: str, mentor_id: str) -> bool:
    """Assign ``student_id`` to ``mentor_id`` (best-effort). Returns success.

    Updates ``profiles.mentor_id`` for the student row. Both ids are coerced to
    their uuid form so the update lands on the same row the rest of the app
    sees. A backend failure logs and returns ``False`` rather than crashing.
    """

    student_id = (student_id or "").strip()
    mentor_id = (mentor_id or "").strip()
    if not student_id or not mentor_id:
        return False
    _remember_readable(student_id)
    _remember_readable(mentor_id)
    table = _table("profiles")
    if table is None:
        return False
    try:
        table.update({"mentor_id": _uid(mentor_id)}).eq(
            "id", _uid(student_id)
        ).execute()
        return True
    except Exception as exc:  # noqa: BLE001 - writes degrade to no-op.
        _LOG.warning("assign_mentor failed: %s", exc)
        return False


# --------------------------------------------------------------------------- #
# Mentor notes (written feedback per student)
# --------------------------------------------------------------------------- #
def save_note(
    student_id: str,
    author_id: str,
    author_name: str,
    text: str,
    author_role: str = "mentor",
) -> bool:
    """Insert one message into a student's feedback thread (best-effort).

    The thread is two-way: a mentor (or teacher) and the student both post into
    the same per-student conversation, distinguished by ``author_role``. Empty
    ``text`` is rejected and returns ``False`` without touching the backend.

    The ``author_id``, ``author_name`` and ``author_role`` describe whoever
    wrote this message and are snapshotted onto the row so the thread reads well
    even if the profile changes later. For backward compatibility the legacy
    ``mentor_id`` / ``mentor_name`` columns are still filled when the author is a
    mentor or teacher (staff-authored rows), so older readers keep working.

    The ``student_id`` names the thread and is stored in its canonical form (see
    :func:`_canonical_id`) so both sides, who may key it by a readable id or the
    uuid their runtime :class:`Profile` carries, land on the same conversation.
    """

    text = (text or "").strip()
    if not text:
        return False
    student_id = (student_id or "").strip()
    author_id = (author_id or "").strip()
    author_role = (author_role or "mentor").strip() or "mentor"
    if not student_id or not author_id:
        return False
    _remember_readable(student_id)
    table = _table("mentor_notes")
    if table is None:
        return False
    display_name = (author_name or "").strip() or (
        "Mentor" if author_role in ("mentor", "teacher") else "Student"
    )
    row: dict = {
        "id": str(uuid.uuid4()),
        "student_id": _canonical_id(student_id),
        "author_id": author_id,
        "author_name": display_name,
        "author_role": author_role,
        "text": text,
        "created_at": _now_iso(),
    }
    # Backward compatibility: keep the legacy staff columns populated for
    # mentor/teacher-authored rows so older readers still see feedback.
    if author_role in ("mentor", "teacher"):
        row["mentor_id"] = author_id
        row["mentor_name"] = display_name
    try:
        table.insert(row).execute()
        return True
    except Exception as exc:  # noqa: BLE001 - writes degrade to no-op.
        _LOG.warning("save_note failed: %s", exc)
        return False


def list_notes(student_id: str) -> list[dict]:
    """Return a student's feedback thread, oldest first (best-effort).

    The thread is a two-way conversation between a mentor (or teacher) and the
    student. Each item is a dict with ``id``, ``author_name``, ``author_role``,
    ``text`` and ``created_at`` keys, ordered oldest-first so callers can render
    it top-to-bottom like a chat. Legacy rows written before the two-way columns
    existed are mentor-authored, so their ``author_name`` / ``author_role`` fall
    back to the old ``mentor_name`` column and the ``"mentor"`` role.

    The ``student_id`` is canonicalized (see :func:`_canonical_id`) to the same
    form :func:`save_note` writes, so a message either side left keyed by a
    readable id is found here by the other side's runtime uuid. Degrades to an
    empty list when the backend is down.
    """

    student_id = (student_id or "").strip()
    if not student_id:
        return []
    _remember_readable(student_id)
    table = _table("mentor_notes")
    if table is None:
        return []
    try:
        res = (
            table.select(
                "id, author_name, author_role, mentor_name, text, created_at"
            )
            .eq("student_id", _canonical_id(student_id))
            .order("created_at", desc=False)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001 - reads degrade to empty.
        _LOG.warning("list_notes failed: %s", exc)
        return []
    notes: list[dict] = []
    for row in res.data or []:
        role = (row.get("author_role") or "mentor").strip() or "mentor"
        name = (row.get("author_name") or "").strip() or (
            row.get("mentor_name") or ""
        ).strip()
        if not name:
            name = "Mentor" if role in ("mentor", "teacher") else "Student"
        notes.append(
            {
                "id": row.get("id", ""),
                "author_name": name,
                "author_role": role,
                "text": row.get("text", "") or "",
                "created_at": row.get("created_at", "") or "",
            }
        )
    return notes
