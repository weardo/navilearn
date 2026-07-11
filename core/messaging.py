"""Supabase-backed direct-message, group, and searchable messaging module.

This module powers the people-to-people messaging surface that sits alongside
the shared Live Classroom: a single always-on "main" room everyone can post in,
one-to-one direct-message rooms, ad-hoc named group rooms, and full-text search
across every room the asking user belongs to.

Design notes:

- It owns exactly one Supabase client, built from ``settings.supabase_url`` +
  ``settings.supabase_service_role_key`` (the same construction pattern as
  ``core/classroom.py``), and maps operations onto plain table calls.
- Every side-effect is best-effort: a backend failure is logged and swallowed so
  the Streamlit UI keeps rendering instead of crashing. Reads degrade to empty
  results; writes degrade to ``False`` / ``""`` no-ops.
- Ids are plain text throughout so demo ids like ``"student-demo"`` work without
  any uuid coercion. Direct-message room ids are deterministic
  (``"dm:" + "|".join(sorted([a_id, b_id]))``) so both participants converge on
  the same room without coordination.

Tables used (public schema, text ids, no RLS), created by the messaging
migration:

- ``rooms(id text pk, type, name, created_by, created_at)`` where ``type`` is one
  of ``main`` | ``dm`` | ``group``.
- ``room_members(room_id, user_id, user_name, joined_at)`` primary key
  ``(room_id, user_id)``.
- ``room_messages(id uuid, room_id, author_id, author_name, text, created_at,
  fts tsvector GENERATED)`` with a gin index over ``fts``.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

_LOG = logging.getLogger(__name__)

# The single shared room every visitor lands in. Fixed, human-readable id so all
# first-time visitors converge on it without any setup.
MAIN_ROOM_ID = "main-room"
MAIN_ROOM_NAME = "Main Room"


# --------------------------------------------------------------------------- #
# Typed rows
# --------------------------------------------------------------------------- #
@dataclass
class Room:
    """A messaging room: the shared main room, a dm, or a named group.

    For a ``dm`` room, :attr:`name` is set (by :func:`list_rooms`) to the *other*
    participant's display name so the caller can label the conversation from the
    asking user's point of view.
    """

    id: str
    type: str
    name: str
    created_by: str = ""
    created_at: str = ""
    member_names: list[str] = field(default_factory=list)


@dataclass
class DMessage:
    """A single message posted to a room.

    :attr:`room_name` is left empty for normal reads and is filled in only by
    :func:`search_messages`, which spans multiple rooms and needs to tell the
    caller which room each hit came from.
    """

    id: str
    room_id: str
    author_id: str
    author_name: str
    text: str
    created_at: str
    room_name: str = ""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""

    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    """Return a fresh opaque string id (uuid4)."""

    return str(uuid.uuid4())


@lru_cache(maxsize=1)
def _client():
    """Return a cached Supabase client built from settings, or ``None``.

    Cached because ``create_client`` opens a session we want to reuse across
    Streamlit reruns. Returns ``None`` (rather than raising) when the project is
    not configured or the client cannot be built, so every caller can degrade to
    a no-op instead of crashing the page.
    """

    from core.config import get_settings

    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        _LOG.warning("Messaging disabled: Supabase URL/service key not configured.")
        return None
    try:
        from supabase import create_client  # local import: optional dependency

        return create_client(
            settings.supabase_url, settings.supabase_service_role_key
        )
    except Exception as exc:  # noqa: BLE001 - a bad client must not crash the UI.
        _LOG.warning("Messaging Supabase client unavailable: %s", exc)
        return None


def _table(name: str):
    """Return a table handle, or ``None`` if the client is unavailable."""

    client = _client()
    return client.table(name) if client is not None else None


def _dm_id(a_id: str, b_id: str) -> str:
    """Return the deterministic dm room id for a pair of user ids."""

    return "dm:" + "|".join(sorted([a_id, b_id]))


# --------------------------------------------------------------------------- #
# Rooms
# --------------------------------------------------------------------------- #
def ensure_main_room() -> str:
    """Return the id of the shared main room, creating it once if missing.

    Idempotent: uses the fixed :data:`MAIN_ROOM_ID`, so concurrent first-time
    visitors all converge on the same room. Falls back to returning the fixed id
    even when the backend is unreachable so the page can still render.
    """

    table = _table("rooms")
    if table is None:
        return MAIN_ROOM_ID
    try:
        existing = table.select("id").eq("id", MAIN_ROOM_ID).limit(1).execute()
        if not (existing.data or []):
            table.upsert(
                {
                    "id": MAIN_ROOM_ID,
                    "type": "main",
                    "name": MAIN_ROOM_NAME,
                    "created_by": "system",
                    "created_at": _now_iso(),
                }
            ).execute()
    except Exception as exc:  # noqa: BLE001 - main room is best-effort.
        _LOG.warning("ensure_main_room failed: %s", exc)
    return MAIN_ROOM_ID


def _members_for_rooms(room_ids: list[str]) -> dict[str, list[dict[str, str]]]:
    """Return ``{room_id: [{'user_id','user_name'}, ...]}`` for the given rooms."""

    out: dict[str, list[dict[str, str]]] = {rid: [] for rid in room_ids}
    if not room_ids:
        return out
    table = _table("room_members")
    if table is None:
        return out
    try:
        res = table.select("room_id,user_id,user_name").in_("room_id", room_ids).execute()
    except Exception as exc:  # noqa: BLE001 - reads degrade to empty.
        _LOG.warning("_members_for_rooms failed: %s", exc)
        return out
    for row in res.data or []:
        rid = row.get("room_id", "")
        if rid not in out:
            out[rid] = []
        out[rid].append(
            {
                "user_id": row.get("user_id", ""),
                "user_name": row.get("user_name", "") or "",
            }
        )
    return out


def list_rooms(user_id: str) -> list[Room]:
    """Return the main room plus every room the user is a member of.

    For a ``dm`` room, :attr:`Room.name` is rewritten to the *other* member's
    display name so the conversation reads correctly from ``user_id``'s side.
    Rooms are ordered by most recent activity where known, then by name, with the
    main room pinned first. Best-effort: returns just the main room on error.
    """

    ensure_main_room()
    rooms_table = _table("rooms")
    members_table = _table("room_members")
    if rooms_table is None or members_table is None:
        return [Room(id=MAIN_ROOM_ID, type="main", name=MAIN_ROOM_NAME)]

    # Which rooms does this user belong to?
    member_room_ids: list[str] = []
    try:
        res = (
            members_table.select("room_id").eq("user_id", user_id).execute()
        )
        member_room_ids = [r.get("room_id", "") for r in (res.data or []) if r.get("room_id")]
    except Exception as exc:  # noqa: BLE001 - reads degrade to empty.
        _LOG.warning("list_rooms member lookup failed: %s", exc)

    wanted_ids = list(dict.fromkeys([MAIN_ROOM_ID, *member_room_ids]))

    room_rows: list[dict[str, Any]] = []
    try:
        res = rooms_table.select("*").in_("id", wanted_ids).execute()
        room_rows = list(res.data or [])
    except Exception as exc:  # noqa: BLE001 - reads degrade to empty.
        _LOG.warning("list_rooms room lookup failed: %s", exc)

    # Ensure the main room is present even if the fetch missed it.
    if not any(r.get("id") == MAIN_ROOM_ID for r in room_rows):
        room_rows.append(
            {"id": MAIN_ROOM_ID, "type": "main", "name": MAIN_ROOM_NAME, "created_at": ""}
        )

    members_by_room = _members_for_rooms([r.get("id", "") for r in room_rows])

    rooms: list[Room] = []
    for row in room_rows:
        rid = row.get("id", "")
        rtype = row.get("type", "group")
        members = members_by_room.get(rid, [])
        member_names = [m["user_name"] for m in members if m["user_name"]]
        name = row.get("name", "") or ""
        if rtype == "dm":
            others = [m for m in members if m["user_id"] != user_id]
            if others:
                name = others[0]["user_name"] or others[0]["user_id"]
            elif not name:
                name = "Direct message"
        rooms.append(
            Room(
                id=rid,
                type=rtype,
                name=name,
                created_by=row.get("created_by", "") or "",
                created_at=row.get("created_at", "") or "",
                member_names=member_names,
            )
        )

    def _sort_key(r: Room) -> tuple[int, str, str]:
        # Main room first; then newest created_at first; then name.
        pinned = 0 if r.type == "main" else 1
        # Reverse created_at ordering by negating via a descending string trick:
        # simpler to sort by (pinned, name) after a created_at-desc pre-sort.
        return (pinned, r.created_at, r.name)

    # Pre-sort by created_at descending, then stable-sort main to the front.
    rooms.sort(key=lambda r: (r.created_at or ""), reverse=True)
    rooms.sort(key=lambda r: 0 if r.type == "main" else 1)
    return rooms


def get_or_create_dm(a_id: str, a_name: str, b_id: str, b_name: str) -> str:
    """Return the deterministic dm room id for two users, creating it if missing.

    The room id is ``"dm:" + "|".join(sorted([a_id, b_id]))`` so both callers
    (in either argument order) resolve to the same room. Idempotent: the room row
    and both ``room_members`` rows are upserted, so repeated calls are safe.
    Returns the room id even when the backend is unreachable.
    """

    room_id = _dm_id(a_id, b_id)
    rooms_table = _table("rooms")
    members_table = _table("room_members")
    if rooms_table is None or members_table is None:
        return room_id
    try:
        rooms_table.upsert(
            {
                "id": room_id,
                "type": "dm",
                "name": "",
                "created_by": a_id,
                "created_at": _now_iso(),
            },
            on_conflict="id",
        ).execute()
    except Exception as exc:  # noqa: BLE001 - creation is best-effort.
        _LOG.warning("get_or_create_dm room upsert failed: %s", exc)
    try:
        members_table.upsert(
            [
                {
                    "room_id": room_id,
                    "user_id": a_id,
                    "user_name": a_name or a_id,
                    "joined_at": _now_iso(),
                },
                {
                    "room_id": room_id,
                    "user_id": b_id,
                    "user_name": b_name or b_id,
                    "joined_at": _now_iso(),
                },
            ],
            on_conflict="room_id,user_id",
        ).execute()
    except Exception as exc:  # noqa: BLE001 - membership is best-effort.
        _LOG.warning("get_or_create_dm member upsert failed: %s", exc)
    return room_id


def create_group(
    name: str,
    creator_id: str,
    creator_name: str,
    members: list[tuple[str, str]],
) -> str:
    """Create a named group room and return its id (best-effort).

    ``members`` is a list of ``(user_id, user_name)`` pairs to add; the creator is
    always included even if omitted. A fresh id is always returned even if the
    insert fails, so the caller can proceed. Duplicate members are de-duplicated
    by user id.
    """

    room_id = "group:" + _new_id()
    rooms_table = _table("rooms")
    members_table = _table("room_members")
    if rooms_table is None or members_table is None:
        return room_id

    # De-duplicate members by id, ensuring the creator is present.
    by_id: dict[str, str] = {}
    by_id[creator_id] = creator_name or creator_id
    for uid, uname in members or []:
        if uid:
            by_id[uid] = uname or uid

    try:
        rooms_table.insert(
            {
                "id": room_id,
                "type": "group",
                "name": (name or "Group").strip() or "Group",
                "created_by": creator_id,
                "created_at": _now_iso(),
            }
        ).execute()
    except Exception as exc:  # noqa: BLE001 - creation is best-effort.
        _LOG.warning("create_group room insert failed: %s", exc)
    try:
        members_table.upsert(
            [
                {
                    "room_id": room_id,
                    "user_id": uid,
                    "user_name": uname,
                    "joined_at": _now_iso(),
                }
                for uid, uname in by_id.items()
            ],
            on_conflict="room_id,user_id",
        ).execute()
    except Exception as exc:  # noqa: BLE001 - membership is best-effort.
        _LOG.warning("create_group member upsert failed: %s", exc)
    return room_id


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #
def _row_to_message(row: dict[str, Any], room_name: str = "") -> DMessage:
    """Map a ``room_messages`` row to a :class:`DMessage`."""

    return DMessage(
        id=str(row.get("id", "")),
        room_id=row.get("room_id", ""),
        author_id=row.get("author_id", "") or "",
        author_name=row.get("author_name", "") or "Anonymous",
        text=row.get("text", "") or "",
        created_at=row.get("created_at", "") or "",
        room_name=room_name,
    )


def list_messages(room_id: str, limit: int = 50) -> list[DMessage]:
    """Return the most recent messages for a room in chronological order.

    Fetches the newest ``limit`` rows and returns them oldest-first so the UI can
    render top-to-bottom like a normal chat transcript. Best-effort: returns an
    empty list on error.
    """

    table = _table("room_messages")
    if table is None:
        return []
    try:
        res = (
            table.select("*")
            .eq("room_id", room_id)
            .order("created_at", desc=True)
            .limit(max(1, int(limit)))
            .execute()
        )
    except Exception as exc:  # noqa: BLE001 - reads degrade to empty.
        _LOG.warning("list_messages failed: %s", exc)
        return []
    rows = list(res.data or [])
    rows.reverse()  # newest-first fetch -> oldest-first for display
    return [_row_to_message(row) for row in rows]


def post_message(room_id: str, user_id: str, user_name: str, text: str) -> bool:
    """Append a message to a room. Returns success (best-effort).

    Empty (or whitespace-only) messages are dropped and return ``False``. The
    author name is snapshotted onto the row so the transcript reads well even if
    profiles change later.
    """

    text = (text or "").strip()
    if not text:
        return False
    table = _table("room_messages")
    if table is None:
        return False
    try:
        table.insert(
            {
                "id": _new_id(),
                "room_id": room_id,
                "author_id": user_id or "",
                "author_name": (user_name or user_id or "Anonymous"),
                "text": text,
                "created_at": _now_iso(),
            }
        ).execute()
        return True
    except Exception as exc:  # noqa: BLE001 - posting is best-effort.
        _LOG.warning("post_message failed: %s", exc)
        return False


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def _room_names(room_ids: list[str]) -> dict[str, str]:
    """Return ``{room_id: display_name}`` for the given rooms (best-effort)."""

    out: dict[str, str] = {}
    if not room_ids:
        return out
    table = _table("rooms")
    if table is None:
        return out
    try:
        res = table.select("id,type,name").in_("id", room_ids).execute()
    except Exception as exc:  # noqa: BLE001 - reads degrade to empty.
        _LOG.warning("_room_names failed: %s", exc)
        return out
    for row in res.data or []:
        rid = row.get("id", "")
        name = row.get("name", "") or ""
        if not name:
            name = MAIN_ROOM_NAME if row.get("type") == "main" else rid
        out[rid] = name
    return out


def search_messages(user_id: str, query: str, limit: int = 50) -> list[DMessage]:
    """Search messages across only the rooms ``user_id`` belongs to.

    Uses Postgres full-text search on the generated ``fts`` column via
    ``.text_search('fts', query, options={'type': 'websearch'})``. If that errors
    (e.g. an unparsable query), it falls back to a case-insensitive
    ``ilike('text', '%query%')`` scan. Each hit's :attr:`DMessage.room_name` is
    populated so the caller can show where the match came from. Best-effort:
    returns an empty list on error or when the query is blank.
    """

    query = (query or "").strip()
    if not query:
        return []
    table = _table("room_messages")
    if table is None:
        return []

    # Restrict to the rooms this user can see (main room + memberships).
    ensure_main_room()
    members_table = _table("room_members")
    room_ids: list[str] = [MAIN_ROOM_ID]
    if members_table is not None:
        try:
            res = members_table.select("room_id").eq("user_id", user_id).execute()
            room_ids.extend(
                r.get("room_id", "") for r in (res.data or []) if r.get("room_id")
            )
        except Exception as exc:  # noqa: BLE001 - reads degrade.
            _LOG.warning("search_messages membership lookup failed: %s", exc)
    room_ids = list(dict.fromkeys(rid for rid in room_ids if rid))
    if not room_ids:
        return []

    lim = max(1, int(limit))
    rows: list[dict[str, Any]] = []
    try:
        # ``text_search`` narrows the builder to one that no longer exposes
        # ``order``/``limit``, so those must be applied before it in the chain.
        res = (
            table.select("*")
            .in_("room_id", room_ids)
            .order("created_at", desc=True)
            .limit(lim)
            .text_search("fts", query, options={"type": "websearch"})
            .execute()
        )
        rows = list(res.data or [])
    except Exception as exc:  # noqa: BLE001 - fall back to ILIKE.
        _LOG.warning("search_messages FTS failed, falling back to ilike: %s", exc)
        try:
            res = (
                table.select("*")
                .in_("room_id", room_ids)
                .ilike("text", f"%{query}%")
                .order("created_at", desc=True)
                .limit(lim)
                .execute()
            )
            rows = list(res.data or [])
        except Exception as exc2:  # noqa: BLE001 - reads degrade to empty.
            _LOG.warning("search_messages ilike fallback failed: %s", exc2)
            return []

    names = _room_names([r.get("room_id", "") for r in rows])
    return [_row_to_message(row, names.get(row.get("room_id", ""), "")) for row in rows]


# --------------------------------------------------------------------------- #
# Directory
# --------------------------------------------------------------------------- #
def list_directory(exclude_user_id: str) -> list[dict[str, str]]:
    """Return other platform users as ``[{'id','name','role'}, ...]``.

    Pulls profiles from the repository and drops ``exclude_user_id`` (typically
    the asking user) so a "start a chat with" picker never lists the user itself.
    Best-effort: returns an empty list on any error.
    """

    try:
        from core.config import get_settings
        from core.repo import get_repo

        profiles = get_repo(get_settings()).list_profiles()
    except Exception as exc:  # noqa: BLE001 - directory is best-effort.
        _LOG.warning("list_directory failed: %s", exc)
        return []

    out: list[dict[str, str]] = []
    for prof in profiles or []:
        pid = getattr(prof, "id", "") or ""
        if not pid or pid == exclude_user_id:
            continue
        name = getattr(prof, "full_name", "") or getattr(prof, "email", "") or pid
        out.append({"id": pid, "name": name, "role": getattr(prof, "role", "") or ""})
    return out
