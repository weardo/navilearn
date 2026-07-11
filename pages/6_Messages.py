"""Messages (Ch6): a Telegram-style chat surface, Supabase-backed.

A two-pane messaging app that sits alongside the Live Classroom:

- Left pane (conversations): a search box, then every room the user belongs to
  rendered as a chat row: an emoji avatar (main / dm / group), the name, and a
  one-line preview of the last message with a short time, newest conversation
  first. Clicking a row opens it; the open row is highlighted. Below the list are
  compact "New direct message" and "New group" creators drawn from the platform
  directory.
- Right pane (thread): a header with the open room's name, a scrollable transcript
  rendered as WhatsApp-style bubbles (mine right-aligned and green, others left
  and grey, with a sender name in group rooms and a small timestamp), and a
  ``st.chat_input`` composer pinned to the bottom of the page.

Live updates are handled by Streamlit fragments: the thread transcript lives in
an ``@st.fragment`` whose ``run_every`` is set to a few seconds while the "Live"
toggle (default on) is enabled, so ONLY the open room's bubbles reload on the
timer while the conversation list, composer, and creators stay interactive and do
not flicker. Turning Live off makes the fragment static. There is no
``time.sleep`` and no websocket; the free-tier pattern is shared Postgres tables
plus a light per-fragment poll.

The open room id lives in ``st.session_state['msg_selected_room_id']`` and
defaults to the shared Main Room. All backend work is delegated to
:mod:`core.messaging`, whose calls are best-effort: a hiccup logs and continues
instead of crashing the page. Message text is HTML-escaped before it reaches the
bubble markup so a message can never inject markup into the page.
"""

from __future__ import annotations

import html
import logging

import streamlit as st

from core.messaging import (
    Room,
    create_group,
    ensure_main_room,
    get_or_create_dm,
    list_directory,
    list_messages,
    list_rooms,
    post_message,
    search_messages,
)
from core.session import require_user

_LOG = logging.getLogger(__name__)

# Session-state key holding the id of the room currently open in the thread pane.
_SELECTED_KEY = "msg_selected_room_id"

# How many messages to load into the thread pane at once.
_THREAD_LIMIT = 60

# How often (seconds) the thread fragment reruns to pull new messages while Live
# is on. The conversation-list fragment refreshes on a slower cadence.
_LIVE_INTERVAL_S = 3.0
_CONV_INTERVAL_S = 6.0


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
_CHAT_CSS = """
<style>
.msg-thread {
  max-height: 62vh;
  overflow-y: auto;
  padding: 10px 6px 4px 6px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  border-radius: 10px;
  background: rgba(0, 0, 0, 0.02);
}
.msg-row { display: flex; width: 100%; }
.msg-row.me { justify-content: flex-end; }
.msg-row.them { justify-content: flex-start; }
.msg-bubble {
  max-width: 78%;
  padding: 7px 11px;
  border-radius: 14px;
  font-size: 0.92rem;
  line-height: 1.35;
  overflow-wrap: anywhere;
  box-shadow: 0 1px 1px rgba(0, 0, 0, 0.08);
}
.msg-bubble.me { background: #dcf8c6; color: #111; border-bottom-right-radius: 4px; }
.msg-bubble.them { background: #f1f0f0; color: #111; border-bottom-left-radius: 4px; }
.msg-sender { font-size: 0.72rem; font-weight: 600; color: #0a8f5b; margin-bottom: 2px; }
.msg-time { display: block; font-size: 0.66rem; color: #5b6b78; margin-top: 3px; text-align: right; }
.msg-empty { color: #7a8894; font-style: italic; padding: 12px 6px; }
@media (prefers-color-scheme: dark) {
  .msg-thread { background: rgba(255, 255, 255, 0.03); }
  .msg-bubble.me { background: #075e54; color: #e9edef; }
  .msg-bubble.them { background: #202c33; color: #e9edef; }
  .msg-time { color: #9fb0ba; }
  .msg-sender { color: #53bdeb; }
}
</style>
"""


def _inject_css() -> None:
    """Inject the chat bubble stylesheet (idempotent per rerun)."""

    st.markdown(_CHAT_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------------- #
def _avatar(room_type: str) -> str:
    """Return the emoji avatar for a room type."""

    if room_type == "main":
        return "📢"
    if room_type == "dm":
        return "💬"
    return "👥"


def _short_time(iso: str) -> str:
    """Return a short ``HH:MM`` clock from an ISO-8601 timestamp (best-effort)."""

    if not iso:
        return ""
    if "T" in iso and len(iso) >= 16:
        return iso[11:16]
    return iso[:16]


def _preview_text(user_id: str, msg) -> str:
    """Return a one-line preview of a room's last message."""

    if msg is None:
        return "No messages yet"
    who = "You: " if msg.author_id == user_id else ""
    body = " ".join((msg.text or "").split())
    if len(body) > 42:
        body = body[:41] + "…"
    return f"{who}{body}" if body else f"{who}(no text)"


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def _selected_room_id() -> str:
    """Return the open room id, defaulting to the shared main room."""

    room_id = st.session_state.get(_SELECTED_KEY)
    if not room_id:
        room_id = ensure_main_room()
        st.session_state[_SELECTED_KEY] = room_id
    return room_id


def _select_room(room_id: str) -> None:
    """Open ``room_id`` in the thread pane and rerun to refresh both panes."""

    st.session_state[_SELECTED_KEY] = room_id
    st.rerun()


# --------------------------------------------------------------------------- #
# Left pane: conversations
# --------------------------------------------------------------------------- #
def _rooms_with_previews(user_id: str) -> list[tuple[Room, object]]:
    """Return ``[(room, last_message_or_None), ...]`` sorted newest-activity first.

    Fetches the single most recent message per room so each conversation row can
    show a preview and be ordered by real activity rather than creation time.
    """

    rooms = list_rooms(user_id)
    paired: list[tuple[Room, object]] = []
    for room in rooms:
        try:
            recent = list_messages(room.id, limit=1)
        except Exception as exc:  # noqa: BLE001 - preview is best-effort.
            _LOG.warning("preview fetch failed for %s: %s", room.id, exc)
            recent = []
        paired.append((room, recent[-1] if recent else None))

    def _key(item: tuple[Room, object]) -> str:
        room, msg = item
        stamp = getattr(msg, "created_at", "") if msg is not None else ""
        return stamp or room.created_at or ""

    paired.sort(key=_key, reverse=True)
    return paired


def _render_conversation_row(
    user_id: str, room: Room, last_msg: object, is_current: bool
) -> None:
    """Render one clickable conversation row: avatar, name, preview, time."""

    with st.container(border=True):
        label = f"{_avatar(room.type)}  {room.name or 'Conversation'}"
        if st.button(
            label,
            key=f"msg_room_btn_{room.id}",
            width="stretch",
            type="primary" if is_current else "secondary",
            disabled=is_current,
        ):
            _select_room(room.id)
        stamp = getattr(last_msg, "created_at", "") if last_msg is not None else ""
        time_txt = _short_time(stamp)
        preview = _preview_text(user_id, last_msg)
        st.caption(f"{preview}  ·  {time_txt}" if time_txt else preview)


def _render_conversations(user, current_room_id: str) -> None:
    """Search box plus the selectable conversation list, newest activity first."""

    st.markdown("**Conversations**")
    query = st.text_input(
        "Search",
        placeholder="Search chats or messages",
        key="msg_search_input",
        label_visibility="collapsed",
    ).strip()

    paired = _rooms_with_previews(user.id)

    if query:
        lowered = query.lower()
        name_hits = [(r, m) for r, m in paired if lowered in (r.name or "").lower()]
        if name_hits:
            for room, last_msg in name_hits:
                _render_conversation_row(
                    user.id, room, last_msg, room.id == current_room_id
                )
        else:
            st.caption("No chat names match. Searching messages...")
        _render_message_hits(user, query)
        return

    if not paired:
        st.caption("No conversations yet.")
        return
    for room, last_msg in paired:
        _render_conversation_row(user.id, room, last_msg, room.id == current_room_id)


def _render_message_hits(user, query: str) -> None:
    """Show message-text search hits as clickable rows that open their room."""

    try:
        results = search_messages(user.id, query, limit=_THREAD_LIMIT)
    except Exception as exc:  # noqa: BLE001 - search is best-effort.
        _LOG.warning("search_messages failed: %s", exc)
        results = []
    if not results:
        st.caption("No messages match that search.")
        return
    st.caption(f"{len(results)} message match(es)")
    for hit in results:
        where = hit.room_name or "Room"
        who = hit.author_name or "Anonymous"
        snippet = hit.text if len(hit.text) <= 90 else hit.text[:89] + "…"
        if st.button(
            f"{where} · {who}: {snippet}",
            key=f"msg_search_hit_{hit.id}",
            width="stretch",
        ):
            _select_room(hit.room_id)


def _render_new_dm(user) -> None:
    """Compact creator: start a one-to-one direct message with a directory person."""

    with st.expander("New direct message"):
        people = list_directory(user.id)
        if not people:
            st.caption("No other people are available to message yet.")
            return
        labels = {
            p["id"]: (p["name"] + (f" ({p['role']})" if p.get("role") else ""))
            for p in people
        }
        choice = st.selectbox(
            "Pick a person",
            options=[p["id"] for p in people],
            format_func=lambda pid: labels.get(pid, pid),
            key="msg_new_dm_person",
        )
        if st.button("Start conversation", key="msg_new_dm_btn", width="stretch"):
            other_name = choice
            for p in people:
                if p["id"] == choice:
                    other_name = p["name"]
                    break
            room_id = get_or_create_dm(
                user.id, user.full_name or user.id, choice, other_name
            )
            _select_room(room_id)


def _render_new_group(user) -> None:
    """Compact creator: build a named group from a multiselect of directory people."""

    with st.expander("New group"):
        people = list_directory(user.id)
        if not people:
            st.caption("No other people are available to add to a group yet.")
            return
        labels = {p["id"]: p["name"] for p in people}
        name = st.text_input(
            "Group name",
            placeholder="Study crew",
            key="msg_new_group_name",
        )
        picked = st.multiselect(
            "Add people",
            options=[p["id"] for p in people],
            format_func=lambda pid: labels.get(pid, pid),
            key="msg_new_group_members",
        )
        if st.button("Create group", key="msg_new_group_btn", width="stretch"):
            if not (name or "").strip():
                st.warning("Give the group a name.")
                return
            if not picked:
                st.warning("Add at least one other person to the group.")
                return
            members = [(pid, labels.get(pid, pid)) for pid in picked]
            room_id = create_group(name, user.id, user.full_name or user.id, members)
            _select_room(room_id)


# --------------------------------------------------------------------------- #
# Right pane: thread
# --------------------------------------------------------------------------- #
def _bubble_html(user_id: str, room: Room, messages) -> str:
    """Return the full scrollable bubble transcript as safe HTML.

    Message text is HTML-escaped before interpolation so nothing in a message can
    inject markup into the page. Sender names are shown for other people's
    messages in the main room and group rooms (redundant in a one-to-one dm).
    """

    parts: list[str] = ['<div class="msg-thread">']
    if not messages:
        parts.append('<div class="msg-empty">No messages yet. Say hello.</div>')
    for msg in messages:
        is_me = msg.author_id == user_id
        side = "me" if is_me else "them"
        text = html.escape(msg.text or "").replace("\n", "<br>")
        sender = ""
        if not is_me and room.type in ("main", "group"):
            sender = (
                '<div class="msg-sender">'
                f'{html.escape(msg.author_name or "Anonymous")}</div>'
            )
        stamp = html.escape(_short_time(msg.created_at))
        parts.append(
            f'<div class="msg-row {side}"><div class="msg-bubble {side}">'
            f'{sender}{text}<span class="msg-time">{stamp}</span></div></div>'
        )
    parts.append("</div>")
    return "".join(parts)


def _render_thread(user, room: Room) -> None:
    """Thread pane: room header and the scrollable bubble transcript."""

    st.markdown(f"### {_avatar(room.type)} {room.name or 'Conversation'}")
    if room.type == "group" and room.member_names:
        st.caption(", ".join(room.member_names))
    try:
        messages = list_messages(room.id, limit=_THREAD_LIMIT)
    except Exception as exc:  # noqa: BLE001 - reads degrade to empty.
        _LOG.warning("list_messages failed for %s: %s", room.id, exc)
        messages = []
    st.markdown(_bubble_html(user.id, room, messages), unsafe_allow_html=True)


def _resolve_room(user, room_id: str) -> Room:
    """Return the Room for ``room_id``, falling back to a bare main-room stub."""

    rooms = list_rooms(user.id)
    room = next((r for r in rooms if r.id == room_id), None)
    if room is None:
        room = Room(id=room_id, type="main", name="Main Room")
    return room


def _thread_fragment(user) -> None:
    """Auto-refreshing thread pane.

    This is the ONLY part of the page that reruns on the Live timer. It reads the
    open room id from ``st.session_state`` on every (partial) rerun so it always
    reflects the current conversation and pulls in new messages, while the
    conversation list, composer, and creators are left untouched (no flicker, no
    lost drafts). When Live is off it renders once as a static fragment.
    """

    room_id = _selected_room_id()
    _render_thread(user, _resolve_room(user, room_id))


def _conversations_fragment(user) -> None:
    """Conversation list on its own slower refresh so previews stay fresh.

    Runs independently of the thread fragment; a room button here calls
    ``st.rerun`` (app scope) so opening a chat still refreshes both panes.
    """

    _render_conversations(user, _selected_room_id())


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    """Gate on login, then render the live chat inbox and open thread."""

    st.set_page_config(page_title="Messages | NaviLearn", page_icon="💬")
    user = require_user()
    _inject_css()

    title_col, live_col = st.columns([3, 1])
    with title_col:
        st.title("Messages")
    with live_col:
        live = st.toggle("Live", value=True, key="msg_live")

    # Live updates are per-fragment, not whole-page. The Live toggle simply picks
    # the fragments' run_every: a cadence in seconds while on, None (static) while
    # off. Only the fragments rerun on the timer; the composer and creators do not.
    thread_every = _LIVE_INTERVAL_S if live else None
    conv_every = _CONV_INTERVAL_S if live else None

    room_id = _selected_room_id()

    left, right = st.columns([1, 2], gap="medium")
    with left:
        st.fragment(run_every=conv_every)(_conversations_fragment)(user)
        st.divider()
        _render_new_dm(user)
        _render_new_group(user)
    with right:
        # Thread transcript is the auto-refreshing fragment; the composer lives
        # OUTSIDE it so typing and sending are never interrupted by the poll.
        st.fragment(run_every=thread_every)(_thread_fragment)(user)
        _render_composer(user, room_id)


def _render_composer(user, room_id: str) -> None:
    """In-column send box at the bottom of the thread pane.

    A form (not st.chat_input) so the composer stays INSIDE the thread column
    instead of docking full-width at the page bottom and overlapping the
    conversation list. clear_on_submit resets the box after each send, and the
    form defers submission so the 3s Live auto-refresh never loses a draft or
    fires a send. Repeated sends work because each submit reruns cleanly.
    """

    with st.form(key="msg_compose_form", clear_on_submit=True):
        box_col, send_col = st.columns([5, 1], gap="small")
        with box_col:
            text = st.text_input(
                "Message",
                placeholder="Type a message",
                key="msg_compose_input",
                label_visibility="collapsed",
            )
        with send_col:
            sent = st.form_submit_button("Send", width="stretch", type="primary")
    if sent:
        if (text or "").strip() and post_message(
            room_id, user.id, user.full_name or user.id, text
        ):
            st.rerun()
        else:
            st.warning("Message not sent. Make sure it is not empty.")


main()
