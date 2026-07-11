"""Study Studio: multi-source ingestion inside the NaviLearn platform.

This is the Ch6 Study Studio, wired into the platform's session and
repository layers. A signed-in learner adds one source (an uploaded
document, transcript, video or audio file, a YouTube URL, or pasted text),
presses Process, and the shared pipeline (:func:`core.pipeline.process`)
turns it into a concept graph, flashcards, layered summaries and a
searchable topic index.

The page is a thin UI over ``core``. On a successful process it persists a
:class:`core.repo.StudySet` for the current user and records a
``'study_session'`` activity event, so generated sets and study time surface
on the student dashboard. The heavy pipeline call is gated behind the Process
button so headless test runs (which never click) stay fast and spend nothing.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Optional

import streamlit as st

from core.artifacts import flashcards_to_csv, flashcards_to_json
from core.config import get_settings
from core.embeddings import get_embedder
from core.messaging import get_or_create_dm, list_directory, post_message
from core.notes import create_note
from core.pipeline import ProcessResult, get_store, process
from core.repo import ActivityEvent, Profile, StudySet
from core.session import current_user, get_repo_cached, require_user
from core.store import TopicStore
from core.summarizer import get_summarizer

_LOG = logging.getLogger(__name__)

_RESULT_KEY = "studio_result"
_ERROR_KEY = "studio_error"
_LABEL_KEY = "studio_source_label"
_OPEN_SET_KEY = "studio_open_set_id"

_SUPPORTED_TYPES = [
    "pdf", "docx", "txt", "md", "srt", "vtt",
    # Video/audio from any source: transcribed via the shared STT pipeline.
    "mp4", "mkv", "webm", "mov", "m4v", "mp3", "wav", "m4a", "ogg", "flac",
]


# --------------------------------------------------------------------------- #
# Source staging
# --------------------------------------------------------------------------- #
def _save_upload_to_temp(uploaded) -> str:
    """Persist a Streamlit UploadedFile to a temp path and return that path.

    The original extension is preserved so the parser can dispatch on it.
    """

    suffix = os.path.splitext(uploaded.name)[1] or ".txt"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(uploaded.getvalue())
    handle.flush()
    handle.close()
    return handle.name


def _save_text_to_temp(text: str) -> str:
    """Write pasted raw text to a temp Markdown file and return the path."""

    handle = tempfile.NamedTemporaryFile(
        delete=False, suffix=".md", mode="w", encoding="utf-8"
    )
    handle.write(text)
    handle.flush()
    handle.close()
    return handle.name


def _resolve_source(
    uploaded, youtube_url: str, pasted_text: str
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Turn the three inputs into a single source string for ``process``.

    Priority: an uploaded file, then a YouTube URL, then pasted text. Returns
    ``(source, label, error)`` where ``error`` is non-None only on failure. The
    ``label`` is a human-friendly description of the origin, used for the saved
    study set's ``source`` field.
    """

    if uploaded is not None:
        try:
            return _save_upload_to_temp(uploaded), f"upload:{uploaded.name}", None
        except OSError as exc:
            return None, None, f"Could not save the uploaded file: {exc}"
    if youtube_url and youtube_url.strip():
        url = youtube_url.strip()
        return url, url, None
    if pasted_text and pasted_text.strip():
        try:
            return _save_text_to_temp(pasted_text), "pasted text", None
        except OSError as exc:
            return None, None, f"Could not stage the pasted text: {exc}"
    return None, None, "Provide a file, a YouTube URL, or some pasted text first."


# --------------------------------------------------------------------------- #
# Processing + persistence
# --------------------------------------------------------------------------- #
def _result_to_content(result: ProcessResult, label: str) -> dict:
    """Serialize a :class:`ProcessResult` into a flat, JSON-safe dict.

    The returned dict holds every artifact needed to re-render a saved set
    after a refresh: overview metrics, topics, layered summaries, flashcards
    and the concept graph. Only plain strings, numbers, lists and dicts are
    used so the whole thing round-trips cleanly through the ``content`` jsonb
    column (Supabase) or a TEXT column of JSON (SQLite).
    """

    concept_map = result.concept_map
    flashcards = [
        {"front": card.front, "back": card.back, "topic": card.topic}
        for card in result.flashcards
    ]
    return {
        "title": result.title or "Untitled source",
        "source": label,
        "n_chunks": result.n_chunks,
        "n_concepts": len(concept_map.concepts),
        "topics": list(concept_map.topics),
        "summary_overall": result.summary.overall or "",
        "summary_per_topic": dict(result.summary.per_topic or {}),
        "flashcards": flashcards,
        "graph_dot": result.graph_dot or "",
        "graph_json": result.graph_json or {},
    }


def _persist_study_set(user: Profile, result: ProcessResult, label: str) -> None:
    """Save a StudySet and a study-session activity event for ``user``.

    Both writes are best-effort: a repository failure is logged and never
    breaks the study flow. The recorded event carries an estimated duration
    (roughly thirty seconds of study per ingested chunk) so the dashboard's
    activity trend reflects the session.
    """

    repo = get_repo_cached()
    title = result.title or "Untitled source"
    try:
        repo.save_study_set(
            StudySet(
                id="",
                owner_id=user.id,
                title=title,
                source=label,
                content=_result_to_content(result, label),
            )
        )
    except Exception as exc:  # noqa: BLE001 - persistence never blocks study.
        _LOG.warning("Study set not saved: %s", exc)

    try:
        repo.record_activity(
            ActivityEvent(
                id="",
                student_id=user.id,
                type="study_session",
                payload={"seconds": max(result.n_chunks, 1) * 30, "title": title},
            )
        )
    except Exception as exc:  # noqa: BLE001 - telemetry never blocks study.
        _LOG.warning("Study session activity not recorded: %s", exc)


def _run_processing(user: Profile, source: str, label: str) -> None:
    """Process ``source``, cache the result, and persist it for ``user``.

    On success the :class:`ProcessResult` is cached in session state and a
    study set plus activity event are written. On failure the error message is
    cached for display and no persistence occurs.
    """

    try:
        result = process(
            source, store=get_store(), owner=user.id, title=label
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure to the user.
        st.session_state[_RESULT_KEY] = None
        st.session_state[_ERROR_KEY] = str(exc)
        return

    st.session_state[_RESULT_KEY] = result
    st.session_state[_ERROR_KEY] = None
    st.session_state[_LABEL_KEY] = label
    _persist_study_set(user, result, label)


# --------------------------------------------------------------------------- #
# Save-to-Notes helpers (Evernote-style web clipper)
# --------------------------------------------------------------------------- #
def _clip_to_notes(user: Profile, title: str, body: str) -> None:
    """Clip an artifact into the learner's personal notes (best-effort).

    Calls :func:`core.notes.create_note` with a ``'study-studio'`` source and
    surfaces success. Any backend failure is logged and swallowed so a failed
    clip never crashes the Study Studio page.
    """

    try:
        create_note(user.id, title, body, source="study-studio")
        st.success("Saved to your Notes.")
    except Exception as exc:  # noqa: BLE001 - clipping never breaks the page.
        _LOG.warning("Save to Notes failed: %s", exc)
        st.warning("Could not save to your Notes. Please try again later.")


def _flashcards_to_note_body(result: ProcessResult) -> str:
    """Render every flashcard as a ``Q: ...\\nA: ...`` block for a note body."""

    blocks = [
        f"Q: {card.front}\nA: {card.back}" for card in result.flashcards
    ]
    return "\n\n".join(blocks)


def _concept_map_to_note_body(result: ProcessResult) -> str:
    """Render the concept map (topics plus the graph DOT) as Markdown.

    Topics become a bullet list and the full graph, including every edge, is
    embedded as a DOT code fence so the relationships round-trip into the note.
    """

    lines = ["## Topics", ""]
    topics = result.concept_map.topics
    if topics:
        lines.extend(f"- {topic}" for topic in topics)
    else:
        lines.append("(no topics extracted)")
    lines.extend(["", "## Concept graph (DOT)", "", "```dot"])
    lines.append(result.graph_dot or "")
    lines.append("```")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Result tabs
# --------------------------------------------------------------------------- #
def _render_overview(result: ProcessResult, user: Profile) -> None:
    """Overview tab: headline counts, the summary and the topic list."""

    st.subheader(result.title or "Untitled source")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Chunks", result.n_chunks)
    col2.metric("Concepts", len(result.concept_map.concepts))
    col3.metric("Flashcards", len(result.flashcards))
    col4.metric("Topics", len(result.concept_map.topics))

    st.markdown("#### Summary")
    if result.summary.overall:
        st.write(result.summary.overall)
        if st.button("Save to Notes", key="studio_save_summary"):
            _clip_to_notes(
                user,
                f"Summary: {result.title or 'Untitled source'}",
                result.summary.overall,
            )
    else:
        st.info("No summary was produced for this source.")

    st.markdown("#### Topics")
    topics = result.concept_map.topics
    if topics:
        for topic in topics:
            st.markdown(f"- {topic}")
    else:
        st.caption("No topics were extracted.")


def _render_graph(result: ProcessResult, user: Profile) -> None:
    """Concept Graph tab: rendered DOT plus a JSON download."""

    concepts = result.concept_map.concepts
    if not concepts and not result.concept_map.edges:
        st.info("No concept graph could be built from this source.")
        return

    st.graphviz_chart(result.graph_dot, width="stretch")
    st.download_button(
        "Download graph JSON",
        data=json.dumps(result.graph_json, indent=2, ensure_ascii=False),
        file_name="concept_graph.json",
        mime="application/json",
        width="stretch",
    )
    if st.button("Save to Notes", key="studio_save_graph"):
        _clip_to_notes(
            user,
            f"Concept map: {result.title or 'Untitled source'}",
            _concept_map_to_note_body(result),
        )


def _render_flashcards(result: ProcessResult, user: Profile) -> None:
    """Flashcards tab: one card each, with JSON and CSV downloads."""

    cards = result.flashcards
    if not cards:
        st.info("No flashcards were generated for this source.")
        return

    if st.button("Save to Notes", key="studio_save_flashcards"):
        _clip_to_notes(
            user,
            f"Flashcards: {result.title or 'Untitled source'}",
            _flashcards_to_note_body(result),
        )

    col1, col2 = st.columns(2)
    col1.download_button(
        "Download flashcards JSON",
        data=flashcards_to_json(cards),
        file_name="flashcards.json",
        mime="application/json",
        width="stretch",
    )
    col2.download_button(
        "Download flashcards CSV",
        data=flashcards_to_csv(cards),
        file_name="flashcards.csv",
        mime="text/csv",
        width="stretch",
    )

    st.divider()
    for index, card in enumerate(cards, start=1):
        with st.container(border=True):
            st.markdown(f"**Q{index}. {card.front}**")
            st.caption(f"Topic: {card.topic}")
            with st.expander("Show answer"):
                st.write(card.back)


def _render_offline_summary(result: ProcessResult) -> None:
    """On-device offline summary of the full ingested text (Ch2, no LLM).

    Gated behind a button so headless AppTest runs (which never click) skip the
    model load entirely. The quantized T5-small runs through onnxruntime with no
    LLM, no Groq and no rate limit: the same graphs that would ship to a browser
    via onnxruntime-web for a client-side deployment.
    """

    with st.expander("Offline summary (on-device, no LLM)", expanded=False):
        st.caption(
            "A quantized T5-small (Challenge 2) running client-side through "
            "onnxruntime: fully offline, no LLM call, no Groq, no rate limit."
        )
        if not (result.full_text and result.full_text.strip()):
            st.info("No ingested text is available to summarize offline.")
            return
        if st.button("Summarize offline (on-device)", key="studio_offline_btn"):
            with st.spinner("Running the on-device ONNX model..."):
                out = get_summarizer().summarize_timed(result.full_text)
            if out.text:
                st.write(out.text)
                m1, m2, m3 = st.columns(3)
                m1.metric("Model size", f"{out.model_size_mb:.0f} MB")
                m2.metric("Latency", f"{out.latency_ms:.0f} ms")
                m3.metric("Load time", f"{out.load_time_ms:.0f} ms")
            else:
                st.info(
                    "The offline model is unavailable on this machine, so no "
                    "on-device summary was produced. The LLM summary above "
                    "still applies."
                )


def _render_summaries(result: ProcessResult, user: Profile) -> None:
    """Summaries tab: the overall summary and a per-topic expander list."""

    title = result.title or "Untitled source"

    st.markdown("#### Overall")
    if result.summary.overall:
        st.write(result.summary.overall)
        if st.button("Save to Notes", key="studio_save_overall_summary"):
            _clip_to_notes(user, f"Summary: {title}", result.summary.overall)
    else:
        st.info("No overall summary is available.")

    _render_offline_summary(result)

    st.markdown("#### By topic")
    per_topic = result.summary.per_topic
    if per_topic:
        topics = list(per_topic.keys())
        sel_key = "studio_summary_topic_sel"
        selected = st.session_state.get(sel_key)
        if selected not in topics:
            selected = topics[0]
            st.session_state[sel_key] = selected

        col_list, col_detail = st.columns([1, 2])
        with col_list:
            for index, topic in enumerate(topics):
                if st.button(
                    topic,
                    key=f"studio_summary_topic_btn_{index}",
                    width="stretch",
                    type="primary" if topic == selected else "secondary",
                ):
                    st.session_state[sel_key] = topic
                    selected = topic
        with col_detail:
            text = per_topic.get(selected, "")
            st.markdown(f"**{selected}**")
            st.write(text or "(no summary)")
            if st.button("Save to Notes", key="studio_save_topic"):
                _clip_to_notes(
                    user,
                    f"Summary: {title} - {selected}",
                    text or "",
                )
    else:
        st.caption("No per-topic summaries were produced.")


def _render_search(store: TopicStore, user: Profile) -> None:
    """Topic Search tab: semantic search plus browse-by-topic.

    Semantic search is scoped to ``user`` so a learner only searches their own
    ingested content, never other learners' chunks.
    """

    if store.count() == 0:
        st.info("The vector store is empty. Process a source to enable search.")
        return

    st.markdown("#### Semantic search")
    query = st.text_input("Search the ingested content", key="studio_search_query")
    top_k = st.slider(
        "Results", min_value=1, max_value=15, value=5, key="studio_search_topk"
    )

    if query and query.strip():
        try:
            embedder = get_embedder(get_settings())
            hits = store.search(
                query.strip(), embedder, top_k=top_k, owner=user.id
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Search failed: {exc}")
            hits = []
        if hits:
            for hit in hits:
                with st.container(border=True):
                    meta = (
                        f"Source: {hit.get('source', '')}  |  "
                        f"Topic: {hit.get('topic', '')}"
                    )
                    st.caption(f"{meta}  |  Score: {hit.get('score', 0.0):.3f}")
                    st.write(hit.get("text", ""))
        else:
            st.warning("No passages matched that query.")

    st.divider()
    st.markdown("#### Browse by topic")
    topics = store.list_topics()
    if not topics:
        st.caption("No topics are stored yet.")
        return
    selected = st.selectbox("Topic", options=topics, key="studio_browse_topic")
    if selected:
        passages = store.search_by_topic(selected)
        if passages:
            st.caption(f"{len(passages)} passage(s) tagged '{selected}'.")
            for passage in passages:
                with st.container(border=True):
                    st.caption(f"Source: {passage.get('source', '')}")
                    st.write(passage.get("text", ""))
        else:
            st.caption("No passages are tagged with that topic.")


# --------------------------------------------------------------------------- #
# Saved library
# --------------------------------------------------------------------------- #
def _render_saved_set(content: dict) -> None:
    """Render a saved study set from its persisted ``content`` dict.

    Mirrors the live result tabs (overview, concept graph, flashcards,
    summaries) but reads everything from the stored, JSON-safe dict so a set
    survives refreshes. Missing keys degrade gracefully to empty defaults.
    """

    title = content.get("title") or "Untitled source"
    st.subheader(title)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Chunks", content.get("n_chunks", 0))
    col2.metric("Concepts", content.get("n_concepts", 0))
    col3.metric("Flashcards", len(content.get("flashcards", []) or []))
    col4.metric("Topics", len(content.get("topics", []) or []))

    tab_overview, tab_graph, tab_cards, tab_summaries = st.tabs(
        ["Overview", "Concept Graph", "Flashcards", "Summaries"]
    )

    with tab_overview:
        st.markdown("#### Summary")
        overall = content.get("summary_overall") or ""
        if overall:
            st.write(overall)
        else:
            st.info("No summary was saved for this set.")
        st.markdown("#### Topics")
        topics = content.get("topics", []) or []
        if topics:
            for topic in topics:
                st.markdown(f"- {topic}")
        else:
            st.caption("No topics were saved.")

    with tab_graph:
        graph_dot = content.get("graph_dot") or ""
        if graph_dot.strip():
            st.graphviz_chart(graph_dot, width="stretch")
        else:
            st.info("No concept graph was saved for this set.")
        graph_json = content.get("graph_json") or {}
        if graph_json:
            st.download_button(
                "Download graph JSON",
                data=json.dumps(graph_json, indent=2, ensure_ascii=False),
                file_name="concept_graph.json",
                mime="application/json",
                width="stretch",
                key="saved_graph_json",
            )

    with tab_cards:
        cards = content.get("flashcards", []) or []
        if not cards:
            st.info("No flashcards were saved for this set.")
        else:
            for index, card in enumerate(cards, start=1):
                with st.container(border=True):
                    st.markdown(f"**Q{index}. {card.get('front', '')}**")
                    if card.get("topic"):
                        st.caption(f"Topic: {card.get('topic')}")
                    with st.expander("Show answer"):
                        st.write(card.get("back", ""))

    with tab_summaries:
        st.markdown("#### Overall")
        overall = content.get("summary_overall") or ""
        if overall:
            st.write(overall)
        else:
            st.info("No overall summary was saved.")
        st.markdown("#### By topic")
        per_topic = content.get("summary_per_topic", {}) or {}
        if per_topic:
            topics = list(per_topic.keys())
            sel_key = "studio_saved_summary_topic_sel"
            selected = st.session_state.get(sel_key)
            if selected not in topics:
                selected = topics[0]
                st.session_state[sel_key] = selected

            col_list, col_detail = st.columns([1, 2])
            with col_list:
                for index, topic in enumerate(topics):
                    if st.button(
                        topic,
                        key=f"studio_saved_summary_topic_btn_{index}",
                        width="stretch",
                        type="primary" if topic == selected else "secondary",
                    ):
                        st.session_state[sel_key] = topic
                        selected = topic
            with col_detail:
                st.markdown(f"**{selected}**")
                st.write(per_topic.get(selected, "") or "(no summary)")
        else:
            st.caption("No per-topic summaries were saved.")


def _share_url(set_id: str) -> str:
    """Build the public share URL for a saved set: ``{scheme}://{host}/Shared?s=id``.

    The host is read from the request headers when available and falls back to
    the local dev host, so a copied link points at the deployment the user is
    actually on. Best-effort: any failure reading the headers uses the fallback.
    """

    host = "localhost:8600"
    try:
        headers = getattr(st.context, "headers", None)
        if headers:
            host = headers.get("host") or host
    except Exception as exc:  # noqa: BLE001 - header access is best-effort.
        _LOG.warning("Could not read request host: %s", exc)
    return f"http://{host}/Shared?s={set_id}"


def _render_share_section(user: Profile, study_set: StudySet) -> None:
    """Render the 'Share' controls for one saved set: link plus send-to-a-person.

    Shows a copyable read-only share URL and a directory picker that posts the
    link into a direct-message room with the chosen person. Both the directory
    lookup and the send are best-effort: a failure warns the user and never
    breaks the library view.
    """

    st.markdown("#### Share")
    url = _share_url(study_set.id)
    st.code(url, language=None)
    st.caption("Anyone with this link can view these notes.")

    st.markdown("**Send to a person**")
    try:
        directory = list_directory(user.id)
    except Exception as exc:  # noqa: BLE001 - directory is best-effort.
        _LOG.warning("Share directory lookup failed: %s", exc)
        directory = []

    if not directory:
        st.caption("No other people are available to send to yet.")
        return

    options = {
        f"{person.get('name', '')} ({person.get('role', '')})": person
        for person in directory
    }
    choice = st.selectbox(
        "Choose someone",
        options=list(options.keys()),
        key=f"studio_share_to_{study_set.id}",
    )
    if st.button("Send", key=f"studio_share_send_{study_set.id}"):
        person = options.get(choice or "")
        if not person:
            st.warning("Pick a person to send these notes to.")
            return
        try:
            room_id = get_or_create_dm(
                user.id, user.full_name, person["id"], person.get("name", "")
            )
            sent = post_message(
                room_id,
                user.id,
                user.full_name,
                f"I shared study notes with you: {url}",
            )
        except Exception as exc:  # noqa: BLE001 - sending never breaks the page.
            _LOG.warning("Share send failed: %s", exc)
            sent = False
        if sent:
            st.success(f"Shared with {person.get('name', 'them')}.")
        else:
            st.warning("Could not send the message. Please try again later.")


def _render_library(user: Profile) -> None:
    """Render '3. Your library': list, search and open saved study sets.

    Sets are listed newest-first. A case-insensitive search filters by title
    or source. Opening a set stashes its id in session state so the detail
    view survives reruns; a close control returns to the list.
    """

    st.subheader("3. Your library")

    try:
        sets = get_repo_cached().list_study_sets(user.id)
    except Exception as exc:  # noqa: BLE001 - the library never crashes the page.
        st.warning(f"Could not load your library: {exc}")
        return

    if not sets:
        st.info("No saved sets yet. Process a source above to build your library.")
        return

    open_id = st.session_state.get(_OPEN_SET_KEY)
    if open_id:
        opened = next((s for s in sets if s.id == open_id), None)
        if opened is None:
            st.session_state[_OPEN_SET_KEY] = None
        else:
            if st.button("X close", key="studio_lib_close"):
                st.session_state[_OPEN_SET_KEY] = None
                st.rerun()
            _render_saved_set(opened.content or {})
            st.divider()
            _render_share_section(user, opened)
            return

    query = st.text_input(
        "Search your library",
        key="studio_lib_search",
        placeholder="Filter by title or source...",
    )
    needle = (query or "").strip().lower()
    if needle:
        visible = [
            s
            for s in sets
            if needle in (s.title or "").lower()
            or needle in (s.source or "").lower()
        ]
    else:
        visible = sets

    if not visible:
        st.caption("No saved sets match that search.")
        return

    for study_set in visible:
        with st.container(border=True):
            col_info, col_open = st.columns([4, 1])
            with col_info:
                st.markdown(f"**{study_set.title or 'Untitled source'}**")
                meta = study_set.created_at or "unknown time"
                if study_set.source:
                    meta = f"{meta}  |  {study_set.source}"
                st.caption(meta)
            with col_open:
                if st.button("Open", key=f"studio_lib_open_{study_set.id}"):
                    st.session_state[_OPEN_SET_KEY] = study_set.id
                    st.rerun()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    """Render the Study Studio page for the signed-in learner."""

    st.set_page_config(page_title="Study Studio | NaviLearn", page_icon="📚")

    user = require_user()

    st.title("Study Studio")
    st.caption(
        "Turn any document, transcript, video or article into a concept graph, "
        "flashcards, summaries and a searchable topic index. Saved sets appear "
        "on your dashboard."
    )

    with st.sidebar:
        st.header("Study Studio")
        st.markdown(f"**{user.full_name}**")
        st.caption(f"Signed in as {user.role}")
        st.divider()
        st.markdown(
            "Add one source below, then press **Process**. Results appear in "
            "the tabs and stay cached while you explore them."
        )

    st.subheader("1. Add a source")
    uploaded = st.file_uploader(
        "Upload a file",
        type=_SUPPORTED_TYPES,
        help=(
            "Documents (PDF, DOCX, TXT, MD), transcripts (SRT, VTT), or any "
            "video/audio (transcribed via STT)."
        ),
    )
    youtube_url = st.text_input(
        "or a YouTube URL", placeholder="https://www.youtube.com/watch?v=..."
    )
    pasted_text = st.text_area(
        "or paste raw text", height=140, placeholder="Paste notes or an article..."
    )

    if st.button("Process", type="primary", width="stretch"):
        source, label, error = _resolve_source(uploaded, youtube_url, pasted_text)
        if error:
            st.warning(error)
        else:
            with st.spinner("Ingesting and generating study artifacts..."):
                _run_processing(user, source, label or source)

    process_error = st.session_state.get(_ERROR_KEY)
    if process_error:
        st.error(f"Processing failed: {process_error}")

    result: Optional[ProcessResult] = st.session_state.get(_RESULT_KEY)

    st.subheader("2. Study artifacts")
    if result is None:
        st.info("No source processed yet. Add a source above and press Process.")
    else:
        st.success("Saved to your library and dashboard.")

        tab_overview, tab_graph, tab_cards, tab_summaries, tab_search = st.tabs(
            ["Overview", "Concept Graph", "Flashcards", "Summaries", "Topic Search"]
        )
        with tab_overview:
            _render_overview(result, user)
        with tab_graph:
            _render_graph(result, user)
        with tab_cards:
            _render_flashcards(result, user)
        with tab_summaries:
            _render_summaries(result, user)
        with tab_search:
            _render_search(get_store(), user)

    st.divider()
    _render_library(user)


main()
