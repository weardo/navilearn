"""NaviLearn: a Streamlit web tool for multi-source learning ingestion.

Upload a document, paste text, or point at a YouTube URL and NaviLearn turns
it into a concept graph, flashcards, layered summaries and a searchable topic
index. The heavy lifting lives in ``core`` (see ``core.pipeline.process``);
this module is only the presentation and interaction layer.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

import streamlit as st

from core.artifacts import flashcards_to_csv, flashcards_to_json
from core.config import get_settings
from core.embeddings import get_embedder
from core.pipeline import ProcessResult, get_store, process
from core.store import TopicStore

_SUPPORTED_TYPES = ["pdf", "docx", "txt", "md", "srt", "vtt"]


def _active_provider() -> str:
    """Return a human-friendly label for the configured LLM provider."""

    settings = get_settings()
    provider = (settings.llm_provider or "unknown").strip()
    model = ""
    if provider == "groq":
        model = settings.groq_model
    elif provider == "openai":
        model = settings.openai_model
    else:
        model = settings.llm_model
    return f"{provider} ({model})" if model else provider


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
) -> tuple[Optional[str], Optional[str]]:
    """Turn the three inputs into a single source string for ``process``.

    Priority: an uploaded file, then a YouTube URL, then pasted text. Returns
    ``(source, error)`` where exactly one is non-None.
    """

    if uploaded is not None:
        try:
            return _save_upload_to_temp(uploaded), None
        except OSError as exc:
            return None, f"Could not save the uploaded file: {exc}"
    if youtube_url and youtube_url.strip():
        return youtube_url.strip(), None
    if pasted_text and pasted_text.strip():
        try:
            return _save_text_to_temp(pasted_text), None
        except OSError as exc:
            return None, f"Could not stage the pasted text: {exc}"
    return None, "Provide a file, a YouTube URL, or some pasted text first."


def _run_processing(source: str) -> None:
    """Process ``source`` and cache the result (or error) in session state."""

    try:
        result = process(source, store=get_store())
        st.session_state["result"] = result
        st.session_state["process_error"] = None
    except Exception as exc:  # noqa: BLE001 - surface any failure to the user.
        st.session_state["result"] = None
        st.session_state["process_error"] = str(exc)


def _render_overview(result: ProcessResult) -> None:
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
    else:
        st.info("No summary was produced for this source.")

    st.markdown("#### Topics")
    topics = result.concept_map.topics
    if topics:
        for topic in topics:
            st.markdown(f"- {topic}")
    else:
        st.caption("No topics were extracted.")


def _render_graph(result: ProcessResult) -> None:
    """Concept Graph tab: rendered DOT plus a JSON download."""

    concepts = result.concept_map.concepts
    if not concepts and not result.concept_map.edges:
        st.info("No concept graph could be built from this source.")
        return

    st.graphviz_chart(result.graph_dot, width="stretch")

    import json

    st.download_button(
        "Download graph JSON",
        data=json.dumps(result.graph_json, indent=2, ensure_ascii=False),
        file_name="concept_graph.json",
        mime="application/json",
        width="stretch",
    )


def _render_flashcards(result: ProcessResult) -> None:
    """Flashcards tab: one card each, with JSON and CSV downloads."""

    cards = result.flashcards
    if not cards:
        st.info("No flashcards were generated for this source.")
        return

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


def _render_summaries(result: ProcessResult) -> None:
    """Summaries tab: the overall summary and a per-topic expander list."""

    st.markdown("#### Overall")
    if result.summary.overall:
        st.write(result.summary.overall)
    else:
        st.info("No overall summary is available.")

    st.markdown("#### By topic")
    per_topic = result.summary.per_topic
    if per_topic:
        for topic, text in per_topic.items():
            with st.expander(topic):
                st.write(text or "(no summary)")
    else:
        st.caption("No per-topic summaries were produced.")


def _render_search(store: TopicStore) -> None:
    """Topic Search tab: semantic search plus browse-by-topic."""

    if store.count() == 0:
        st.info("The vector store is empty. Process a source to enable search.")
        return

    st.markdown("#### Semantic search")
    query = st.text_input("Search the ingested content", key="search_query")
    top_k = st.slider("Results", min_value=1, max_value=15, value=5, key="search_topk")

    if query and query.strip():
        try:
            embedder = get_embedder(get_settings())
            hits = store.search(query.strip(), embedder, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Search failed: {exc}")
            hits = []
        if hits:
            for hit in hits:
                with st.container(border=True):
                    meta = f"Source: {hit.get('source', '')}  |  Topic: {hit.get('topic', '')}"
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
    selected = st.selectbox("Topic", options=topics, key="browse_topic")
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


def main() -> None:
    """Render the NaviLearn single-page app."""

    st.set_page_config(page_title="NaviLearn", page_icon="📚", layout="wide")

    st.title("📚 NaviLearn")
    st.caption(
        "Turn any document, transcript or video into a concept graph, "
        "flashcards, summaries and a searchable topic index."
    )

    with st.sidebar:
        st.header("NaviLearn")
        st.write("Multi-source learning content ingestion.")
        st.markdown(f"**Active provider:** {_active_provider()}")
        st.divider()
        st.markdown(
            "Add one source below, then press **Process**. Results appear in "
            "the tabs and stay cached while you explore them."
        )

    st.subheader("1. Add a source")
    uploaded = st.file_uploader(
        "Upload a file",
        type=_SUPPORTED_TYPES,
        help="Supported: PDF, DOCX, TXT, MD, SRT, VTT.",
    )
    youtube_url = st.text_input(
        "or a YouTube URL", placeholder="https://www.youtube.com/watch?v=..."
    )
    pasted_text = st.text_area(
        "or paste raw text", height=140, placeholder="Paste notes or an article..."
    )

    if st.button("Process", type="primary", width="stretch"):
        source, error = _resolve_source(uploaded, youtube_url, pasted_text)
        if error:
            st.warning(error)
        else:
            with st.spinner("Ingesting and generating study artifacts..."):
                _run_processing(source)

    process_error = st.session_state.get("process_error")
    if process_error:
        st.error(f"Processing failed: {process_error}")

    result: Optional[ProcessResult] = st.session_state.get("result")

    st.subheader("2. Study artifacts")
    if result is None:
        st.info("No source processed yet. Add a source above and press Process.")
        return

    tab_overview, tab_graph, tab_cards, tab_summaries, tab_search = st.tabs(
        ["Overview", "Concept Graph", "Flashcards", "Summaries", "Topic Search"]
    )
    with tab_overview:
        _render_overview(result)
    with tab_graph:
        _render_graph(result)
    with tab_cards:
        _render_flashcards(result)
    with tab_summaries:
        _render_summaries(result)
    with tab_search:
        _render_search(get_store())


main()
