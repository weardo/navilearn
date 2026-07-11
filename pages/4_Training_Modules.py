"""Training Modules: turn recurring meeting notes into reusable LMS content.

This page realizes Challenge 7 as a mentor / teacher tool. A signed-in mentor or
teacher pastes (or uploads) the meeting notes they keep repeating in standups and
onboarding sessions, and NaviLearn distills them into self-contained, role-based
training modules (tutorial + how-to steps + FAQ) ready to drop into an LMS.

The page is a thin UI over :mod:`core.training_modules`. All model work is gated
behind the "Generate modules" button, so under Streamlit's AppTest harness (which
never clicks) the page renders instantly and spends nothing. The Ch6 pattern is
reused: extract text with :func:`core.parsers.extract_text`, call the shared LLM
gateway via :func:`core.llm.get_llm`, and parse defensively in the core module.
"""

from __future__ import annotations

import logging
import os
import tempfile

import streamlit as st

from core.config import get_settings
from core.llm import get_llm
from core.parsers import extract_text
from core.repo import ActivityEvent, Course, get_repo
from core.session import require_user
from core.training_modules import (
    PUBLISH_MODULE_LABEL,
    TrainingModule,
    extract_themes,
    generate_modules,
    publish_modules_as_lessons,
    render_all,
    render_markdown,
)

_LOG = logging.getLogger(__name__)

_MENTOR_ROLES = {"mentor", "teacher"}
_UPLOAD_TYPES = ["txt", "md", "docx"]
# Notes pasted into one text area are split into separate meetings on a divider
# so the model can spot themes that recur ACROSS meetings, not within one.
_DIVIDER = "---"
# Namespaced session-state key holding the last generated run. Results are stored
# here on Generate and rendered OUTSIDE the button block, so any later widget
# interaction (for example a download click) reruns the script without the button
# returning True yet still finds the modules to re-render.
_STATE_KEY = "training_modules_result"
# Namespaced session-state key holding the index of the module selected in the
# master-detail list. Clicking a title button on the left stores the index here;
# the right panel renders that module. Defaults to the first module.
_SELECTED_KEY = "training_modules_selected_index"


def _split_pasted(text: str) -> list[str]:
    """Split a single pasted block into per-meeting notes on a divider line."""

    if not text or not text.strip():
        return []
    parts = [part.strip() for part in text.split(_DIVIDER)]
    return [part for part in parts if part]


def _extract_uploads(uploaded_files) -> list[str]:
    """Extract text from each uploaded .txt/.md/.docx file, best-effort."""

    notes: list[str] = []
    for uploaded in uploaded_files or []:
        suffix = os.path.splitext(uploaded.name)[1] or ".txt"
        path = None
        try:
            handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            handle.write(uploaded.getvalue())
            handle.flush()
            handle.close()
            path = handle.name
            text, _title = extract_text(path)
            if text and text.strip():
                notes.append(text.strip())
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the rest.
            _LOG.warning("Could not read upload %s: %s", uploaded.name, exc)
            st.warning(f"Could not read {uploaded.name}: {exc}")
        finally:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
    return notes


def _render_module(module: TrainingModule, index: int) -> None:
    """Render one module with its steps, FAQs, and a Markdown download."""

    st.markdown(f"### {index}. {module.title}")
    meta: list[str] = []
    if module.role:
        meta.append(f"Role: {module.role}")
    if module.source_theme:
        meta.append(f"Theme: {module.source_theme}")
    if meta:
        st.caption("  ·  ".join(meta))

    if module.overview:
        st.write(module.overview)

    if module.steps:
        st.markdown("**Steps**")
        for step_index, step in enumerate(module.steps, start=1):
            st.markdown(f"{step_index}. {step}")

    if module.faqs:
        st.markdown("**FAQ**")
        for faq in module.faqs:
            question = str(faq.get("q", "")).strip()
            answer = str(faq.get("a", "")).strip()
            if not question:
                continue
            with st.expander(question):
                st.write(answer or "(no answer)")

    slug = "".join(c if c.isalnum() else "_" for c in module.title).strip("_") or "module"
    st.download_button(
        "Download this module (Markdown)",
        data=render_markdown(module),
        file_name=f"{slug}.md",
        mime="text/markdown",
        key=f"dl_module_{index}",
    )
    st.divider()


def _gather_notes(pasted: str, uploaded_files) -> list[str]:
    """Combine pasted (divider-split) and uploaded notes into one list."""

    notes = _split_pasted(pasted)
    notes.extend(_extract_uploads(uploaded_files))
    return notes


def _record_generation(student_id: str, modules: list[TrainingModule], themes: list[str]) -> None:
    """Record a best-effort activity event so a run shows up in analytics.

    There is no DB table for modules, so we persist the fact that a run happened
    (count + themes) as an :class:`ActivityEvent`. Telemetry must never break the
    tool, so any failure is logged and swallowed.
    """

    if not student_id or not modules:
        return
    try:
        repo = get_repo(get_settings())
        repo.record_activity(
            ActivityEvent(
                id="",
                student_id=student_id,
                type="training_module_generated",
                payload={
                    "count": len(modules),
                    "titles": [module.title for module in modules][:8],
                    "themes": themes[:8],
                },
            )
        )
    except Exception as exc:  # noqa: BLE001 - analytics never breaks generation.
        _LOG.warning("Training-module activity not recorded: %s", exc)


def _generate(notes: list[str], max_modules: int, student_id: str) -> None:
    """Detect themes, generate modules, persist a best-effort run, and cache it.

    The result is written to ``st.session_state`` under :data:`_STATE_KEY` and
    rendered later, outside the button block, so it survives reruns.
    """

    if not notes:
        st.warning("Paste or upload at least one meeting note first.")
        return

    llm = get_llm(get_settings())

    with st.spinner("Detecting recurring themes and building modules..."):
        try:
            themes = extract_themes(llm, notes)
        except Exception as exc:  # noqa: BLE001 - theme detection is a nice-to-have.
            _LOG.warning("Theme detection failed: %s", exc)
            themes = []
        try:
            modules = generate_modules(llm, notes, max_modules=max_modules)
        except Exception as exc:  # noqa: BLE001 - surface the error, do not crash.
            st.error(f"Module generation failed: {exc}")
            return

    _record_generation(student_id, modules, themes)
    st.session_state[_STATE_KEY] = {"modules": modules, "themes": themes}


def _record_publish(student_id: str, course: Course, count: int) -> None:
    """Record a best-effort activity event that a publish happened.

    Telemetry must never break the publish flow, so any failure is logged and
    swallowed.
    """

    if not student_id or count <= 0:
        return
    try:
        repo = get_repo(get_settings())
        repo.record_activity(
            ActivityEvent(
                id="",
                student_id=student_id,
                type="training_published",
                payload={
                    "course_id": course.id,
                    "course_title": course.title,
                    "lessons": count,
                },
            )
        )
    except Exception as exc:  # noqa: BLE001 - analytics never breaks publishing.
        _LOG.warning("Publish activity not recorded: %s", exc)


def _publish(
    modules: list[TrainingModule],
    choice: Course | None,
    new_title: str,
    student_id: str,
) -> None:
    """Publish the generated modules as real lessons in the chosen course.

    ``choice`` is an existing :class:`Course`, or ``None`` to create a new one
    from ``new_title``. Each module becomes a lesson under the
    :data:`PUBLISH_MODULE_LABEL` section so students can take it in Learn.
    """

    if not modules:
        st.warning("Generate some modules first.")
        return

    repo = get_repo(get_settings())

    if choice is None:
        title = (new_title or "").strip()
        if not title:
            st.warning("Enter a title for the new course.")
            return
        try:
            course = repo.create_course(
                Course(id="", title=title, description="Published from Training Modules.")
            )
        except Exception as exc:  # noqa: BLE001 - surface the failure, do not crash.
            st.error(f"Could not create the course: {exc}")
            return
    else:
        course = choice

    created = publish_modules_as_lessons(repo, course.id, modules)
    _record_publish(student_id, course, len(created))

    if created:
        st.success(
            f"Published {len(created)} lesson(s) to '{course.title}'. They now "
            "appear under Learn, grouped as "
            f"'{PUBLISH_MODULE_LABEL}', for students to take."
        )
    else:
        st.error("No lessons were published. Please try again.")


def _render_publish(modules: list[TrainingModule], student_id: str) -> None:
    """Render the 'Publish to a course' section under the generated modules.

    A mentor picks an existing course (or creates a new one), then publishes
    every generated module into it as a real lesson. All widgets carry stable
    keys so a rerun (for example after a download click) preserves selections.
    """

    st.divider()
    st.subheader("Publish to a course")
    st.caption(
        "Turn these modules into real lessons students can take in Learn. "
        "Pick an existing course or create a new one."
    )

    repo = get_repo(get_settings())
    try:
        courses = repo.list_courses()
    except Exception as exc:  # noqa: BLE001 - never crash the tool on a read.
        _LOG.warning("Could not list courses for publishing: %s", exc)
        st.warning(f"Could not load existing courses: {exc}")
        courses = []

    # ``None`` is the "create a new course" sentinel, shown first.
    options: list[Course | None] = [None] + list(courses)
    choice = st.selectbox(
        "Course",
        options,
        format_func=lambda c: "Create a new course" if c is None else c.title,
        key="training_publish_course",
    )

    new_title = ""
    if choice is None:
        new_title = st.text_input("New course title", key="training_publish_new_title")

    if st.button("Publish modules", key="training_publish_btn", width="stretch"):
        _publish(modules, choice, new_title, student_id)


def _render_results(student_id: str = "") -> None:
    """Render the last generated run from session state, if any.

    This runs on every rerun (not only right after Generate), so downloads and
    other widget interactions re-read the cached modules instead of wiping them.
    """

    result = st.session_state.get(_STATE_KEY)
    if not result:
        return

    themes = result.get("themes") or []
    modules = result.get("modules") or []

    if themes:
        st.subheader("Recurring themes")
        st.write("  ·  ".join(themes))

    if not modules:
        st.info("No modules could be generated from these notes.")
        return

    st.subheader(f"Training modules ({len(modules)})")

    # Clamp the stored selection to a valid module: a fresh run may return fewer
    # modules than a previous one, so a stale index must fall back to the first.
    selected = st.session_state.get(_SELECTED_KEY, 0)
    if not isinstance(selected, int) or selected < 0 or selected >= len(modules):
        selected = 0
        st.session_state[_SELECTED_KEY] = 0

    left, right = st.columns([1, 2])
    with left:
        for idx, module in enumerate(modules):
            label = f"{idx + 1}. {module.title}"
            if st.button(
                label,
                key=f"training_module_pick_{idx}",
                width="stretch",
                type="primary" if idx == selected else "secondary",
            ):
                st.session_state[_SELECTED_KEY] = idx
                selected = idx
    with right:
        _render_module(modules[selected], selected + 1)

    st.download_button(
        "Download all modules (Markdown)",
        data=render_all(modules),
        file_name="training_modules.md",
        mime="text/markdown",
        key="dl_all_modules",
    )

    _render_publish(modules, student_id)


def main() -> None:
    """Entry point: gate on the mentor/teacher role, then render the tool."""

    st.set_page_config(page_title="Training Modules | NaviLearn", page_icon="📎")
    user = require_user()
    st.title("Training Modules")
    st.caption(
        "Turn recurring standup and onboarding notes into reusable, LMS-ready "
        "training modules."
    )

    role = (user.role or "").strip().lower()
    if role not in _MENTOR_ROLES:
        st.info(
            "This tool is for mentors and teachers. Sign in with a mentor or "
            "teacher account to turn meeting notes into training modules."
        )
        return

    st.markdown(
        "Paste one or more meeting notes below, separating each meeting with a "
        f"line containing `{_DIVIDER}`. You can also upload .txt, .md, or .docx "
        "files. The more the notes repeat, the better the modules."
    )

    sample_path = os.path.join("data", "samples", "meeting_notes.md")
    default_text = ""
    if os.path.exists(sample_path):
        try:
            with open(sample_path, "r", encoding="utf-8") as handle:
                default_text = handle.read()
        except OSError:
            default_text = ""

    pasted = st.text_area(
        "Meeting notes",
        value=default_text,
        height=300,
        help=f"Separate distinct meetings with a `{_DIVIDER}` line.",
    )
    uploaded_files = st.file_uploader(
        "Or upload notes",
        type=_UPLOAD_TYPES,
        accept_multiple_files=True,
    )
    max_modules = st.slider("Maximum modules", min_value=1, max_value=8, value=5)

    if st.button("Generate modules", type="primary", width="stretch"):
        notes = _gather_notes(pasted, uploaded_files)
        _generate(notes, max_modules, user.id)

    # Render OUTSIDE the button block so downloads and other reruns (when the
    # button returns False) never wipe the results.
    _render_results(user.id)


main()
