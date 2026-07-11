"""Learn: a Coursera-style course player that closes the progress loop.

A signed-in student picks a course, then works through it inside a two-pane
layout that mirrors Coursera: a persistent left OUTLINE (a hierarchy of modules
and lessons) beside a right CONTENT pane for the lesson that is open. The outline
shows, at a glance, what is done (a check), what is open (a filled dot), and what
is still ahead (an open circle), plus a per-module completed count and an overall
progress bar. Opening a lesson shows its media (video and/or document) and its
markdown body; marking it complete writes a :class:`ProgressRow`, records a
``lesson_completed`` activity event, and advances to the next lesson so the
outline and progress bar update immediately.

The page is a thin UI over the backend-agnostic :class:`Repository`. It reads the
signed-in user with :func:`require_user` and the cached repository with
:func:`get_repo_cached`, mirroring every other page. All repository calls are
wrapped defensively: a failure degrades a single panel to a friendly message and
is logged, never crashing a flow. Navigation state (the selected course and open
lesson) lives in ``st.session_state`` under the ``learn_`` namespace so reruns
preserve where the learner is.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import streamlit as st
import streamlit.components.v1 as components

from core.repo import (
    ActivityEvent,
    Course,
    Lesson,
    ProgressRow,
    Repository,
)
from core.session import get_repo_cached, require_user

_LOG = logging.getLogger(__name__)

# Session-state keys, namespaced so they never collide with widget keys or other
# pages' state. ``_COURSE_KEY`` holds the id of the open course; ``_LESSON_KEY``
# the id of the open lesson within it.
_COURSE_KEY = "learn_course_id"
_LESSON_KEY = "learn_lesson_id"

# Outline status icons. A completed lesson gets a check, the open lesson a filled
# dot, everything else an open circle, so state reads at a glance in the tree.
_ICON_COMPLETED = "✓"
_ICON_CURRENT = "●"
_ICON_TODO = "○"

# Content-type icons hint at what a lesson holds before it is opened.
_ICON_VIDEO = "🎬"
_ICON_DOC = "📄"
_ICON_READING = "📖"

# Fixed time credited per completion, matching the previous Learn behaviour.
_COMPLETION_SECONDS = 300


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""

    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Repository reads (all defensive)
# --------------------------------------------------------------------------- #
def _status_by_lesson(repo: Repository, student_id: str) -> dict[str, str]:
    """Return a map of lesson id to progress status for one student.

    Best-effort: a repository failure yields an empty map so the page still
    renders every lesson as "not started" instead of crashing.
    """

    try:
        rows = repo.list_progress(student_id)
    except Exception as exc:  # noqa: BLE001 - a read failure must not crash the page.
        _LOG.warning("Could not load progress for %s: %s", student_id, exc)
        return {}
    return {row.lesson_id: row.status for row in rows}


def _list_lessons(repo: Repository, course_id: str) -> list[Lesson]:
    """Return a course's lessons ordered by ``order_index``, best-effort."""

    try:
        return repo.list_lessons(course_id)
    except Exception as exc:  # noqa: BLE001 - degrade to an empty course.
        _LOG.warning("Could not load lessons for course %s: %s", course_id, exc)
        return []


def _list_courses(repo: Repository) -> list[Course]:
    """Return all courses, best-effort (empty list on failure)."""

    try:
        return repo.list_courses()
    except Exception as exc:  # noqa: BLE001 - degrade to no courses.
        _LOG.warning("Could not load courses: %s", exc)
        return []


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #
def _course_progress(
    lessons: list[Lesson], status_by_lesson: dict[str, str]
) -> tuple[int, int, float]:
    """Return (completed, total, fraction) for a course's lessons.

    ``fraction`` is in the range 0.0 to 1.0 so it can feed ``st.progress``.
    """

    total = len(lessons)
    done = sum(1 for les in lessons if status_by_lesson.get(les.id) == "completed")
    fraction = (done / total) if total else 0.0
    return done, total, fraction


def _module_name(lesson: Lesson) -> str:
    """Return a lesson's module name, defaulting to a generic heading."""

    return (lesson.module or "").strip() or "Lessons"


def _group_by_module(lessons: list[Lesson]) -> list[tuple[str, list[Lesson]]]:
    """Group ordered lessons into (module, lessons) sections, order preserved.

    Modules appear in the order their first lesson appears, and lessons keep
    their ``order_index`` order within a module. Lessons with no module fall
    under a generic "Lessons" heading.
    """

    order: dict[str, int] = {}
    groups: list[tuple[str, list[Lesson]]] = []
    for lesson in lessons:
        module = _module_name(lesson)
        if module not in order:
            order[module] = len(groups)
            groups.append((module, []))
        groups[order[module]][1].append(lesson)
    return groups


def _first_incomplete(
    lessons: list[Lesson], status_by_lesson: dict[str, str]
) -> Optional[Lesson]:
    """Return the first not-yet-completed lesson in order, or ``None`` if done."""

    for lesson in lessons:
        if status_by_lesson.get(lesson.id) != "completed":
            return lesson
    return None


def _content_icon(lesson: Lesson) -> str:
    """Return the icon hinting at a lesson's primary content type."""

    if lesson.video_url:
        return _ICON_VIDEO
    if lesson.doc_url:
        return _ICON_DOC
    return _ICON_READING


def _status_icon(lesson: Lesson, status_by_lesson: dict[str, str], is_current: bool) -> str:
    """Return the outline status icon for one lesson."""

    if status_by_lesson.get(lesson.id) == "completed":
        return _ICON_COMPLETED
    if is_current:
        return _ICON_CURRENT
    return _ICON_TODO


def _looks_like_pdf(url: str) -> bool:
    """Return whether a document URL points at a PDF (best-effort, by path)."""

    path = (url or "").split("?", 1)[0].split("#", 1)[0].lower()
    return path.endswith(".pdf")


def _is_http_url(url: str) -> bool:
    """Return whether ``url`` is a plain http(s) URL safe to embed.

    Blocks javascript:, data:, and other schemes that could execute in an embed.
    """

    try:
        return urlparse(url or "").scheme in ("http", "https")
    except (ValueError, TypeError):
        return False


def _neighbours(
    lessons: list[Lesson], lesson_id: str
) -> tuple[Optional[Lesson], Optional[Lesson]]:
    """Return the (previous, next) lessons around ``lesson_id`` across modules.

    Navigation follows the whole course's ``order_index`` order, so Next crosses
    module boundaries. Either side is ``None`` at the ends of the course.
    """

    ids = [les.id for les in lessons]
    try:
        idx = ids.index(lesson_id)
    except ValueError:
        return None, None
    prev_lesson = lessons[idx - 1] if idx > 0 else None
    next_lesson = lessons[idx + 1] if idx < len(lessons) - 1 else None
    return prev_lesson, next_lesson


# --------------------------------------------------------------------------- #
# Progress writes
# --------------------------------------------------------------------------- #
def _mark_complete(repo: Repository, student_id: str, lesson: Lesson) -> bool:
    """Persist a completed progress row and record a completion event.

    Returns ``True`` when the progress write succeeded. The activity event is a
    best-effort follow-up: its failure is logged but does not fail completion,
    since progress is what the dashboard reads.
    """

    try:
        repo.upsert_progress(
            ProgressRow(
                id="",
                student_id=student_id,
                lesson_id=lesson.id,
                status="completed",
                time_spent_seconds=_COMPLETION_SECONDS,
                completed_at=_now_iso(),
            )
        )
    except Exception as exc:  # noqa: BLE001 - surface, do not crash.
        _LOG.warning("Could not mark lesson %s complete: %s", lesson.id, exc)
        st.error(f"Could not save your progress: {exc}")
        return False

    try:
        repo.record_activity(
            ActivityEvent(
                id="",
                student_id=student_id,
                type="lesson_completed",
                payload={"lesson_id": lesson.id, "title": lesson.title},
            )
        )
    except Exception as exc:  # noqa: BLE001 - telemetry never blocks completion.
        _LOG.warning("Completion activity not recorded for %s: %s", lesson.id, exc)
    return True


def _mark_not_complete(repo: Repository, student_id: str, lesson: Lesson) -> bool:
    """Move a lesson back to in-progress, clearing its completion timestamp."""

    try:
        repo.upsert_progress(
            ProgressRow(
                id="",
                student_id=student_id,
                lesson_id=lesson.id,
                status="in_progress",
                time_spent_seconds=0,
                completed_at=None,
            )
        )
    except Exception as exc:  # noqa: BLE001 - surface, do not crash.
        _LOG.warning("Could not reopen lesson %s: %s", lesson.id, exc)
        st.error(f"Could not update your progress: {exc}")
        return False
    return True


# --------------------------------------------------------------------------- #
# Left pane: the course outline (module -> lesson hierarchy)
# --------------------------------------------------------------------------- #
def _render_outline(
    lessons: list[Lesson],
    status_by_lesson: dict[str, str],
    current_id: Optional[str],
) -> None:
    """Render the persistent left outline: modules with their lessons.

    Each module is an expander (expanded when it holds the open lesson) headed by
    its name and a "k/n" completed count. Each lesson is a full-width button
    labelled with a status icon, a content-type icon, and its title. The open
    lesson's button is primary and disabled; clicking any other opens it.
    """

    st.markdown("#### Course content")
    for module, module_lessons in _group_by_module(lessons):
        done = sum(
            1
            for les in module_lessons
            if status_by_lesson.get(les.id) == "completed"
        )
        total = len(module_lessons)
        has_current = any(les.id == current_id for les in module_lessons)
        with st.expander(f"{module}  ({done}/{total})", expanded=has_current):
            for lesson in module_lessons:
                is_current = lesson.id == current_id
                label = (
                    f"{_status_icon(lesson, status_by_lesson, is_current)} "
                    f"{_content_icon(lesson)} {lesson.title}"
                )
                if st.button(
                    label,
                    key=f"learn_outline_{lesson.id}",
                    width="stretch",
                    type="primary" if is_current else "secondary",
                    disabled=is_current,
                ):
                    st.session_state[_LESSON_KEY] = lesson.id
                    st.rerun()


# --------------------------------------------------------------------------- #
# Right pane: the open lesson's content
# --------------------------------------------------------------------------- #
def _render_lesson_media(lesson: Lesson) -> None:
    """Render a lesson's optional video and document, each best-effort."""

    if lesson.video_url:
        try:
            st.video(lesson.video_url)
        except Exception as exc:  # noqa: BLE001 - a bad URL must not crash the page.
            _LOG.warning("Could not render video for %s: %s", lesson.id, exc)
            st.caption("The video for this lesson could not be loaded.")

    if lesson.doc_url:
        st.link_button("Open document", lesson.doc_url)
        if _looks_like_pdf(lesson.doc_url) and _is_http_url(lesson.doc_url):
            # Inline preview for PDFs. Use the component iframe (which sets src as
            # a DOM property, not via string interpolation) plus an http(s)-only
            # check, so an author-supplied doc_url cannot inject iframe attributes
            # or markup. The link button above is the reliable fallback if the
            # browser blocks the embed.
            try:
                components.iframe(lesson.doc_url, height=600)
            except Exception as exc:  # noqa: BLE001 - embed is a nice-to-have.
                _LOG.warning("Could not embed document for %s: %s", lesson.id, exc)


def _render_content(
    repo: Repository,
    student_id: str,
    course: Course,
    lessons: list[Lesson],
    lesson: Lesson,
    status_by_lesson: dict[str, str],
) -> None:
    """Render the right content pane for the open lesson.

    Shows a breadcrumb, the title, media, the markdown body, the completion
    controls, and Prev/Next navigation across the whole course.
    """

    st.caption(f"{course.title}  ›  {_module_name(lesson)}  ›  {lesson.title}")
    st.header(lesson.title)

    _render_lesson_media(lesson)

    content = (lesson.content or "").strip()
    if content:
        st.markdown(content)
    else:
        st.info("This lesson has no written material yet.")

    st.divider()

    prev_lesson, next_lesson = _neighbours(lessons, lesson.id)
    status = status_by_lesson.get(lesson.id, "not_started")

    action_col, badge_col = st.columns([2, 3])
    with action_col:
        if status == "completed":
            if st.button("Mark not complete", key="learn_mark_not_complete", width="stretch"):
                if _mark_not_complete(repo, student_id, lesson):
                    st.rerun()
        else:
            if st.button(
                "Mark complete",
                key="learn_mark_complete",
                type="primary",
                width="stretch",
            ):
                if _mark_complete(repo, student_id, lesson):
                    # Advance to the next lesson so the learner keeps moving and
                    # the outline / progress bar refresh on the rerun.
                    if next_lesson is not None:
                        st.session_state[_LESSON_KEY] = next_lesson.id
                    st.rerun()
    with badge_col:
        if status == "completed":
            st.success("Completed", icon="✅")

    # Prev / Next navigation across the whole course, crossing module boundaries.
    nav_prev, nav_next = st.columns(2)
    with nav_prev:
        if st.button(
            "← Previous",
            key="learn_prev",
            width="stretch",
            disabled=prev_lesson is None,
        ):
            if prev_lesson is not None:
                st.session_state[_LESSON_KEY] = prev_lesson.id
                st.rerun()
    with nav_next:
        if st.button(
            "Next →",
            key="learn_next",
            width="stretch",
            disabled=next_lesson is None,
        ):
            if next_lesson is not None:
                st.session_state[_LESSON_KEY] = next_lesson.id
                st.rerun()

    if next_lesson is not None:
        st.caption(f"Next up: {next_lesson.title}")
    else:
        st.caption("This is the last lesson in the course.")


# --------------------------------------------------------------------------- #
# Course selection + header
# --------------------------------------------------------------------------- #
def _select_course(courses: list[Course]) -> Course:
    """Return the course to show, offering a selector when there is more than one.

    The choice is remembered in ``st.session_state`` under ``_COURSE_KEY`` so it
    survives reruns. A single course needs no selector.
    """

    if len(courses) == 1:
        st.session_state[_COURSE_KEY] = courses[0].id
        return courses[0]

    ids = [course.id for course in courses]
    stored = st.session_state.get(_COURSE_KEY)
    index = ids.index(stored) if stored in ids else 0
    chosen = st.selectbox(
        "Course",
        options=courses,
        index=index,
        format_func=lambda course: course.title,
        key="learn_course_select",
    )
    st.session_state[_COURSE_KEY] = chosen.id
    return chosen


def _resolve_open_lesson(
    lessons: list[Lesson], status_by_lesson: dict[str, str]
) -> Optional[Lesson]:
    """Return the lesson to open: the stored one if valid, else the first to do.

    Keeps the stored selection when it still belongs to the current course.
    Otherwise auto-selects the first incomplete lesson (falling back to the very
    first lesson) so the content pane is never empty on a fresh course, mirroring
    how Coursera drops you straight into where you left off.
    """

    by_id = {les.id: les for les in lessons}
    stored = st.session_state.get(_LESSON_KEY)
    if stored in by_id:
        return by_id[stored]

    chosen = _first_incomplete(lessons, status_by_lesson) or (lessons[0] if lessons else None)
    if chosen is not None:
        st.session_state[_LESSON_KEY] = chosen.id
    else:
        st.session_state.pop(_LESSON_KEY, None)
    return chosen


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    """Render the Coursera-style Learn experience for the signed-in student."""

    st.set_page_config(page_title="Learn | NaviLearn", page_icon="🎓", layout="wide")
    user = require_user()

    repo = get_repo_cached()
    status_by_lesson = _status_by_lesson(repo, user.id)

    courses = _list_courses(repo)
    if not courses:
        st.title("Learn")
        st.info(
            "No courses are available yet. Once a teacher publishes a course it "
            "will appear here for you to work through."
        )
        return

    course = _select_course(courses)
    lessons = _list_lessons(repo, course.id)

    # Header: course title and an overall progress bar for this course.
    done, total, fraction = _course_progress(lessons, status_by_lesson)
    st.markdown(f"## {course.title}")
    if course.description:
        st.caption(course.description)
    st.progress(fraction, text=f"{done} of {total} lessons complete")

    if not lessons:
        st.info("This course has no lessons yet.")
        return

    open_lesson = _resolve_open_lesson(lessons, status_by_lesson)
    current_id = open_lesson.id if open_lesson is not None else None

    outline_col, content_col = st.columns([1, 2], gap="large")
    with outline_col:
        _render_outline(lessons, status_by_lesson, current_id)
    with content_col:
        if open_lesson is None:
            st.info("Select a lesson from the outline to begin.")
        else:
            _render_content(
                repo, user.id, course, lessons, open_lesson, status_by_lesson
            )


main()
