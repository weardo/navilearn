"""Author: a mentor/teacher surface for creating course content.

Where the student-facing pages consume courses and lessons, this page is where
that material is *made*. A mentor or teacher can:

- create a new course (title plus description), and
- add a lesson to any existing course, giving it a title, a module/section
  name, a markdown body, an optional external video URL, and an optional
  uploaded file (a video or a document). Uploaded files land in the public
  ``course-media`` bucket via :func:`core.storage.upload_media`, and the
  returned public URL is stored on the lesson as ``video_url`` or ``doc_url``
  depending on the file type.

The page is a thin presentation layer over :class:`core.repo.Repository`
(``list_courses``, ``create_course``, ``list_lessons``, ``create_lesson``) and
:mod:`core.storage`. It is intended for mentors and teachers, but it renders for
any signed-in role so a single account can author without a role switch. Every
side effect is best-effort: a failed upload warns but never crashes the flow,
and a failed backend read degrades to a friendly empty state.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import streamlit as st

from core.repo import Course, Lesson
from core.session import get_repo_cached, require_user
from core.storage import upload_media

_LOG = logging.getLogger(__name__)

# File extensions/MIME hints we treat as video vs document, used to decide
# whether an uploaded file populates ``video_url`` or ``doc_url``.
_VIDEO_TYPES = ["mp4", "webm"]
_DOC_TYPES = ["pdf", "docx"]


def _safe_call(fn, *args, default: Any, what: str) -> Any:
    """Call ``fn(*args)`` returning ``default`` on any failure.

    Authoring reads (course and lesson lists) must never blank the page, so a
    flaky backend call is logged and swallowed per the project's best-effort
    side-effects convention.
    """

    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 - authoring read is best-effort.
        _LOG.warning("Author page: %s failed: %s", what, exc)
        return default


def _course_label(course: Course) -> str:
    """Return a human-friendly label for a course in a select box."""

    return (course.title or "").strip() or course.id


def _is_video_upload(filename: str, content_type: str) -> bool:
    """Return ``True`` when an uploaded file looks like a video.

    Decided from the MIME type first (``video/*``) and then the filename
    extension, so a browser that omits or mislabels the type still routes a
    ``.mp4`` or ``.webm`` file to ``video_url`` rather than ``doc_url``.
    """

    ctype = (content_type or "").lower()
    if ctype.startswith("video/"):
        return True
    if ctype.startswith("application/") and "pdf" in ctype:
        return False
    name = (filename or "").lower()
    return any(name.endswith(f".{ext}") for ext in _VIDEO_TYPES)


def _upload_and_classify(uploaded: Any) -> tuple[str, str]:
    """Upload a Streamlit file and return ``(video_url, doc_url)``.

    On success exactly one of the two URLs is set based on whether the file is a
    video or a document. Best-effort throughout: an empty upload, a storage
    failure, or an unresolved URL warns in the UI and returns two empty strings
    so the lesson is still created with whatever else the author supplied.
    """

    if uploaded is None:
        return "", ""

    try:
        data = uploaded.getvalue()
    except Exception as exc:  # noqa: BLE001 - reading the upload is best-effort.
        _LOG.warning("Author page: could not read uploaded file: %s", exc)
        st.warning("Could not read the uploaded file, so it was skipped.")
        return "", ""

    if not data:
        st.warning("The uploaded file was empty, so it was skipped.")
        return "", ""

    filename = getattr(uploaded, "name", "upload")
    content_type = getattr(uploaded, "type", "") or "application/octet-stream"

    url = _safe_call(
        upload_media,
        data,
        filename,
        content_type,
        default="",
        what="upload_media",
    )
    if not url:
        st.warning(
            "The file upload did not complete, so the lesson was saved without "
            "it. You can add a media link later."
        )
        return "", ""

    if _is_video_upload(filename, content_type):
        return url, ""
    return "", url


def _render_create_course(repo: Any) -> None:
    """Render the create-a-course form and persist it on submit."""

    st.subheader("Create a course")
    with st.form("author_create_course", clear_on_submit=True):
        title = st.text_input("Course title", placeholder="e.g. Python Foundations")
        description = st.text_area(
            "Description",
            placeholder="A short summary of what this course covers.",
        )
        submitted = st.form_submit_button("Create course", type="primary")

    if not submitted:
        return

    clean_title = (title or "").strip()
    if not clean_title:
        st.warning("A course needs a title.")
        return

    course = _safe_call(
        repo.create_course,
        Course(id="", title=clean_title, description=(description or "").strip()),
        default=None,
        what="create_course",
    )
    if course is None:
        st.error("Could not create the course. Please try again.")
        return

    st.success(f"Created course: {clean_title}")


def _render_existing_lessons(repo: Any, course_id: str) -> None:
    """List a course's lessons so authors see what students will get."""

    lessons = _safe_call(
        repo.list_lessons, course_id, default=[], what="list_lessons"
    )
    st.markdown("#### Existing lessons")
    if not lessons:
        st.caption("No lessons yet. Add the first one below.")
        return

    for lesson in lessons:
        module = (lesson.module or "").strip() or "No module"
        markers = []
        if (lesson.video_url or "").strip():
            markers.append("video")
        if (lesson.doc_url or "").strip():
            markers.append("doc")
        media = ", ".join(markers) if markers else "no media"
        title = (lesson.title or "").strip() or "Untitled lesson"
        st.markdown(f"- **{title}**  ·  {module}  ·  {media}")


def _render_add_lesson(repo: Any, courses: list[Course]) -> None:
    """Render the add-a-lesson form for a chosen course and persist on submit."""

    st.subheader("Add a lesson")
    if not courses:
        st.info("Create a course first, then add lessons to it.")
        return

    labels = {_course_label(course): course for course in courses}
    chosen_label = st.selectbox("Course", list(labels.keys()))
    course = labels.get(chosen_label)
    if course is None:
        return

    _render_existing_lessons(repo, course.id)

    with st.form("author_add_lesson", clear_on_submit=True):
        title = st.text_input("Lesson title", placeholder="e.g. Variables and Types")
        module = st.text_input(
            "Module / section", placeholder="e.g. Fundamentals"
        )
        content = st.text_area(
            "Lesson content (markdown)",
            height=220,
            placeholder="# Heading\n\nWrite the lesson body here using markdown.",
        )
        video_url = st.text_input(
            "External video URL (optional)",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        uploaded = st.file_uploader(
            "Upload a video or document (optional)",
            type=_VIDEO_TYPES + _DOC_TYPES,
            accept_multiple_files=False,
        )
        submitted = st.form_submit_button("Add lesson", type="primary")

    if not submitted:
        return

    clean_title = (title or "").strip()
    if not clean_title:
        st.warning("A lesson needs a title.")
        return

    uploaded_video, uploaded_doc = _upload_and_classify(uploaded)

    # An external video URL and an uploaded video both target ``video_url``; the
    # uploaded file wins when both are present since it is the fresher choice.
    final_video = uploaded_video or (video_url or "").strip()
    final_doc = uploaded_doc

    lesson = _safe_call(
        repo.create_lesson,
        Lesson(
            id="",
            course_id=course.id,
            title=clean_title,
            order_index=0,
            content=(content or "").strip(),
            module=(module or "").strip(),
            video_url=final_video,
            doc_url=final_doc,
        ),
        default=None,
        what="create_lesson",
    )
    if lesson is None:
        st.error("Could not add the lesson. Please try again.")
        return

    st.success(f"Added lesson: {clean_title}")
    st.rerun()


def main() -> None:
    """Render the mentor/teacher content authoring page."""

    st.set_page_config(page_title="Author | NaviLearn", page_icon="✏️")

    # Intended for mentors and teachers, but renders for any signed-in role so a
    # single account can author without switching roles.
    require_user()

    st.title("Author")
    st.caption(
        "Create courses and write the lessons students will study. Lessons "
        "support markdown, an external video link, and an uploaded video or "
        "document."
    )

    repo = get_repo_cached()

    _render_create_course(repo)

    st.divider()

    courses = _safe_call(repo.list_courses, default=[], what="list_courses")
    _render_add_lesson(repo, courses)


main()
