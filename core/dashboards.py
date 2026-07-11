"""Reusable staff dashboard bodies: mentor mentee-management and teacher cohort.

This module holds the render *logic* for the two staff dashboards so they can be
shown both on their dedicated pages (``pages/3_Mentor.py``, ``pages/7_Teacher.py``)
and inline on the Home page for signed-in mentors and teachers. It is a pure UI
module: it imports Streamlit and draws widgets, but it never calls
``st.set_page_config`` or ``require_user`` and never gates on role, so the caller
stays in control of page setup and access.

Two public entry points:

- :func:`render_mentor_dashboard` renders the full mentor workspace body: the
  mentee roster + per-student detail (progress, interview reports, study sets),
  the feedback form + note history, and the "assign a student to me" section.
- :func:`render_teacher_dashboard` renders the whole teacher cohort overview:
  headline metrics, the per-student table, and the cohort activity chart.

Reads flow through :class:`core.repo.Repository`; mentor writes and mentor-scoped
lookups flow through :mod:`core.mentoring`. Every backend call is best-effort so
a Supabase hiccup degrades a panel instead of crashing the page.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
import streamlit as st

from core import mentoring
from core.repo import InterviewReport, Profile, StudySet

_LOG = logging.getLogger(__name__)

# Score fields an :class:`InterviewReport` may carry, in display order.
_SCORE_FIELDS = [
    ("overall", "Overall"),
    ("technical_depth", "Technical depth"),
    ("clarity", "Clarity"),
    ("originality", "Originality"),
    ("implementation_understanding", "Implementation"),
]

# Score field on an interview report treated as the headline "latest score".
_OVERALL_SCORE_KEY = "overall"

# Window, in days, for the cohort activity trend.
_ACTIVITY_DAYS = 21


# --------------------------------------------------------------------------- #
# Mentor dashboard
# --------------------------------------------------------------------------- #
def _student_label(student: Profile) -> str:
    """Return a human-friendly label for a student in a picker or heading."""

    name = (student.full_name or "").strip() or student.email or student.id
    email = (student.email or "").strip()
    return f"{name} ({email})" if email and email != name else name


def _render_progress(progress: list[dict]) -> None:
    """Render per-course completion as a bar chart plus a compact table."""

    st.markdown("**Course progress**")
    if not progress:
        st.caption("No course progress recorded yet.")
        return
    frame = pd.DataFrame(progress)
    chart = frame.set_index("course")[["pct"]].rename(columns={"pct": "Percent complete"})
    st.bar_chart(chart, width="stretch")
    st.dataframe(
        frame[["course", "completed", "total", "pct"]].rename(
            columns={
                "course": "Course",
                "completed": "Completed",
                "total": "Lessons",
                "pct": "Percent",
            }
        ),
        hide_index=True,
        width="stretch",
    )


def _render_study_sets(study_sets: list[StudySet]) -> None:
    """List the student's saved study sets, newest first."""

    st.markdown("**Recent study sets**")
    if not study_sets:
        st.caption("No study sets saved yet.")
        return
    for study_set in study_sets[:5]:
        created = (study_set.created_at or "")[:10]
        title = study_set.title or "Untitled set"
        source = study_set.source or ""
        suffix = f" · {source}" if source else ""
        st.markdown(f"- **{title}**{suffix}  \n  _{created}_")


def _render_interview_reports(reports: list[InterviewReport]) -> None:
    """Show each recent scored interview report with its per-dimension scores."""

    st.markdown("**Recent interview reports**")
    if not reports:
        st.caption("No interview reports yet.")
        return
    for report in reports[:5]:
        created = (report.created_at or "")[:10]
        title = report.project_title or "Untitled project"
        with st.container(border=True):
            st.markdown(f"**{title}**  ·  _{created}_")
            scores = report.scores or {}
            present = [(key, label) for key, label in _SCORE_FIELDS if key in scores]
            if present:
                cols = st.columns(len(present))
                for col, (key, label) in zip(cols, present):
                    value = scores.get(key)
                    display = f"{float(value):.1f}" if value is not None else "n/a"
                    col.metric(label, f"{display} / 10")
            if report.feedback:
                st.markdown("**Feedback**")
                st.write(report.feedback)


def _render_notes_history(student_id: str) -> None:
    """Render the mentor-note history for a student, newest first."""

    notes = mentoring.list_notes(student_id)
    st.markdown("**Feedback history**")
    if not notes:
        st.caption("No feedback left yet. Be the first to write a note.")
        return
    for note in notes:
        stamp = (note.get("created_at") or "")[:16].replace("T", " ")
        author = note.get("mentor_name") or "Mentor"
        meta = f"{author} · {stamp}" if stamp else author
        st.markdown(f"> {note.get('text', '')}  \n  _{meta}_")


def _render_feedback_form(mentor: Profile, student: Profile) -> None:
    """Render the write-feedback form and persist a note on submit."""

    form_key = f"note-form-{student.id}"
    with st.form(form_key, clear_on_submit=True):
        text = st.text_area(
            "Write feedback for this student",
            key=f"note-text-{student.id}",
            placeholder="Share encouragement, next steps, or specific pointers.",
            height=100,
        )
        submitted = st.form_submit_button("Send feedback")
    if submitted:
        mentor_name = (mentor.full_name or "").strip() or mentor.email or "Mentor"
        ok = mentoring.save_note(student.id, mentor.id, mentor_name, text)
        if ok:
            st.success("Feedback sent.")
            st.rerun()
        else:
            st.warning("Could not send feedback. Enter some text and try again.")


def _render_student_detail(repo, mentor: Profile, student: Profile) -> None:
    """Render the full two-way detail panel for one mentee in the right column."""

    st.subheader(_student_label(student))
    try:
        _render_progress(repo.progress_by_course(student.id))
    except Exception:  # noqa: BLE001 - a read hiccup must not blank the panel.
        st.caption("Course progress is unavailable right now.")

    left, right = st.columns(2)
    with left:
        try:
            _render_study_sets(repo.list_study_sets(student.id))
        except Exception:  # noqa: BLE001
            st.caption("Study sets are unavailable right now.")
    with right:
        try:
            _render_interview_reports(repo.list_interview_reports(student.id))
        except Exception:  # noqa: BLE001
            st.caption("Interview reports are unavailable right now.")

    st.divider()
    _render_feedback_form(mentor, student)
    _render_notes_history(student.id)


def _render_students_master_detail(
    repo, mentor: Profile, students: list[Profile]
) -> None:
    """Render students as a two-column master-detail workspace.

    The left column is a clickable button list of the mentor's students; the
    right column is the selected student's detail panel. The first student is
    selected by default when nothing is chosen yet, and a stale selection (a
    student no longer assigned) falls back to the first student.
    """

    ids = [s.id for s in students]
    selected_id = st.session_state.get("mentor_selected_student")
    if selected_id not in ids:
        selected_id = ids[0]
        st.session_state["mentor_selected_student"] = selected_id

    left, right = st.columns([1, 2])
    with left:
        for student in students:
            is_selected = student.id == selected_id
            if st.button(
                _student_label(student),
                key=f"mentor-student-{student.id}",
                width="stretch",
                type="primary" if is_selected else "secondary",
            ):
                st.session_state["mentor_selected_student"] = student.id
                st.rerun()
    with right:
        current = next((s for s in students if s.id == selected_id), students[0])
        _render_student_detail(repo, mentor, current)


def _render_assign_section(mentor: Profile) -> None:
    """Render the "claim an unassigned student" control."""

    st.subheader("Assign a student to me")
    unassigned = mentoring.list_unassigned_students()
    if not unassigned:
        st.caption("Every student already has a mentor. Nothing to assign.")
        return
    labels = [_student_label(s) for s in unassigned]
    choice = st.selectbox(
        "Students without a mentor",
        options=range(len(unassigned)),
        format_func=lambda i: labels[i],
        key="assign-select",
    )
    if st.button("Assign to me", key="assign-button"):
        student = unassigned[choice]
        if mentoring.assign_mentor(student.id, mentor.id):
            st.success(f"{_student_label(student)} is now one of your students.")
            st.rerun()
        else:
            st.warning("Could not assign that student. Please try again.")


def render_mentor_dashboard(repo, user: Profile) -> None:
    """Render the full mentor mentee-management dashboard body.

    Draws everything the Mentor page shows after its title: a caption, the
    mentor's student roster as a master-detail workspace (or an empty-state
    hint), and the "assign a student to me" section. Assumes the caller has
    already gated on the mentor/teacher role and set up the page.
    """

    st.caption(f"Signed in as {user.full_name or user.email} · mentor view")

    students = mentoring.list_students_for_mentor(user.id)
    if not students:
        st.info(
            "You have no students yet. Use the 'Assign a student to me' section "
            "below to claim a mentee, then come back to coach them."
        )
    else:
        st.subheader(f"My students ({len(students)})")
        _render_students_master_detail(repo, user, students)

    st.divider()
    _render_assign_section(user)


# --------------------------------------------------------------------------- #
# Teacher dashboard
# --------------------------------------------------------------------------- #
@dataclass
class StudentSummary:
    """One row of the per-student cohort table.

    All numeric fields default to safe zeros or ``None`` so a student with no
    data still renders a complete, non-crashing row.
    """

    student_id: str
    name: str
    email: str
    overall_pct: float = 0.0
    courses_in_progress: int = 0
    study_sets: int = 0
    latest_score: Optional[float] = None
    mentor_name: str = "unassigned"


def _student_name(profile: Profile) -> str:
    """Return a human-friendly display name for a profile."""

    return (profile.full_name or "").strip() or profile.email or profile.id


def _safe_call(fn, *args, default: Any, what: str) -> Any:
    """Call ``fn(*args)`` returning ``default`` on any failure.

    Cohort rollups touch every student, so one bad row must not crash the page.
    Failures are logged and swallowed (best-effort), matching the project's
    side-effects convention.
    """

    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 - oversight read is best-effort.
        _LOG.warning("Teacher dashboard: %s failed: %s", what, exc)
        return default


def _overall_progress_pct(course_rows: list[dict[str, Any]]) -> tuple[float, int]:
    """Reduce per-course progress into one cohort-friendly pair.

    Returns ``(overall_percent, courses_in_progress)`` where the overall percent
    is lessons completed divided by total lessons across all courses (so long
    courses weigh more than short ones), and courses-in-progress counts courses
    with at least one completed lesson but not yet finished.
    """

    total = 0
    completed = 0
    in_progress = 0
    for row in course_rows or []:
        row_total = int(row.get("total", 0) or 0)
        row_done = int(row.get("completed", 0) or 0)
        total += row_total
        completed += row_done
        if row_done > 0 and row_done < row_total:
            in_progress += 1
        elif row_done > 0 and row_total == 0:
            # Defensive: completed lessons but no known total still counts.
            in_progress += 1
    pct = round(100.0 * completed / total, 1) if total else 0.0
    return pct, in_progress


def _latest_overall_score(reports: list[Any]) -> Optional[float]:
    """Return the overall score of the most recent interview report.

    Reports from the repository arrive newest-first; we still sort defensively
    on ``created_at`` so ordering assumptions never silently corrupt the number.
    Returns ``None`` when there is no scored report.
    """

    if not reports:
        return None
    ordered = sorted(
        reports, key=lambda r: getattr(r, "created_at", "") or "", reverse=True
    )
    for report in ordered:
        scores = getattr(report, "scores", None) or {}
        value = scores.get(_OVERALL_SCORE_KEY)
        if value is None:
            continue
        try:
            return round(float(value), 1)
        except (TypeError, ValueError):
            continue
    return None


def _build_summaries(
    repo, students: list[Profile], mentor_names: dict[str, str]
) -> list[StudentSummary]:
    """Assemble one :class:`StudentSummary` per student from repository reads."""

    summaries: list[StudentSummary] = []
    for student in students:
        course_rows = _safe_call(
            repo.progress_by_course,
            student.id,
            default=[],
            what="progress_by_course",
        )
        overall_pct, courses_in_progress = _overall_progress_pct(course_rows)

        study_sets = _safe_call(
            repo.list_study_sets, student.id, default=[], what="list_study_sets"
        )
        reports = _safe_call(
            repo.list_interview_reports,
            student.id,
            default=[],
            what="list_interview_reports",
        )

        mentor_id = (student.mentor_id or "").strip()
        mentor_name = mentor_names.get(mentor_id, "unassigned") if mentor_id else "unassigned"

        summaries.append(
            StudentSummary(
                student_id=student.id,
                name=_student_name(student),
                email=(student.email or "").strip(),
                overall_pct=overall_pct,
                courses_in_progress=courses_in_progress,
                study_sets=len(study_sets),
                latest_score=_latest_overall_score(reports),
                mentor_name=mentor_name,
            )
        )
    return summaries


def _render_headline(
    summaries: list[StudentSummary], mentor_count: int
) -> None:
    """Render the top row of cohort-wide headline metrics."""

    total_students = len(summaries)
    with_mentor = sum(1 for s in summaries if s.mentor_name != "unassigned")
    coverage = round(100.0 * with_mentor / total_students, 0) if total_students else 0.0
    total_study_sets = sum(s.study_sets for s in summaries)
    scored = [s.latest_score for s in summaries if s.latest_score is not None]
    total_reports = len(scored)
    avg_score = round(sum(scored) / len(scored), 1) if scored else None

    st.subheader("Cohort at a glance")
    row_one = st.columns(3)
    row_one[0].metric("Students", total_students)
    row_one[1].metric("Mentors", mentor_count)
    row_one[2].metric(
        "Mentor coverage",
        f"{coverage:.0f}%",
        help=f"{with_mentor} of {total_students} students have a mentor assigned.",
    )

    row_two = st.columns(3)
    row_two[0].metric("Study sets created", total_study_sets)
    row_two[1].metric("Students with interview scores", total_reports)
    row_two[2].metric(
        "Average latest interview score",
        f"{avg_score:.1f} / 10" if avg_score is not None else "n/a",
        help="Mean of each student's most recent overall interview score.",
    )


def _render_student_table(summaries: list[StudentSummary]) -> None:
    """Render the per-student cohort table."""

    st.subheader("Students")
    if not summaries:
        st.caption("No students to show.")
        return

    frame = pd.DataFrame(
        [
            {
                "Student": s.name,
                "Email": s.email,
                "Progress %": s.overall_pct,
                "Courses in progress": s.courses_in_progress,
                "Study sets": s.study_sets,
                "Latest interview": (
                    f"{s.latest_score:.1f}" if s.latest_score is not None else "n/a"
                ),
                "Mentor": s.mentor_name,
            }
            for s in summaries
        ]
    )
    st.dataframe(frame, hide_index=True, width="stretch")


def _render_cohort_activity(repo, summaries: list[StudentSummary]) -> None:
    """Sum each student's activity time series into one cohort trend."""

    st.subheader(f"Cohort learning activity (last {_ACTIVITY_DAYS} days)")
    if not summaries:
        st.caption("No activity to show.")
        return

    totals: dict[str, float] = {}
    any_data = False
    for summary in summaries:
        series = _safe_call(
            repo.activity_timeseries,
            summary.student_id,
            _ACTIVITY_DAYS,
            default=[],
            what="activity_timeseries",
        )
        for point in series or []:
            day = str(point.get("date", ""))
            if not day:
                continue
            minutes = float(point.get("minutes", 0) or 0)
            totals[day] = totals.get(day, 0.0) + minutes
            if minutes:
                any_data = True

    if not totals or not any_data:
        st.caption("No learning activity recorded across the cohort yet.")
        return

    frame = (
        pd.DataFrame(
            [{"date": day, "Minutes": round(mins, 1)} for day, mins in totals.items()]
        )
        .sort_values("date")
        .set_index("date")
    )
    st.bar_chart(frame, width="stretch")


def _mentor_name_map(repo) -> tuple[dict[str, str], int]:
    """Return a mentor-id to name map and the mentor count.

    Both explicit ``mentor`` and ``teacher`` profiles can be assigned as a
    student's mentor, so we index both. The count reflects dedicated mentors.
    """

    mentors = _safe_call(repo.list_profiles, "mentor", default=[], what="list_profiles(mentor)")
    teachers = _safe_call(repo.list_profiles, "teacher", default=[], what="list_profiles(teacher)")
    names: dict[str, str] = {}
    for profile in list(mentors) + list(teachers):
        names[profile.id] = _student_name(profile)
    return names, len(mentors)


def render_teacher_dashboard(repo, user: Profile) -> None:
    """Render the full teacher cohort dashboard body.

    Draws everything the Teacher page shows after its title: a caption, the
    cohort headline metrics, the per-student table, and the cohort activity
    chart (or an empty-state when no students are registered). Assumes the
    caller has already gated on the teacher role and set up the page.
    """

    st.caption(f"Signed in as {_student_name(user)} · cohort oversight")

    students = _safe_call(repo.list_profiles, "student", default=[], what="list_profiles(student)")
    if not students:
        st.info(
            "No students are registered yet. Once learners join, this dashboard "
            "will show cohort progress, study activity, and interview outcomes."
        )
        return

    mentor_names, mentor_count = _mentor_name_map(repo)
    summaries = _build_summaries(repo, students, mentor_names)

    _render_headline(summaries, mentor_count)
    st.divider()
    _render_student_table(summaries)
    st.divider()
    _render_cohort_activity(repo, summaries)
