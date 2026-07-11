"""NaviLearn home: sign-in and the student progress dashboard.

This is the Streamlit entry point. It handles authentication (one-click demo
accounts plus a manual form) and, once signed in, routes by role:

- **student**: a progress dashboard rendered entirely from the repository
  (top metrics, progress-per-course, a 30-day activity trend, and a simple
  "recommended next" rule).
- **mentor / teacher**: their real dashboard rendered inline (via
  :mod:`core.dashboards`), with a compact row of page links below it.

The UI is a thin consumer of :mod:`core.session` and :class:`core.repo`; it
holds no business logic beyond shaping repository output for display.
"""

from __future__ import annotations

import logging
from datetime import datetime
from statistics import mean
from typing import Any, Optional

import streamlit as st

from core.exporters import activity_csv, progress_csv
from core.repo import Course, Profile, Repository
from core.session import (
    current_user,
    demo_accounts,
    get_repo_cached,
    login,
    logout,
)

_LOG = logging.getLogger(__name__)

_ROLES = ["student", "mentor", "teacher"]

# Staff dashboard bodies, imported defensively so a bad import in the dashboard
# module never hard-crashes Home. When unavailable, the staff branch degrades to
# a short message plus its page links (see :func:`_render_staff_landing`).
try:  # best-effort: Home must render even if a dashboard module fails to import.
    from core.dashboards import render_mentor_dashboard, render_teacher_dashboard
except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash Home.
    render_mentor_dashboard = None  # type: ignore[assignment]
    render_teacher_dashboard = None  # type: ignore[assignment]
    _LOG.warning("staff dashboards unavailable: %s", exc)

# Sparkline glyphs, low to high. Used for the Arrow-free activity trend.
_SPARK_GLYPHS = "▁▂▃▄▅▆▇█"


def _fmt_date(value: str) -> str:
    """Return a short, human-friendly date from an ISO timestamp.

    Falls back to the raw string (or a dash) when the value cannot be parsed,
    so a malformed timestamp never breaks a card.
    """

    if not value:
        return "n/a"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return value[:10] if len(value) >= 10 else value
    return parsed.strftime("%d %b %Y")


def _sparkline(values: list[float]) -> str:
    """Return a unicode sparkline for ``values`` (empty string if none).

    Values are min-max scaled onto the eight block glyphs. This renders a
    trend as plain markdown text, avoiding the Arrow/dataframe serialization
    that Streamlit chart elements use (see :func:`_render_activity_chart`).
    """

    if not values:
        return ""
    low = min(values)
    high = max(values)
    span = high - low
    glyphs: list[str] = []
    last = len(_SPARK_GLYPHS) - 1
    for value in values:
        index = 0 if span <= 0 else round((value - low) / span * last)
        glyphs.append(_SPARK_GLYPHS[index])
    return "".join(glyphs)


# --------------------------------------------------------------------------- #
# Sign-in
# --------------------------------------------------------------------------- #
def _render_login(repo: Repository) -> None:
    """Render the sign-in panel: one-click demo accounts and a manual form."""

    st.subheader("Sign in")
    st.caption(
        "Pick a demo account for an instant tour, or sign in manually as a "
        "student, mentor, or teacher."
    )

    accounts = demo_accounts(repo)
    if accounts:
        st.markdown("#### One-click demo accounts")
        cols = st.columns(len(accounts))
        for col, account in zip(cols, accounts):
            label = f"{account.full_name}\n\n{account.role.title()}"
            if col.button(label, key=f"demo_{account.id}", width="stretch"):
                login(account.email, account.full_name, account.role)
                st.rerun()

    st.divider()
    st.markdown("#### Or sign in manually")
    with st.form("manual_login"):
        email = st.text_input("Email", placeholder="you@navilearn.dev")
        full_name = st.text_input("Full name", placeholder="Your name")
        role = st.selectbox("Role", options=_ROLES, index=0)
        submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
    if submitted:
        if not (email or "").strip():
            st.warning("Enter an email to sign in.")
        else:
            login(email, full_name, role)
            st.rerun()


def _render_sidebar(user: Profile) -> None:
    """Render the account sidebar: identity, page hints, and logout."""

    with st.sidebar:
        st.header("NaviLearn")
        st.markdown(f"**{user.full_name}**")
        st.caption(f"Signed in as {user.role}")
        st.divider()
        st.markdown(
            "Use the pages above to open **Study Studio**, the "
            "**AI Interview**, and **Messages**."
        )
        st.divider()
        if st.button("Log out", width="stretch"):
            logout()
            st.rerun()


# --------------------------------------------------------------------------- #
# Student dashboard
# --------------------------------------------------------------------------- #
def _report_overall(scores: dict[str, Any]) -> Optional[float]:
    """Return an overall score for one report's ``scores`` mapping.

    Uses an explicit ``overall`` score when present, otherwise the mean of the
    numeric sub-scores. Returns ``None`` when there is nothing numeric to score.
    """

    scores = scores or {}
    overall = scores.get("overall")
    if isinstance(overall, (int, float)):
        return float(overall)
    numeric = [float(v) for v in scores.values() if isinstance(v, (int, float))]
    return round(mean(numeric), 1) if numeric else None


def _latest_interview_overall(repo: Repository, student_id: str) -> Optional[float]:
    """Return the latest interview overall score, or ``None`` if there is none."""

    reports = repo.list_interview_reports(student_id)
    if not reports:
        return None
    return _report_overall(reports[0].scores)


def _render_metrics(repo: Repository, user: Profile) -> None:
    """Render the four headline metrics for a student."""

    progress = repo.list_progress(user.id)
    lessons_completed = sum(1 for row in progress if row.status == "completed")
    total_hours = round(sum(row.time_spent_seconds for row in progress) / 3600.0, 1)

    by_course = repo.progress_by_course(user.id)
    in_progress = sum(1 for c in by_course if 0.0 < c["pct"] < 100.0)

    overall = _latest_interview_overall(repo, user.id)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lessons completed", lessons_completed)
    col2.metric("Time spent (hours)", total_hours)
    col3.metric("Courses in progress", in_progress)
    col4.metric(
        "Latest interview",
        f"{overall}/10" if overall is not None else "n/a",
    )


def _render_progress_chart(repo: Repository, user: Profile) -> None:
    """Render a bar chart of completion percentage per course."""

    by_course = repo.progress_by_course(user.id)
    st.markdown("#### Progress per course")
    if not by_course:
        st.info("No course progress yet. Start a lesson to see it here.")
        return
    # Rendered as labeled completion bars rather than a dataframe chart. Course
    # titles are strings, and serializing a string column/index to Arrow (which
    # st.bar_chart does) can crash pyarrow under Streamlit rerun threads. Native
    # progress bars keep the same "bar per course" reading with no Arrow path.
    for course in by_course:
        pct = float(course["pct"])
        st.markdown(
            f"**{course['course']}** · {course['completed']}/{course['total']} "
            f"lessons ({pct:.0f}%)"
        )
        st.progress(min(max(pct / 100.0, 0.0), 1.0))


def _render_progress_donut(repo: Repository, user: Profile) -> None:
    """Render a matplotlib donut of completed vs remaining lessons.

    Aggregates ``progress_by_course`` into a single completed/remaining split
    across all courses and draws it as a donut. matplotlib is used (with the
    Arrow-free ``Agg`` backend and ``st.pyplot``) deliberately: every Streamlit
    chart element serializes through Apache Arrow, whose pyarrow build segfaults
    on Streamlit's per-rerun worker threads. A matplotlib figure has no Arrow
    path, so it is safe under the testing harness.
    """

    import matplotlib

    matplotlib.use("Agg")  # headless, no Arrow, thread-safe for reruns
    import matplotlib.pyplot as plt

    by_course = repo.progress_by_course(user.id)
    completed = sum(int(c["completed"]) for c in by_course)
    total = sum(int(c["total"]) for c in by_course)
    remaining = max(total - completed, 0)

    st.markdown("#### Completion")
    if total <= 0:
        st.info("No lessons available yet.")
        return

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    if completed == 0:
        values = [1.0]
        colors = ["#e0e0e0"]
    else:
        values = [completed, remaining]
        colors = ["#2e7d32", "#e0e0e0"]
    ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 1.5},
    )
    pct = round(100.0 * completed / total) if total else 0
    ax.text(
        0.0,
        0.0,
        f"{pct}%",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#2e7d32",
    )
    ax.text(0.0, -0.28, "complete", ha="center", va="center", fontsize=9, color="#666")
    ax.set(aspect="equal")
    fig.patch.set_alpha(0.0)
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"{completed} of {total} lessons completed · {remaining} remaining")


def _render_exports(repo: Repository, user: Profile) -> None:
    """Render CSV download buttons for the student's progress and activity."""

    st.markdown("#### Export your data")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download progress (CSV)",
            data=progress_csv(repo, user.id),
            file_name="navilearn_progress.csv",
            mime="text/csv",
            width="stretch",
        )
    with col2:
        st.download_button(
            "Download activity (CSV)",
            data=activity_csv(repo, user.id),
            file_name="navilearn_activity.csv",
            mime="text/csv",
            width="stretch",
        )


def _render_activity_chart(repo: Repository, user: Profile) -> None:
    """Render a line chart of daily study minutes over the last 30 days."""

    series = repo.activity_timeseries(user.id, days=30)
    st.markdown("#### Activity (last 30 days)")
    if not series:
        st.info("No activity recorded in the last 30 days.")
        return
    # Rendered as a unicode sparkline plus summary metrics instead of a chart
    # element. Every Streamlit chart (line_chart, altair_chart, ...) serializes
    # its data through Apache Arrow; this build's pyarrow segfaults when that
    # serialization runs on a thread other than the one that first imported it,
    # which Streamlit's testing harness (a fresh thread per rerun) triggers. A
    # markdown sparkline conveys the same daily trend with no Arrow dependency.
    minutes = [float(point["minutes"]) for point in series]
    st.markdown(
        f"<div style='font-size:1.7rem; line-height:1; letter-spacing:2px'>"
        f"{_sparkline(minutes)}</div>",
        unsafe_allow_html=True,
    )
    active_days = sum(1 for value in minutes if value > 0)
    total_minutes = round(sum(minutes))
    best_day = round(max(minutes)) if minutes else 0
    col1, col2, col3 = st.columns(3)
    col1.metric("Total minutes", total_minutes)
    col2.metric("Active days", active_days)
    col3.metric("Best day (min)", best_day)


def _recommend_next(repo: Repository, user: Profile) -> Optional[dict[str, Any]]:
    """Return the recommended next course/lesson, or ``None`` if all done.

    Rule: pick the course with the lowest completion percentage that is not yet
    finished, then the first lesson in it that is not completed.
    """

    courses = repo.list_courses()
    if not courses:
        return None

    completed_lessons = {
        row.lesson_id for row in repo.list_progress(user.id)
        if row.status == "completed"
    }

    best: Optional[tuple[float, Course, list]] = None
    for course in courses:
        lessons = repo.list_lessons(course.id)
        total = len(lessons)
        if total == 0:
            continue
        done = sum(1 for lesson in lessons if lesson.id in completed_lessons)
        pct = 100.0 * done / total
        if pct >= 100.0:
            continue
        if best is None or pct < best[0]:
            best = (pct, course, lessons)

    if best is None:
        return None

    _pct, course, lessons = best
    next_lesson = next(
        (lesson for lesson in lessons if lesson.id not in completed_lessons),
        None,
    )
    return {"course": course, "lesson": next_lesson, "pct": round(_pct, 1)}


def _render_study_notes(repo: Repository, user: Profile) -> None:
    """Render the "My study notes" card: the student's recent saved study sets.

    Reads real saved sets via ``repo.list_study_sets`` and shows the five most
    recent with title, created date, and source. The full library lives in
    Study Studio, so the empty state and footer both point there.
    """

    st.markdown("#### My study notes")
    sets = repo.list_study_sets(user.id)[:5]
    if not sets:
        st.caption(
            "No saved study notes yet. Generate flashcards, summaries, and "
            "concept graphs in **Study Studio**, then save a set to see it here."
        )
        return
    for study_set in sets:
        source = (study_set.source or "").strip()
        suffix = f" · from {source}" if source else ""
        st.markdown(f"**{study_set.title}**")
        st.caption(f"Saved {_fmt_date(study_set.created_at)}{suffix}")
    st.caption("Your full library lives in **Study Studio**.")


def _render_interview_link() -> None:
    """Render the page link to the full, openable interviews page (best-effort)."""

    try:
        st.page_link(
            "pages/Interviews.py", label="Open all interviews", icon="🎯"
        )
    except Exception:  # noqa: BLE001 - navigation is best-effort.
        st.caption("Open **My Interviews** from the sidebar to see them all.")


def _render_interviews(repo: Repository, user: Profile) -> None:
    """Render the "My interviews" card: a summary plus a link to the full page.

    Shows up to five recent AI-interview reports with an overall score and the
    date as a summary, then links to the openable **My Interviews** page where
    each report expands to its full rubric breakdown and feedback. The empty
    state points at the AI Interview page.
    """

    st.markdown("#### My interviews")
    reports = repo.list_interview_reports(user.id)[:5]
    if not reports:
        st.caption(
            "No interviews yet. Run a mock interview on one of your projects in "
            "**AI Interview** to get scored feedback here."
        )
        _render_interview_link()
        return
    for report in reports:
        overall = _report_overall(report.scores)
        score = f"{overall}/10" if overall is not None else "n/a"
        title = report.project_title or "Untitled project"
        st.markdown(f"**{title}** · {score}")
        st.caption(f"Scored {_fmt_date(report.created_at)}")
    _render_interview_link()


def _render_messages(user: Profile) -> None:
    """Render the "Messages" card: conversation count and recent room names.

    The messaging module is imported defensively so a momentary import or
    backend failure logs and degrades to a caption rather than crashing Home.
    """

    st.markdown("#### Messages")
    try:
        from core.messaging import list_rooms

        rooms = list_rooms(user.id)
    except Exception as exc:  # best-effort: never crash Home over messaging
        _LOG.warning("messages card unavailable: %s", exc)
        st.caption("Messages are momentarily unavailable. Try again shortly.")
        return

    if not rooms:
        st.caption(
            "No conversations yet. Start one from the **Messages** page in the "
            "sidebar."
        )
        return

    count = len(rooms)
    label = "conversation" if count == 1 else "conversations"
    st.markdown(f"**{count} {label}**")
    for room in rooms[:4]:
        name = (room.name or "").strip() or "Untitled room"
        st.caption(f"· {name}")
    st.caption("**Open Messages** in the sidebar to read and reply.")


def _render_shared_with_you(user: Profile) -> None:
    """Render the "Shared with you" card: notes other users sent to this user.

    Lists notes delivered directly to ``user.id`` via ``core.notes`` targeted
    sharing (newest share first), each with its title, who shared it, and its
    date. The notes module is imported defensively so a momentary import or
    backend failure logs and degrades to a caption rather than crashing Home.
    """

    st.markdown("#### Shared with you")
    try:
        from core.notes import list_shared_with_me

        shared = list_shared_with_me(user.id)
    except Exception as exc:  # best-effort: never crash Home over shared notes
        _LOG.warning("shared-with-you card unavailable: %s", exc)
        st.caption("Shared notes are momentarily unavailable. Try again shortly.")
        return

    if not shared:
        st.caption("Nothing shared with you yet.")
        return

    for note in shared:
        title = (note.title or "").strip() or "Untitled note"
        sharer = (note.shared_by or "").strip() or "someone"
        st.markdown(f"**{title}**")
        st.caption(f"from {sharer} · {_fmt_date(note.updated_at or note.created_at)}")


def _render_mentor_link() -> None:
    """Render the page link to the full mentor feedback page (best-effort)."""

    try:
        st.page_link(
            "pages/Mentor_Feedback.py", label="Open mentor feedback", icon="🤝"
        )
    except Exception:  # noqa: BLE001 - navigation is best-effort.
        st.caption("Open **Mentor Feedback** from the sidebar to read and reply.")


def _render_mentor_feedback(user: Profile) -> None:
    """Render the "Mentor feedback" card: a summary plus a link to the full page.

    The full two-way conversation (and the reply box) now lives on the dedicated
    **Mentor Feedback** page. This card is a compact summary: it shows the most
    recent messages labelled by author, then links out to the full thread.
    Messages are read via :func:`core.mentoring.list_notes`, keyed by ``user.id``
    (the same runtime id the student carries), which the mentoring layer
    canonicalizes to match how a mentor stored them. The mentoring module is
    imported defensively so a momentary import or backend failure logs and
    degrades to a caption rather than crashing Home.
    """

    st.markdown("#### Feedback from your mentor")
    try:
        from core.mentoring import list_notes
    except Exception as exc:  # best-effort: never crash Home over mentoring
        _LOG.warning("mentor feedback card unavailable: %s", exc)
        st.caption("Mentor feedback is momentarily unavailable. Try again shortly.")
        return

    try:
        notes = list_notes(user.id)
    except Exception as exc:  # best-effort: never crash Home over mentoring
        _LOG.warning("mentor feedback thread unavailable: %s", exc)
        st.caption("Mentor feedback is momentarily unavailable. Try again shortly.")
        notes = []

    if not notes:
        st.caption(
            "No mentor feedback yet. Once a mentor reviews your work, their "
            "notes will appear here and you can reply to them."
        )
        _render_mentor_link()
        return

    # Show only the most recent few messages as a preview; the full thread and
    # the reply box live on the dedicated Mentor Feedback page.
    for note in notes[-3:]:
        role = (note.get("author_role") or "mentor").strip() or "mentor"
        is_student = role == "student"
        avatar = "🎓" if is_student else "🧑‍🏫"
        chat_role = "user" if is_student else "assistant"
        author = (note.get("author_name") or "").strip() or (
            "You" if is_student else "Mentor"
        )
        text = (note.get("text") or "").strip()
        with st.chat_message(chat_role, avatar=avatar):
            if text:
                st.write(text)
            st.caption(f"{author} · {_fmt_date(note.get('created_at', ''))}")
    _render_mentor_link()


def _render_recommendation(repo: Repository, user: Profile) -> None:
    """Render the "recommended next" panel."""

    st.markdown("#### Recommended next")
    rec = _recommend_next(repo, user)
    if rec is None:
        st.success("You have completed every available course. Great work.")
        return
    st.markdown(f"**{rec['course'].title}** · {rec['pct']}% complete")
    lesson = rec["lesson"]
    if lesson is not None:
        st.write(f"Next lesson: **{lesson.title}**")
    else:
        st.write("Pick up where you left off in this course.")
    if rec["course"].description:
        st.caption(rec["course"].description)


def _render_student_dashboard(repo: Repository, user: Profile) -> None:
    """Render the full student dashboard as an organized card grid.

    Layout: a greeting header, the four headline metrics, then a two-column
    grid of bordered cards (each reading real repository data and pointing at a
    destination), the 30-day activity trend, and finally the CSV exports.
    """

    st.subheader(f"Welcome back, {user.full_name.split()[0]}")
    st.caption(
        "Your learning at a glance. Each card opens a tool from the sidebar."
    )
    _render_metrics(repo, user)
    st.divider()

    # Two-column card grid. Cards are laid out row by row so the left and right
    # columns stay balanced regardless of how much content each card holds.
    row1_left, row1_right = st.columns(2, gap="large")
    with row1_left:
        with st.container(border=True):
            _render_progress_chart(repo, user)
    with row1_right:
        with st.container(border=True):
            _render_study_notes(repo, user)

    row2_left, row2_right = st.columns(2, gap="large")
    with row2_left:
        with st.container(border=True):
            _render_interviews(repo, user)
    with row2_right:
        with st.container(border=True):
            _render_messages(user)

    row3_left, row3_right = st.columns(2, gap="large")
    with row3_left:
        with st.container(border=True):
            _render_recommendation(repo, user)
    with row3_right:
        with st.container(border=True):
            _render_progress_donut(repo, user)

    with st.container(border=True):
        _render_shared_with_you(user)

    with st.container(border=True):
        _render_mentor_feedback(user)

    st.divider()
    with st.container(border=True):
        _render_activity_chart(repo, user)

    st.divider()
    _render_exports(repo, user)
    st.caption(
        "Study Studio, AI Interview, and Messages live in the sidebar pages "
        "on the left."
    )


# --------------------------------------------------------------------------- #
# Mentor / teacher landing
# --------------------------------------------------------------------------- #
def _render_staff_tool_links(role: str) -> None:
    """Render a compact row of page links to the staff tools below the dashboard.

    Teachers get a link to the cohort-wide Teacher Dashboard page; both roles get
    the mentor page and the shared authoring, classroom, messaging, and learn
    tools. All targets are real, working pages.
    """

    tiles: list[tuple[str, str, str]] = [
        ("pages/3_Mentor.py", "Mentor Dashboard", "👥"),
    ]
    if role == "teacher":
        tiles.append(("pages/7_Teacher.py", "Teacher Dashboard", "📊"))
    tiles += [
        ("pages/9_Author.py", "Author courses", "✏️"),
        ("pages/5_Classroom.py", "Live Classroom", "💬"),
        ("pages/6_Messages.py", "Messages", "📨"),
        ("pages/8_Learn.py", "Learn (student view)", "🎓"),
    ]

    st.markdown("#### More dashboards and tools")
    columns = st.columns(len(tiles))
    for column, (path, label, icon) in zip(columns, tiles):
        with column:
            try:
                st.page_link(path, label=label, icon=icon)
            except Exception:  # noqa: BLE001 - navigation is best-effort.
                st.markdown(f"{icon} {label}")


def _render_staff_landing(repo: Repository, user: Profile) -> None:
    """Render a signed-in mentor or teacher's dashboard inline on Home.

    A one-line welcome header sits above the role-appropriate dashboard body
    (mentor mentee-management, or the teacher cohort overview), followed by a
    compact row of links to the other staff pages. The dashboard render is
    wrapped best-effort: if it is unavailable or errors, Home degrades to a short
    message plus the links instead of crashing.
    """

    st.subheader(f"Welcome, {user.full_name.split()[0]}")
    st.caption(f"You are signed in as a **{user.role}**. Your dashboard is below.")

    renderer = (
        render_teacher_dashboard if user.role == "teacher" else render_mentor_dashboard
    )
    if renderer is None:
        st.info(
            "Your dashboard could not be loaded right now. Use the links below to "
            "open it on its own page."
        )
    else:
        try:
            renderer(repo, user)
        except Exception as exc:  # noqa: BLE001 - never crash Home over a dashboard.
            _LOG.warning("inline staff dashboard failed for %s: %s", user.role, exc)
            st.info(
                "Your dashboard hit a snag loading here. Use the links below to "
                "open it on its own page."
            )

    st.divider()
    _render_staff_tool_links(user.role)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    """Render the NaviLearn home page: sign-in or the role-appropriate view."""

    st.set_page_config(page_title="NaviLearn", page_icon="🧭", layout="wide")
    st.title("NaviLearn")

    repo = get_repo_cached()
    user = current_user()

    if user is None:
        st.caption(
            "A holistic, Indic-first learning platform for students, mentors, "
            "and teachers."
        )
        _render_login(repo)
        return

    _render_sidebar(user)

    if user.role == "student":
        _render_student_dashboard(repo, user)
    else:
        _render_staff_landing(repo, user)


main()
