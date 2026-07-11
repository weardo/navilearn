"""My Interviews: openable, scored AI-interview reports for a student.

The dashboard only ever listed interviews. This page makes each one openable:
every scored :class:`~core.repo.InterviewReport` is rendered as an expander whose
header carries the project title, date, and overall score, and whose body breaks
out each rubric dimension (0..10) as a labelled progress bar, restates the
overall score, and shows the full written feedback.

The page is a thin consumer of :mod:`core.session` and :class:`core.repo`; it
holds no business logic beyond shaping report scores for display.
"""

from __future__ import annotations

from datetime import datetime
from statistics import mean
from typing import Any, Optional

import streamlit as st

from core.session import get_repo_cached, require_user


def _fmt_date(value: str) -> str:
    """Return a short, human-friendly date from an ISO timestamp.

    Falls back to the raw string (or a dash) when the value cannot be parsed, so
    a malformed timestamp never breaks a report header.
    """

    if not value:
        return "n/a"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return value[:10] if len(value) >= 10 else value
    return parsed.strftime("%d %b %Y")


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


def _clamp_unit(value: float) -> float:
    """Return ``value / 10`` clamped to the 0..1 range for a progress bar."""

    return min(max(value / 10.0, 0.0), 1.0)


def _dimension_label(key: str) -> str:
    """Return a human-friendly label for a rubric dimension key."""

    return key.replace("_", " ").strip().title() or key


def _render_report(report: Any) -> None:
    """Render one interview report as an openable expander.

    The header shows the project title, when it was scored, and the overall
    score. Inside, each numeric rubric dimension (excluding the aggregate
    ``overall``) becomes a labelled progress bar on the 0..10 scale, the overall
    score is restated as a metric, and the full feedback text is shown below.
    """

    scores: dict[str, Any] = report.scores or {}
    overall = _report_overall(scores)
    title = (report.project_title or "").strip() or "Untitled project"
    overall_text = f"{overall}/10" if overall is not None else "n/a"
    header = f"{title}  ·  {_fmt_date(report.created_at)}  ·  {overall_text}"

    with st.expander(header):
        dimensions = [
            (key, float(value))
            for key, value in scores.items()
            if key != "overall" and isinstance(value, (int, float))
        ]
        if dimensions:
            st.markdown("**Rubric breakdown**")
            for key, value in dimensions:
                st.markdown(f"**{_dimension_label(key)}** · {value:g}/10")
                st.progress(_clamp_unit(value))
        else:
            st.caption("No per-dimension scores were recorded for this interview.")

        st.divider()
        st.metric("Overall score", overall_text)

        feedback = (report.feedback or "").strip()
        st.markdown("**Feedback**")
        if feedback:
            st.write(feedback)
        else:
            st.caption("No written feedback was recorded for this interview.")


def _render_reports(user: Any) -> None:
    """Render the signed-in student's interview reports, newest first.

    The empty state points at the AI Interview page so a student with no scored
    interviews knows where to start one.
    """

    repo = get_repo_cached()
    reports = repo.list_interview_reports(user.id)

    if not reports:
        st.info(
            "You have no scored interviews yet. Run a mock interview on one of "
            "your projects in **AI Interview** to get scored feedback here."
        )
        try:
            st.page_link(
                "pages/2_AI_Interview.py", label="Open AI Interview", icon="🎤"
            )
        except Exception:  # noqa: BLE001 - navigation is best-effort.
            st.caption("Open **AI Interview** from the sidebar to start one.")
        return

    count = len(reports)
    label = "interview" if count == 1 else "interviews"
    st.caption(f"{count} scored {label}. Open any one to see its full breakdown.")
    for report in reports:
        _render_report(report)


def main() -> None:
    """Gate on login, then render the student's openable interview reports."""

    st.set_page_config(page_title="My Interviews | NaviLearn", page_icon="🎯")
    user = require_user()
    st.title("My Interviews")
    _render_reports(user)


main()
