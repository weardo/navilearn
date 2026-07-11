"""Teacher Dashboard: a school or admin view of the whole learner cohort.

Where the Mentor Dashboard drills into one assigned student, this page rolls the
entire cohort up into a single class-wide overview for a teacher or school
admin. It answers oversight questions at a glance: how many learners and mentors
exist, how well mentorship covers the cohort, how much study and interview
activity is happening, and where individual students stand.

The page is a thin shell: it sets up the page, gates on the teacher role, then
delegates the whole read-only cohort body to
:func:`core.dashboards.render_teacher_dashboard`, the single source of truth also
rendered inline on the Home page.
"""

from __future__ import annotations

import streamlit as st

from core.dashboards import render_teacher_dashboard
from core.session import get_repo_cached, require_user

# Roles allowed to see the class-wide oversight view.
_TEACHER_ROLES = {"teacher", "mentor"}


def main() -> None:
    """Entry point: gate on the teacher role, then render the cohort dashboard."""

    st.set_page_config(page_title="Teacher | NaviLearn", page_icon="🧑‍🏫", layout="wide")
    user = require_user()
    st.title("Teacher Dashboard")

    role = (user.role or "").strip().lower()
    if role not in _TEACHER_ROLES:
        st.info(
            "This class-wide view is for teachers and school admins. Sign in with "
            "a teacher account to see cohort-level oversight."
        )
        return

    render_teacher_dashboard(get_repo_cached(), user)


main()
