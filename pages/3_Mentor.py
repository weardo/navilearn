"""Mentor Dashboard: a two-way workspace for a mentor and their mentees.

This page realizes Challenge 4's mentor role as something a mentor can actually
*do* things in, not just look at. A signed-in mentor sees only the students
assigned to them (:func:`core.mentoring.list_students_for_mentor`), and for each
one can review course progress, recent interview reports, and saved study sets,
then leave written feedback that persists as a mentor note. A dedicated section
lets the mentor claim any student who has no mentor yet.

The page itself is a thin shell: it sets up the page, gates on the mentor role,
then delegates the whole dashboard body to
:func:`core.dashboards.render_mentor_dashboard`, which is the single source of
truth also rendered inline on the Home page.
"""

from __future__ import annotations

import streamlit as st

from core.dashboards import render_mentor_dashboard
from core.session import get_repo_cached, require_user

_MENTOR_ROLES = {"mentor", "teacher"}


def main() -> None:
    """Entry point: gate on the mentor role, then render the dashboard."""

    st.set_page_config(page_title="Mentor Dashboard", layout="wide")
    user = require_user()
    st.title("Mentor Dashboard")

    role = (user.role or "").strip().lower()
    if role not in _MENTOR_ROLES:
        st.info(
            "This view is for mentors and teachers. Sign in with a mentor or "
            "teacher account to monitor and coach your students."
        )
        return

    render_mentor_dashboard(get_repo_cached(), user)


main()
