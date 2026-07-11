"""Offline tests for the swappable data layer (SQLite backend, temp db).

These never touch the network or Supabase. They exercise profile round-trips
and verify that seed_demo produces sensible progress-by-course and activity
time-series shapes for a dashboard.
"""

from __future__ import annotations

import os

import pytest

from core.repo import (
    Profile,
    SqliteRepo,
    get_repo,
    seed_demo,
)
from core.config import Settings


@pytest.fixture()
def repo(tmp_path) -> SqliteRepo:
    db_path = os.path.join(str(tmp_path), "navilearn_test.db")
    return SqliteRepo(db_path)


def test_get_repo_defaults_to_sqlite(tmp_path):
    settings = Settings(sqlite_path=os.path.join(str(tmp_path), "factory.db"))
    made = get_repo(settings)
    assert isinstance(made, SqliteRepo)


def test_upsert_and_get_profile(repo: SqliteRepo):
    saved = repo.upsert_profile(
        Profile(id="p1", email="a@b.dev", full_name="Ada Lovelace", role="student")
    )
    assert saved.id == "p1"

    fetched = repo.get_profile("p1")
    assert fetched is not None
    assert fetched.email == "a@b.dev"
    assert fetched.full_name == "Ada Lovelace"
    assert fetched.role == "student"

    # Upsert updates in place rather than duplicating.
    repo.upsert_profile(
        Profile(id="p1", email="a@b.dev", full_name="Ada L.", role="mentor")
    )
    updated = repo.get_profile("p1")
    assert updated is not None
    assert updated.full_name == "Ada L."
    assert updated.role == "mentor"
    assert len(repo.list_profiles()) == 1


def test_upsert_profile_assigns_id_when_blank(repo: SqliteRepo):
    saved = repo.upsert_profile(
        Profile(id="", email="x@y.dev", full_name="No Id", role="student")
    )
    assert saved.id
    assert repo.get_profile(saved.id) is not None


def test_list_profiles_filters_by_role(repo: SqliteRepo):
    seed_demo(repo)
    students = repo.list_profiles(role="student")
    mentors = repo.list_profiles(role="mentor")
    assert len(students) == 1
    assert len(mentors) == 1
    assert students[0].role == "student"
    assert mentors[0].role == "mentor"


def test_seed_demo_row_counts(repo: SqliteRepo):
    counts = seed_demo(repo)
    assert counts["profiles"] == 2
    assert counts["courses"] == 2
    assert counts["lessons"] == 9
    assert counts["progress_rows"] == 9
    assert counts["activity_events"] > 0
    assert counts["study_sets"] == 1
    assert counts["interview_reports"] == 1

    # Courses and lessons are queryable.
    courses = repo.list_courses()
    assert len(courses) == 2
    total_lessons = sum(len(repo.list_lessons(c.id)) for c in courses)
    assert total_lessons == 9


def test_progress_by_course_shape(repo: SqliteRepo):
    seed_demo(repo)
    rows = repo.progress_by_course("student-demo")
    assert len(rows) == 2
    for row in rows:
        assert set(row) == {"course", "completed", "total", "pct"}
        assert row["total"] > 0
        assert 0 <= row["completed"] <= row["total"]
        assert 0.0 <= row["pct"] <= 100.0
    # The Python course is fully completed in the seed.
    python = next(r for r in rows if r["course"] == "Python Foundations")
    assert python["completed"] == python["total"]
    assert python["pct"] == 100.0


def test_activity_timeseries_shape(repo: SqliteRepo):
    seed_demo(repo)
    series = repo.activity_timeseries("student-demo", days=30)
    assert len(series) == 30
    for point in series:
        assert set(point) == {"date", "minutes"}
        assert point["minutes"] >= 0.0
    # Dates are ascending and unique.
    dates = [p["date"] for p in series]
    assert dates == sorted(dates)
    assert len(set(dates)) == 30
    # At least some days have recorded study minutes.
    assert sum(1 for p in series if p["minutes"] > 0) > 0
    assert sum(p["minutes"] for p in series) > 0


def test_study_sets_and_reports_persist(repo: SqliteRepo):
    seed_demo(repo)
    sets = repo.list_study_sets("student-demo")
    reports = repo.list_interview_reports("student-demo")
    assert len(sets) == 1
    assert sets[0].title
    assert len(reports) == 1
    assert reports[0].scores  # non-empty dict round-tripped from JSON
    assert "communication" in reports[0].scores
