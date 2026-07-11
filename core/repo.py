"""Swappable data layer for NaviLearn.

The dashboard and features talk to a :class:`Repository` (a ``Protocol``) and
never bind to a concrete database. Two backends implement it:

- :class:`SqliteRepo` (default): stdlib ``sqlite3`` at ``settings.sqlite_path``,
  schema bootstrapped from ``db/schema.sql`` on first use. Zero network.
- :class:`SupabaseRepo`: supabase-py table operations against a Postgres/pgvector
  database. Best-effort thin implementation used when ``db_backend='supabase'``.

Pick a backend with :func:`get_repo`. Entities are plain dataclasses so callers
get typed rows regardless of which backend answered.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Protocol, runtime_checkable

from core.config import Settings, get_settings

_SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "schema.sql"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _new_id() -> str:
    """Return a fresh opaque string id (uuid4 hex-dashed)."""

    return str(uuid.uuid4())


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""

    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 string into an aware UTC datetime, best-effort."""

    try:
        dt = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #
@dataclass
class Profile:
    """A platform user: student, mentor, or teacher."""

    id: str
    email: str
    full_name: str
    role: str = "student"  # student | mentor | teacher
    mentor_id: Optional[str] = None


@dataclass
class Course:
    """A course grouping ordered lessons."""

    id: str
    title: str
    description: str = ""


@dataclass
class Lesson:
    """A single lesson within a course."""

    id: str
    course_id: str
    title: str
    order_index: int = 0


@dataclass
class ProgressRow:
    """A student's progress against one lesson."""

    id: str
    student_id: str
    lesson_id: str
    status: str = "not_started"  # not_started | in_progress | completed
    time_spent_seconds: int = 0
    completed_at: Optional[str] = None


@dataclass
class ActivityEvent:
    """A timestamped learner action feeding the analytics time series."""

    id: str
    student_id: str
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


@dataclass
class StudySet:
    """A saved set of study artifacts produced from a source."""

    id: str
    owner_id: str
    title: str
    source: str = ""
    created_at: str = ""


@dataclass
class InterviewReport:
    """A scored AI-interview result for a student's project."""

    id: str
    student_id: str
    project_title: str
    scores: dict[str, Any] = field(default_factory=dict)
    feedback: str = ""
    created_at: str = ""


# --------------------------------------------------------------------------- #
# Repository protocol
# --------------------------------------------------------------------------- #
@runtime_checkable
class Repository(Protocol):
    """Backend-agnostic contract the dashboard and features depend on."""

    # Profiles.
    def upsert_profile(self, profile: Profile) -> Profile: ...
    def get_profile(self, profile_id: str) -> Optional[Profile]: ...
    def list_profiles(self, role: Optional[str] = None) -> list[Profile]: ...

    # Courses and lessons.
    def list_courses(self) -> list[Course]: ...
    def list_lessons(self, course_id: str) -> list[Lesson]: ...

    # Activity.
    def record_activity(self, event: ActivityEvent) -> ActivityEvent: ...
    def list_activity(
        self, student_id: str, since: Optional[str] = None
    ) -> list[ActivityEvent]: ...

    # Progress.
    def upsert_progress(self, row: ProgressRow) -> ProgressRow: ...
    def list_progress(self, student_id: str) -> list[ProgressRow]: ...
    def progress_by_course(self, student_id: str) -> list[dict[str, Any]]: ...
    def activity_timeseries(
        self, student_id: str, days: int = 30
    ) -> list[dict[str, Any]]: ...

    # Study sets.
    def save_study_set(self, study_set: StudySet) -> StudySet: ...
    def list_study_sets(self, owner_id: str) -> list[StudySet]: ...

    # Interview reports.
    def save_interview_report(self, report: InterviewReport) -> InterviewReport: ...
    def list_interview_reports(self, student_id: str) -> list[InterviewReport]: ...


# --------------------------------------------------------------------------- #
# SQLite backend
# --------------------------------------------------------------------------- #
class SqliteRepo:
    """Default backend: stdlib ``sqlite3``, schema from ``db/schema.sql``."""

    def __init__(self, sqlite_path: str, schema_path: str = _SCHEMA_PATH) -> None:
        self._path = sqlite_path
        self._schema_path = schema_path
        self._conn = sqlite3.connect(sqlite_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with open(self._schema_path, "r", encoding="utf-8") as handle:
            self._conn.executescript(handle.read())
        self._conn.commit()

    # -- Profiles -------------------------------------------------------- #
    def upsert_profile(self, profile: Profile) -> Profile:
        if not profile.id:
            profile.id = _new_id()
        self._conn.execute(
            """
            INSERT INTO profiles (id, email, full_name, role, mentor_id)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                email = excluded.email,
                full_name = excluded.full_name,
                role = excluded.role,
                mentor_id = excluded.mentor_id
            """,
            (
                profile.id,
                profile.email,
                profile.full_name,
                profile.role,
                profile.mentor_id,
            ),
        )
        self._conn.commit()
        return profile

    def get_profile(self, profile_id: str) -> Optional[Profile]:
        row = self._conn.execute(
            "SELECT * FROM profiles WHERE id = ?", (profile_id,)
        ).fetchone()
        return self._row_to_profile(row) if row else None

    def list_profiles(self, role: Optional[str] = None) -> list[Profile]:
        if role:
            rows = self._conn.execute(
                "SELECT * FROM profiles WHERE role = ? ORDER BY full_name", (role,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM profiles ORDER BY full_name"
            ).fetchall()
        return [self._row_to_profile(r) for r in rows]

    # -- Courses and lessons --------------------------------------------- #
    def list_courses(self) -> list[Course]:
        rows = self._conn.execute("SELECT * FROM courses ORDER BY title").fetchall()
        return [Course(id=r["id"], title=r["title"], description=r["description"]) for r in rows]

    def list_lessons(self, course_id: str) -> list[Lesson]:
        rows = self._conn.execute(
            "SELECT * FROM lessons WHERE course_id = ? ORDER BY order_index",
            (course_id,),
        ).fetchall()
        return [
            Lesson(
                id=r["id"],
                course_id=r["course_id"],
                title=r["title"],
                order_index=r["order_index"],
            )
            for r in rows
        ]

    # -- Activity -------------------------------------------------------- #
    def record_activity(self, event: ActivityEvent) -> ActivityEvent:
        if not event.id:
            event.id = _new_id()
        if not event.created_at:
            event.created_at = _now_iso()
        self._conn.execute(
            """
            INSERT INTO activity_events (id, student_id, type, payload, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.student_id,
                event.type,
                json.dumps(event.payload),
                event.created_at,
            ),
        )
        self._conn.commit()
        return event

    def list_activity(
        self, student_id: str, since: Optional[str] = None
    ) -> list[ActivityEvent]:
        if since:
            rows = self._conn.execute(
                """
                SELECT * FROM activity_events
                WHERE student_id = ? AND created_at >= ?
                ORDER BY created_at
                """,
                (student_id, since),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM activity_events WHERE student_id = ? ORDER BY created_at",
                (student_id,),
            ).fetchall()
        return [self._row_to_activity(r) for r in rows]

    # -- Progress -------------------------------------------------------- #
    def upsert_progress(self, row: ProgressRow) -> ProgressRow:
        if not row.id:
            row.id = _new_id()
        self._conn.execute(
            """
            INSERT INTO progress
                (id, student_id, lesson_id, status, time_spent_seconds, completed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (student_id, lesson_id) DO UPDATE SET
                status = excluded.status,
                time_spent_seconds = excluded.time_spent_seconds,
                completed_at = excluded.completed_at
            """,
            (
                row.id,
                row.student_id,
                row.lesson_id,
                row.status,
                row.time_spent_seconds,
                row.completed_at,
            ),
        )
        self._conn.commit()
        return row

    def list_progress(self, student_id: str) -> list[ProgressRow]:
        rows = self._conn.execute(
            "SELECT * FROM progress WHERE student_id = ?", (student_id,)
        ).fetchall()
        return [self._row_to_progress(r) for r in rows]

    def progress_by_course(self, student_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT
                c.title AS course,
                COUNT(l.id) AS total,
                COUNT(
                    CASE WHEN p.status = 'completed' THEN 1 END
                ) AS completed
            FROM courses c
            JOIN lessons l ON l.course_id = c.id
            LEFT JOIN progress p
                ON p.lesson_id = l.id AND p.student_id = ?
            GROUP BY c.id, c.title
            ORDER BY c.title
            """,
            (student_id,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            total = int(r["total"] or 0)
            completed = int(r["completed"] or 0)
            pct = round(100.0 * completed / total, 1) if total else 0.0
            out.append(
                {
                    "course": r["course"],
                    "completed": completed,
                    "total": total,
                    "pct": pct,
                }
            )
        return out

    def activity_timeseries(
        self, student_id: str, days: int = 30
    ) -> list[dict[str, Any]]:
        since = datetime.now(timezone.utc) - timedelta(days=days - 1)
        since_date = since.date()
        events = self.list_activity(student_id, since=since.isoformat())
        minutes_by_date: dict[str, float] = {}
        for event in events:
            day = _parse_iso(event.created_at).date().isoformat()
            seconds = float(event.payload.get("seconds", 0) or 0)
            minutes_by_date[day] = minutes_by_date.get(day, 0.0) + seconds / 60.0
        series: list[dict[str, Any]] = []
        for offset in range(days):
            day = (since_date + timedelta(days=offset)).isoformat()
            series.append(
                {"date": day, "minutes": round(minutes_by_date.get(day, 0.0), 1)}
            )
        return series

    # -- Study sets ------------------------------------------------------ #
    def save_study_set(self, study_set: StudySet) -> StudySet:
        if not study_set.id:
            study_set.id = _new_id()
        if not study_set.created_at:
            study_set.created_at = _now_iso()
        self._conn.execute(
            """
            INSERT INTO study_sets (id, owner_id, title, source, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                title = excluded.title,
                source = excluded.source
            """,
            (
                study_set.id,
                study_set.owner_id,
                study_set.title,
                study_set.source,
                study_set.created_at,
            ),
        )
        self._conn.commit()
        return study_set

    def list_study_sets(self, owner_id: str) -> list[StudySet]:
        rows = self._conn.execute(
            "SELECT * FROM study_sets WHERE owner_id = ? ORDER BY created_at DESC",
            (owner_id,),
        ).fetchall()
        return [
            StudySet(
                id=r["id"],
                owner_id=r["owner_id"],
                title=r["title"],
                source=r["source"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # -- Interview reports ----------------------------------------------- #
    def save_interview_report(self, report: InterviewReport) -> InterviewReport:
        if not report.id:
            report.id = _new_id()
        if not report.created_at:
            report.created_at = _now_iso()
        self._conn.execute(
            """
            INSERT INTO interview_reports
                (id, student_id, project_title, scores, feedback, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                report.id,
                report.student_id,
                report.project_title,
                json.dumps(report.scores),
                report.feedback,
                report.created_at,
            ),
        )
        self._conn.commit()
        return report

    def list_interview_reports(self, student_id: str) -> list[InterviewReport]:
        rows = self._conn.execute(
            "SELECT * FROM interview_reports WHERE student_id = ? ORDER BY created_at DESC",
            (student_id,),
        ).fetchall()
        out: list[InterviewReport] = []
        for r in rows:
            out.append(
                InterviewReport(
                    id=r["id"],
                    student_id=r["student_id"],
                    project_title=r["project_title"],
                    scores=json.loads(r["scores"] or "{}"),
                    feedback=r["feedback"],
                    created_at=r["created_at"],
                )
            )
        return out

    # -- Row mappers ----------------------------------------------------- #
    @staticmethod
    def _row_to_profile(row: sqlite3.Row) -> Profile:
        return Profile(
            id=row["id"],
            email=row["email"] or "",
            full_name=row["full_name"] or "",
            role=row["role"],
            mentor_id=row["mentor_id"],
        )

    @staticmethod
    def _row_to_activity(row: sqlite3.Row) -> ActivityEvent:
        return ActivityEvent(
            id=row["id"],
            student_id=row["student_id"],
            type=row["type"],
            payload=json.loads(row["payload"] or "{}"),
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_progress(row: sqlite3.Row) -> ProgressRow:
        return ProgressRow(
            id=row["id"],
            student_id=row["student_id"],
            lesson_id=row["lesson_id"],
            status=row["status"],
            time_spent_seconds=int(row["time_spent_seconds"] or 0),
            completed_at=row["completed_at"],
        )


# --------------------------------------------------------------------------- #
# Supabase backend (best-effort, thin)
# --------------------------------------------------------------------------- #
class SupabaseRepo:
    """Thin Postgres backend via supabase-py.

    Best-effort: it maps the same operations onto REST table calls. The
    aggregate helpers (``progress_by_course``, ``activity_timeseries``) are
    computed in Python from fetched rows to avoid depending on server-side SQL.
    Requires a reachable Supabase project, so it is not exercised by the
    offline test suite.
    """

    def __init__(self, url: str, service_role_key: str) -> None:
        from supabase import create_client  # local import: optional dependency

        if not url or not service_role_key:
            raise ValueError(
                "SupabaseRepo needs supabase_url and supabase_service_role_key"
            )
        self._client = create_client(url, service_role_key)

    def _table(self, name: str):
        return self._client.table(name)

    # -- Profiles -------------------------------------------------------- #
    def upsert_profile(self, profile: Profile) -> Profile:
        if not profile.id:
            profile.id = _new_id()
        self._table("profiles").upsert(
            {
                "id": profile.id,
                "email": profile.email,
                "full_name": profile.full_name,
                "role": profile.role,
                "mentor_id": profile.mentor_id,
            }
        ).execute()
        return profile

    def get_profile(self, profile_id: str) -> Optional[Profile]:
        res = self._table("profiles").select("*").eq("id", profile_id).limit(1).execute()
        data = res.data or []
        return self._dict_to_profile(data[0]) if data else None

    def list_profiles(self, role: Optional[str] = None) -> list[Profile]:
        query = self._table("profiles").select("*").order("full_name")
        if role:
            query = query.eq("role", role)
        res = query.execute()
        return [self._dict_to_profile(d) for d in (res.data or [])]

    # -- Courses and lessons --------------------------------------------- #
    def list_courses(self) -> list[Course]:
        res = self._table("courses").select("*").order("title").execute()
        return [
            Course(id=d["id"], title=d.get("title", ""), description=d.get("description", ""))
            for d in (res.data or [])
        ]

    def list_lessons(self, course_id: str) -> list[Lesson]:
        res = (
            self._table("lessons")
            .select("*")
            .eq("course_id", course_id)
            .order("order_index")
            .execute()
        )
        return [
            Lesson(
                id=d["id"],
                course_id=d["course_id"],
                title=d.get("title", ""),
                order_index=int(d.get("order_index", 0) or 0),
            )
            for d in (res.data or [])
        ]

    # -- Activity -------------------------------------------------------- #
    def record_activity(self, event: ActivityEvent) -> ActivityEvent:
        if not event.id:
            event.id = _new_id()
        if not event.created_at:
            event.created_at = _now_iso()
        self._table("activity_events").insert(
            {
                "id": event.id,
                "student_id": event.student_id,
                "type": event.type,
                "payload": event.payload,
                "created_at": event.created_at,
            }
        ).execute()
        return event

    def list_activity(
        self, student_id: str, since: Optional[str] = None
    ) -> list[ActivityEvent]:
        query = (
            self._table("activity_events")
            .select("*")
            .eq("student_id", student_id)
            .order("created_at")
        )
        if since:
            query = query.gte("created_at", since)
        res = query.execute()
        return [self._dict_to_activity(d) for d in (res.data or [])]

    # -- Progress -------------------------------------------------------- #
    def upsert_progress(self, row: ProgressRow) -> ProgressRow:
        if not row.id:
            row.id = _new_id()
        self._table("progress").upsert(
            {
                "id": row.id,
                "student_id": row.student_id,
                "lesson_id": row.lesson_id,
                "status": row.status,
                "time_spent_seconds": row.time_spent_seconds,
                "completed_at": row.completed_at,
            },
            on_conflict="student_id,lesson_id",
        ).execute()
        return row

    def list_progress(self, student_id: str) -> list[ProgressRow]:
        res = (
            self._table("progress").select("*").eq("student_id", student_id).execute()
        )
        return [self._dict_to_progress(d) for d in (res.data or [])]

    def progress_by_course(self, student_id: str) -> list[dict[str, Any]]:
        courses = self.list_courses()
        completed = {
            p.lesson_id
            for p in self.list_progress(student_id)
            if p.status == "completed"
        }
        out: list[dict[str, Any]] = []
        for course in courses:
            lessons = self.list_lessons(course.id)
            total = len(lessons)
            done = sum(1 for lesson in lessons if lesson.id in completed)
            pct = round(100.0 * done / total, 1) if total else 0.0
            out.append(
                {"course": course.title, "completed": done, "total": total, "pct": pct}
            )
        return out

    def activity_timeseries(
        self, student_id: str, days: int = 30
    ) -> list[dict[str, Any]]:
        since = datetime.now(timezone.utc) - timedelta(days=days - 1)
        since_date = since.date()
        events = self.list_activity(student_id, since=since.isoformat())
        minutes_by_date: dict[str, float] = {}
        for event in events:
            day = _parse_iso(event.created_at).date().isoformat()
            seconds = float(event.payload.get("seconds", 0) or 0)
            minutes_by_date[day] = minutes_by_date.get(day, 0.0) + seconds / 60.0
        series: list[dict[str, Any]] = []
        for offset in range(days):
            day = (since_date + timedelta(days=offset)).isoformat()
            series.append(
                {"date": day, "minutes": round(minutes_by_date.get(day, 0.0), 1)}
            )
        return series

    # -- Study sets ------------------------------------------------------ #
    def save_study_set(self, study_set: StudySet) -> StudySet:
        if not study_set.id:
            study_set.id = _new_id()
        if not study_set.created_at:
            study_set.created_at = _now_iso()
        self._table("study_sets").upsert(
            {
                "id": study_set.id,
                "owner_id": study_set.owner_id,
                "title": study_set.title,
                "source": study_set.source,
                "created_at": study_set.created_at,
            }
        ).execute()
        return study_set

    def list_study_sets(self, owner_id: str) -> list[StudySet]:
        res = (
            self._table("study_sets")
            .select("*")
            .eq("owner_id", owner_id)
            .order("created_at", desc=True)
            .execute()
        )
        return [
            StudySet(
                id=d["id"],
                owner_id=d["owner_id"],
                title=d.get("title", ""),
                source=d.get("source", ""),
                created_at=d.get("created_at", ""),
            )
            for d in (res.data or [])
        ]

    # -- Interview reports ----------------------------------------------- #
    def save_interview_report(self, report: InterviewReport) -> InterviewReport:
        if not report.id:
            report.id = _new_id()
        if not report.created_at:
            report.created_at = _now_iso()
        self._table("interview_reports").insert(
            {
                "id": report.id,
                "student_id": report.student_id,
                "project_title": report.project_title,
                "scores": report.scores,
                "feedback": report.feedback,
                "created_at": report.created_at,
            }
        ).execute()
        return report

    def list_interview_reports(self, student_id: str) -> list[InterviewReport]:
        res = (
            self._table("interview_reports")
            .select("*")
            .eq("student_id", student_id)
            .order("created_at", desc=True)
            .execute()
        )
        out: list[InterviewReport] = []
        for d in res.data or []:
            out.append(
                InterviewReport(
                    id=d["id"],
                    student_id=d["student_id"],
                    project_title=d.get("project_title", ""),
                    scores=d.get("scores") or {},
                    feedback=d.get("feedback", ""),
                    created_at=d.get("created_at", ""),
                )
            )
        return out

    # -- Dict mappers ---------------------------------------------------- #
    @staticmethod
    def _dict_to_profile(d: dict[str, Any]) -> Profile:
        return Profile(
            id=d["id"],
            email=d.get("email") or "",
            full_name=d.get("full_name") or "",
            role=d.get("role", "student"),
            mentor_id=d.get("mentor_id"),
        )

    @staticmethod
    def _dict_to_activity(d: dict[str, Any]) -> ActivityEvent:
        payload = d.get("payload") or {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = {}
        return ActivityEvent(
            id=d["id"],
            student_id=d["student_id"],
            type=d.get("type", ""),
            payload=payload,
            created_at=d.get("created_at", ""),
        )

    @staticmethod
    def _dict_to_progress(d: dict[str, Any]) -> ProgressRow:
        return ProgressRow(
            id=d["id"],
            student_id=d["student_id"],
            lesson_id=d["lesson_id"],
            status=d.get("status", "not_started"),
            time_spent_seconds=int(d.get("time_spent_seconds", 0) or 0),
            completed_at=d.get("completed_at"),
        )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def get_repo(settings: Optional[Settings] = None) -> Repository:
    """Return a :class:`Repository` chosen by ``settings.db_backend``.

    ``"sqlite"`` (default) returns a :class:`SqliteRepo`; ``"supabase"`` returns
    a :class:`SupabaseRepo`. Any unknown value falls back to SQLite.
    """

    if settings is None:
        settings = get_settings()
    backend = (settings.db_backend or "sqlite").strip().lower()
    if backend == "supabase":
        return SupabaseRepo(
            settings.supabase_url, settings.supabase_service_role_key
        )
    return SqliteRepo(settings.sqlite_path)


# --------------------------------------------------------------------------- #
# Demo seed
# --------------------------------------------------------------------------- #
def seed_demo(repo: Repository) -> dict[str, int]:
    """Populate ``repo`` with a small but realistic demo dataset.

    Inserts one mentor and one student, two courses with a handful of lessons
    each, roughly 30 days of activity events, and matching progress rows so a
    dashboard has real donut (progress-by-course) and time-series data. Returns
    a dict of row counts for verification.
    """

    import random

    rng = random.Random(42)

    mentor = repo.upsert_profile(
        Profile(
            id="mentor-demo",
            email="mentor@navilearn.dev",
            full_name="Maya Mentor",
            role="mentor",
        )
    )
    student = repo.upsert_profile(
        Profile(
            id="student-demo",
            email="student@navilearn.dev",
            full_name="Sam Student",
            role="student",
            mentor_id=mentor.id,
        )
    )

    # Courses and lessons. Course ids fixed so re-seeding is idempotent.
    course_specs = [
        (
            "course-python",
            "Python Foundations",
            "Variables, control flow, functions, and data structures.",
            [
                "Variables and Types",
                "Control Flow",
                "Functions",
                "Collections",
                "Modules and Packages",
            ],
        ),
        (
            "course-ml",
            "Intro to Machine Learning",
            "Supervised learning, evaluation, and a first model.",
            [
                "What is ML",
                "Data and Features",
                "Train and Test Split",
                "Your First Model",
            ],
        ),
    ]

    lessons_created = 0
    all_lessons: list[tuple[str, str]] = []  # (course_id, lesson_id)
    for course_id, title, desc, lesson_titles in course_specs:
        # Seed a course row directly through SQLite when available; otherwise
        # rely on the migration having created reference data. For the demo we
        # write courses/lessons via the raw connection on SqliteRepo.
        _seed_course(repo, course_id, title, desc, lesson_titles)
        for order, _lesson_title in enumerate(lesson_titles):
            lesson_id = f"{course_id}-l{order}"
            all_lessons.append((course_id, lesson_id))
            lessons_created += 1

    # Progress: complete the first course fully, partway through the second.
    progress_rows = 0
    python_lessons = [lid for cid, lid in all_lessons if cid == "course-python"]
    ml_lessons = [lid for cid, lid in all_lessons if cid == "course-ml"]
    for lid in python_lessons:
        repo.upsert_progress(
            ProgressRow(
                id="",
                student_id=student.id,
                lesson_id=lid,
                status="completed",
                time_spent_seconds=rng.randint(600, 1800),
                completed_at=_now_iso(),
            )
        )
        progress_rows += 1
    for index, lid in enumerate(ml_lessons):
        status = "completed" if index < 2 else ("in_progress" if index == 2 else "not_started")
        repo.upsert_progress(
            ProgressRow(
                id="",
                student_id=student.id,
                lesson_id=lid,
                status=status,
                time_spent_seconds=rng.randint(300, 1500) if status != "not_started" else 0,
                completed_at=_now_iso() if status == "completed" else None,
            )
        )
        progress_rows += 1

    # ~30 days of activity. Most days have a study session (some rest days).
    activity_rows = 0
    now = datetime.now(timezone.utc)
    event_types = ["lesson_view", "flashcard_review", "quiz_attempt", "interview_practice"]
    for day_offset in range(30):
        if rng.random() < 0.2:
            continue  # occasional rest day keeps the series realistic
        day = now - timedelta(days=day_offset)
        sessions = rng.randint(1, 3)
        for _ in range(sessions):
            minutes = rng.randint(5, 45)
            stamp = day.replace(
                hour=rng.randint(8, 21), minute=rng.randint(0, 59), second=0, microsecond=0
            )
            repo.record_activity(
                ActivityEvent(
                    id="",
                    student_id=student.id,
                    type=rng.choice(event_types),
                    payload={"seconds": minutes * 60, "lesson_id": rng.choice(python_lessons + ml_lessons)},
                    created_at=stamp.isoformat(),
                )
            )
            activity_rows += 1

    # A study set and an interview report so those surfaces have data too.
    repo.save_study_set(
        StudySet(
            id="",
            owner_id=student.id,
            title="Python Foundations: flashcards",
            source="course-python",
        )
    )
    repo.save_interview_report(
        InterviewReport(
            id="",
            student_id=student.id,
            project_title="Todo CLI in Python",
            scores={"communication": 7, "technical_depth": 6, "problem_solving": 8},
            feedback="Solid structure. Practice articulating trade-offs out loud.",
        )
    )

    return {
        "profiles": 2,
        "courses": len(course_specs),
        "lessons": lessons_created,
        "progress_rows": progress_rows,
        "activity_events": activity_rows,
        "study_sets": 1,
        "interview_reports": 1,
    }


def _seed_course(
    repo: Repository,
    course_id: str,
    title: str,
    description: str,
    lesson_titles: list[str],
) -> None:
    """Insert a course and its lessons.

    Courses/lessons are reference data with no dedicated write method on the
    protocol, so we write them through the backend directly: raw SQL for
    SQLite, table upserts for Supabase.
    """

    if isinstance(repo, SqliteRepo):
        conn = repo._conn  # noqa: SLF001 - seed helper is backend-aware by design
        conn.execute(
            """
            INSERT INTO courses (id, title, description) VALUES (?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET title = excluded.title,
                                           description = excluded.description
            """,
            (course_id, title, description),
        )
        for order, lesson_title in enumerate(lesson_titles):
            conn.execute(
                """
                INSERT INTO lessons (id, course_id, title, order_index)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET title = excluded.title,
                                               order_index = excluded.order_index
                """,
                (f"{course_id}-l{order}", course_id, lesson_title, order),
            )
        conn.commit()
        return

    if isinstance(repo, SupabaseRepo):
        repo._table("courses").upsert(  # noqa: SLF001
            {"id": course_id, "title": title, "description": description}
        ).execute()
        rows = [
            {
                "id": f"{course_id}-l{order}",
                "course_id": course_id,
                "title": lesson_title,
                "order_index": order,
            }
            for order, lesson_title in enumerate(lesson_titles)
        ]
        repo._table("lessons").upsert(rows).execute()  # noqa: SLF001
        return

    raise TypeError(f"Cannot seed courses for repo type {type(repo).__name__}")


if __name__ == "__main__":
    _settings = get_settings()
    _repo = get_repo(_settings)
    _counts = seed_demo(_repo)
    print("Seeded NaviLearn demo data:")
    for _key, _value in _counts.items():
        print(f"  {_key}: {_value}")
    _series = _repo.activity_timeseries("student-demo", days=30)
    _active = sum(1 for point in _series if point["minutes"] > 0)
    print(f"activity_timeseries: {len(_series)} days, {_active} active")
    for _row in _repo.progress_by_course("student-demo"):
        print(f"  {_row['course']}: {_row['completed']}/{_row['total']} ({_row['pct']}%)")
