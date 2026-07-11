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
    """A single lesson within a course.

    Beyond ordering, a lesson now carries real teaching material: ``content``
    (markdown body shown in Learn), a ``module`` section name used to group a
    course's lessons, and optional ``video_url`` / ``doc_url`` media links.
    """

    id: str
    course_id: str
    title: str
    order_index: int = 0
    content: str = ""
    module: str = ""
    video_url: str = ""
    doc_url: str = ""


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
    """A saved set of study artifacts produced from a source.

    ``content`` holds the full generated artifacts (summary, flashcards, concept
    graph) as a plain dict so a saved set can be reopened after a refresh, not
    just listed by title. Empty for legacy rows saved before content persistence.
    """

    id: str
    owner_id: str
    title: str
    source: str = ""
    created_at: str = ""
    content: dict = field(default_factory=dict)


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
    def create_course(self, course: Course) -> Course: ...
    def list_lessons(self, course_id: str) -> list[Lesson]: ...
    def get_lesson(self, lesson_id: str) -> Optional[Lesson]: ...
    def create_lesson(self, lesson: Lesson) -> Lesson: ...
    def backfill_lesson_content(self, repo: "Repository") -> int: ...

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
    def get_study_set(self, set_id: str) -> Optional[StudySet]: ...

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
        # check_same_thread=False: Streamlit reruns each script on a pooled
        # thread, so a session-cached connection is used across threads.
        self._conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with open(self._schema_path, "r", encoding="utf-8") as handle:
            self._conn.executescript(handle.read())
        self._conn.commit()
        # Best-effort migration for databases created before ``content`` existed.
        # ``CREATE TABLE IF NOT EXISTS`` will not add the column to an old table.
        try:
            self._conn.execute(
                "ALTER TABLE study_sets ADD COLUMN content TEXT NOT NULL DEFAULT '{}'"
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            # Column already present: nothing to do.
            pass
        # Best-effort migration for the richer ``lessons`` columns.
        for _column in ("content", "module", "video_url", "doc_url"):
            try:
                self._conn.execute(
                    f"ALTER TABLE lessons ADD COLUMN {_column} TEXT NOT NULL DEFAULT ''"
                )
                self._conn.commit()
            except sqlite3.OperationalError:
                # Column already present: nothing to do.
                pass

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

    def create_course(self, course: Course) -> Course:
        if not course.id:
            course.id = _new_id()
        self._conn.execute(
            """
            INSERT INTO courses (id, title, description) VALUES (?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                title = excluded.title,
                description = excluded.description
            """,
            (course.id, course.title, course.description),
        )
        self._conn.commit()
        return course

    @staticmethod
    def _row_to_lesson(row: sqlite3.Row) -> Lesson:
        keys = row.keys()
        return Lesson(
            id=row["id"],
            course_id=row["course_id"],
            title=row["title"] or "",
            order_index=int(row["order_index"] or 0),
            content=(row["content"] if "content" in keys else "") or "",
            module=(row["module"] if "module" in keys else "") or "",
            video_url=(row["video_url"] if "video_url" in keys else "") or "",
            doc_url=(row["doc_url"] if "doc_url" in keys else "") or "",
        )

    def list_lessons(self, course_id: str) -> list[Lesson]:
        rows = self._conn.execute(
            "SELECT * FROM lessons WHERE course_id = ? ORDER BY order_index",
            (course_id,),
        ).fetchall()
        return [self._row_to_lesson(r) for r in rows]

    def get_lesson(self, lesson_id: str) -> Optional[Lesson]:
        row = self._conn.execute(
            "SELECT * FROM lessons WHERE id = ?", (lesson_id,)
        ).fetchone()
        return self._row_to_lesson(row) if row is not None else None

    def create_lesson(self, lesson: Lesson) -> Lesson:
        if not lesson.id:
            lesson.id = _new_id()
        if not lesson.order_index:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(order_index), -1) AS m FROM lessons WHERE course_id = ?",
                (lesson.course_id,),
            ).fetchone()
            # COALESCE guarantees a non-null value; do not use ``or`` here since a
            # legitimate max of 0 is falsy and would collapse to -1.
            lesson.order_index = int(row["m"]) + 1
        self._conn.execute(
            """
            INSERT INTO lessons
                (id, course_id, title, order_index, content, module, video_url, doc_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                title = excluded.title,
                order_index = excluded.order_index,
                content = excluded.content,
                module = excluded.module,
                video_url = excluded.video_url,
                doc_url = excluded.doc_url
            """,
            (
                lesson.id,
                lesson.course_id,
                lesson.title,
                lesson.order_index,
                lesson.content,
                lesson.module,
                lesson.video_url,
                lesson.doc_url,
            ),
        )
        self._conn.commit()
        return lesson

    def backfill_lesson_content(self, repo: "Repository") -> int:
        return _backfill_lesson_content(repo)

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
            INSERT INTO study_sets (id, owner_id, title, source, content, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                title = excluded.title,
                source = excluded.source,
                content = excluded.content
            """,
            (
                study_set.id,
                study_set.owner_id,
                study_set.title,
                study_set.source,
                json.dumps(study_set.content or {}),
                study_set.created_at,
            ),
        )
        self._conn.commit()
        return study_set

    def _row_to_study_set(self, row: sqlite3.Row) -> StudySet:
        """Map a ``study_sets`` row to a :class:`StudySet`, decoding content."""

        try:
            content = json.loads(row["content"]) if row["content"] else {}
        except (ValueError, TypeError):
            content = {}
        return StudySet(
            id=row["id"],
            owner_id=row["owner_id"],
            title=row["title"],
            source=row["source"],
            created_at=row["created_at"],
            content=content if isinstance(content, dict) else {},
        )

    def list_study_sets(self, owner_id: str) -> list[StudySet]:
        rows = self._conn.execute(
            "SELECT * FROM study_sets WHERE owner_id = ? ORDER BY created_at DESC",
            (owner_id,),
        ).fetchall()
        return [self._row_to_study_set(r) for r in rows]

    def get_study_set(self, set_id: str) -> Optional[StudySet]:
        row = self._conn.execute(
            "SELECT * FROM study_sets WHERE id = ?",
            (set_id,),
        ).fetchone()
        return self._row_to_study_set(row) if row is not None else None

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

    @staticmethod
    def _uid(value: Optional[str]) -> Optional[str]:
        """Coerce an id/foreign-key to a valid UUID string.

        Postgres id columns are ``uuid``, but callers (notably ``seed_demo``)
        use readable string ids such as ``"student-demo"`` or
        ``"course-python-l0"``. Values that already parse as UUIDs pass through
        unchanged; anything else maps to a deterministic uuid5 so the same
        string always yields the same UUID and foreign keys stay consistent.
        """

        if value is None:
            return None
        try:
            return str(uuid.UUID(str(value)))
        except (ValueError, AttributeError, TypeError):
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"navilearn:{value}"))

    # -- Profiles -------------------------------------------------------- #
    def upsert_profile(self, profile: Profile) -> Profile:
        if not profile.id:
            profile.id = _new_id()
        self._table("profiles").upsert(
            {
                "id": self._uid(profile.id),
                "email": profile.email,
                "full_name": profile.full_name,
                "role": profile.role,
                "mentor_id": self._uid(profile.mentor_id),
            }
        ).execute()
        return profile

    def get_profile(self, profile_id: str) -> Optional[Profile]:
        res = self._table("profiles").select("*").eq("id", self._uid(profile_id)).limit(1).execute()
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

    def create_course(self, course: Course) -> Course:
        if not course.id:
            course.id = _new_id()
        self._table("courses").upsert(
            {
                "id": self._uid(course.id),
                "title": course.title,
                "description": course.description,
            }
        ).execute()
        return course

    @staticmethod
    def _dict_to_lesson(d: dict[str, Any]) -> Lesson:
        return Lesson(
            id=d["id"],
            course_id=d["course_id"],
            title=d.get("title", ""),
            order_index=int(d.get("order_index", 0) or 0),
            content=d.get("content") or "",
            module=d.get("module") or "",
            video_url=d.get("video_url") or "",
            doc_url=d.get("doc_url") or "",
        )

    def list_lessons(self, course_id: str) -> list[Lesson]:
        res = (
            self._table("lessons")
            .select("*")
            .eq("course_id", course_id)
            .order("order_index")
            .execute()
        )
        return [self._dict_to_lesson(d) for d in (res.data or [])]

    def get_lesson(self, lesson_id: str) -> Optional[Lesson]:
        res = (
            self._table("lessons")
            .select("*")
            .eq("id", self._uid(lesson_id))
            .limit(1)
            .execute()
        )
        rows = res.data or []
        return self._dict_to_lesson(rows[0]) if rows else None

    def create_lesson(self, lesson: Lesson) -> Lesson:
        if not lesson.id:
            lesson.id = _new_id()
        course_uid = self._uid(lesson.course_id)
        if not lesson.order_index:
            res = (
                self._table("lessons")
                .select("order_index")
                .eq("course_id", course_uid)
                .order("order_index", desc=True)
                .limit(1)
                .execute()
            )
            rows = res.data or []
            lesson.order_index = (int(rows[0]["order_index"] or 0) + 1) if rows else 0
        self._table("lessons").upsert(
            {
                "id": self._uid(lesson.id),
                "course_id": course_uid,
                "title": lesson.title,
                "order_index": lesson.order_index,
                "content": lesson.content,
                "module": lesson.module,
                "video_url": lesson.video_url,
                "doc_url": lesson.doc_url,
            }
        ).execute()
        return lesson

    def backfill_lesson_content(self, repo: "Repository") -> int:
        return _backfill_lesson_content(repo)

    # -- Activity -------------------------------------------------------- #
    def record_activity(self, event: ActivityEvent) -> ActivityEvent:
        if not event.id:
            event.id = _new_id()
        if not event.created_at:
            event.created_at = _now_iso()
        self._table("activity_events").insert(
            {
                "id": self._uid(event.id),
                "student_id": self._uid(event.student_id),
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
            .eq("student_id", self._uid(student_id))
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
                "id": self._uid(row.id),
                "student_id": self._uid(row.student_id),
                "lesson_id": self._uid(row.lesson_id),
                "status": row.status,
                "time_spent_seconds": row.time_spent_seconds,
                "completed_at": row.completed_at,
            },
            on_conflict="student_id,lesson_id",
        ).execute()
        return row

    def list_progress(self, student_id: str) -> list[ProgressRow]:
        res = (
            self._table("progress").select("*").eq("student_id", self._uid(student_id)).execute()
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
                "id": self._uid(study_set.id),
                "owner_id": self._uid(study_set.owner_id),
                "title": study_set.title,
                "source": study_set.source,
                "content": study_set.content or {},
                "created_at": study_set.created_at,
            }
        ).execute()
        return study_set

    def _dict_to_study_set(self, d: dict[str, Any]) -> StudySet:
        """Map a Supabase ``study_sets`` row dict to a :class:`StudySet`."""

        content = d.get("content") or {}
        return StudySet(
            id=d["id"],
            owner_id=d["owner_id"],
            title=d.get("title", ""),
            source=d.get("source", ""),
            created_at=d.get("created_at", ""),
            content=content if isinstance(content, dict) else {},
        )

    def list_study_sets(self, owner_id: str) -> list[StudySet]:
        res = (
            self._table("study_sets")
            .select("*")
            .eq("owner_id", self._uid(owner_id))
            .order("created_at", desc=True)
            .execute()
        )
        return [self._dict_to_study_set(d) for d in (res.data or [])]

    def get_study_set(self, set_id: str) -> Optional[StudySet]:
        res = (
            self._table("study_sets")
            .select("*")
            .eq("id", self._uid(set_id))
            .limit(1)
            .execute()
        )
        rows = res.data or []
        return self._dict_to_study_set(rows[0]) if rows else None

    # -- Interview reports ----------------------------------------------- #
    def save_interview_report(self, report: InterviewReport) -> InterviewReport:
        if not report.id:
            report.id = _new_id()
        if not report.created_at:
            report.created_at = _now_iso()
        self._table("interview_reports").insert(
            {
                "id": self._uid(report.id),
                "student_id": self._uid(report.student_id),
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
            .eq("student_id", self._uid(student_id))
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
        # Prefer the service-role/secret key (bypasses RLS for server writes);
        # fall back to the anon/publishable key when no secret key is set.
        key = settings.supabase_service_role_key or settings.supabase_anon_key
        return SupabaseRepo(settings.supabase_url, key)
    return SqliteRepo(settings.sqlite_path)


# --------------------------------------------------------------------------- #
# Lesson content helpers
# --------------------------------------------------------------------------- #
# A real, short Python intro video used so the Learn surface can render media.
DEMO_VIDEO_URL = "https://www.youtube.com/watch?v=rfscVS0vtbw"


def _module_for(order_index: int, total: int) -> str:
    """Return a section name grouping a course's lessons into 1-2 modules.

    The first half of a course (by ``order_index``) is "Fundamentals"; the
    remainder is "Applied Practice". Single-lesson courses use one section.
    """

    if total <= 1:
        return "Getting Started"
    half = (total + 1) // 2
    return "Fundamentals" if order_index < half else "Applied Practice"


def _content_stub(course_title: str, lesson_title: str) -> str:
    """Return a short markdown body derived from a lesson and its course."""

    return (
        f"# {lesson_title}\n\n"
        f"Part of **{course_title}**.\n\n"
        f"In this lesson you will explore {lesson_title.lower()} and how it fits "
        f"into {course_title}. Work through the notes below, then mark the lesson "
        f"complete when you are comfortable with the ideas.\n\n"
        "## Key points\n\n"
        f"- Understand what {lesson_title.lower()} means and why it matters.\n"
        "- See a small, concrete example you can reproduce.\n"
        "- Practice by adapting the example to your own case.\n\n"
        "## Try it\n\n"
        f"Write a few lines that apply {lesson_title.lower()} yourself, then "
        "compare against the example above.\n"
    )


def _is_ml_course(course: Course) -> bool:
    """Best-effort detection of the machine-learning demo course by title/id."""

    text = f"{course.title} {course.id}".lower()
    return "machine learning" in text or "-ml" in text or text.endswith(" ml")


def _persist_lesson_update(
    repo: Repository,
    lesson: Lesson,
    content: str,
    module: str,
    video_url: str,
) -> None:
    """Update a lesson's content/module/video in place without reordering.

    ``create_lesson`` reassigns a zero ``order_index``; backfill must preserve
    existing order, so it writes the mutable fields directly per backend.
    """

    if isinstance(repo, SqliteRepo):
        repo._conn.execute(  # noqa: SLF001 - backend-aware maintenance helper
            "UPDATE lessons SET content = ?, module = ?, video_url = ? WHERE id = ?",
            (content, module, video_url, lesson.id),
        )
        repo._conn.commit()  # noqa: SLF001
        return
    if isinstance(repo, SupabaseRepo):
        repo._table("lessons").update(  # noqa: SLF001
            {"content": content, "module": module, "video_url": video_url}
        ).eq("id", repo._uid(lesson.id)).execute()  # noqa: SLF001
        return
    raise TypeError(f"Cannot backfill lessons for repo type {type(repo).__name__}")


def _backfill_lesson_content(repo: Repository) -> int:
    """Give every content-less lesson a markdown stub, a module, and (ML) a video.

    For each course, lessons with empty ``content`` receive a title-derived
    markdown stub; lessons with empty ``module`` are grouped into 1-2 named
    sections by ``order_index``; and the first lesson of the machine-learning
    course gets the demo video when it has none. Returns the number of lessons
    updated. Best-effort: a failure on one lesson is logged and skipped.
    """

    import logging

    log = logging.getLogger(__name__)
    updated = 0
    for course in repo.list_courses():
        lessons = repo.list_lessons(course.id)
        total = len(lessons)
        is_ml = _is_ml_course(course)
        first_order = min((les.order_index for les in lessons), default=0)
        for lesson in lessons:
            content = lesson.content or ""
            module = lesson.module or ""
            video_url = lesson.video_url or ""
            changed = False
            if not content.strip():
                content = _content_stub(course.title, lesson.title)
                changed = True
            if not module.strip():
                module = _module_for(lesson.order_index, total)
                changed = True
            if is_ml and lesson.order_index == first_order and not video_url.strip():
                video_url = DEMO_VIDEO_URL
                changed = True
            if not changed:
                continue
            try:
                _persist_lesson_update(repo, lesson, content, module, video_url)
                updated += 1
            except Exception as exc:  # noqa: BLE001 - best-effort maintenance
                log.warning("backfill_lesson_content skipped %s: %s", lesson.id, exc)
    return updated


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
    repo.upsert_profile(
        Profile(
            id="teacher-demo",
            email="teacher@navilearn.dev",
            full_name="Tara Teacher",
            role="teacher",
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
        # Build real lesson material: each lesson gets a grouping module and a
        # short markdown body; the ML course's first lesson gets a demo video so
        # the Learn surface renders an embedded player.
        total = len(lesson_titles)
        is_ml = "ml" in course_id.split("-")
        lesson_specs: list[dict[str, str]] = []
        for order, lesson_title in enumerate(lesson_titles):
            lesson_specs.append(
                {
                    "title": lesson_title,
                    "module": _module_for(order, total),
                    "content": _content_stub(title, lesson_title),
                    "video_url": DEMO_VIDEO_URL if (is_ml and order == 0) else "",
                    "doc_url": "",
                }
            )
        # Seed a course row directly through SQLite when available; otherwise
        # rely on the migration having created reference data. For the demo we
        # write courses/lessons via the raw connection on SqliteRepo.
        _seed_course(repo, course_id, title, desc, lesson_specs)
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
    lesson_specs: list[dict[str, str]],
) -> None:
    """Insert a course and its lessons with real content, module, and media.

    Courses/lessons are reference data with no dedicated write method on the
    protocol, so we write them through the backend directly: raw SQL for
    SQLite, table upserts for Supabase. Each entry in ``lesson_specs`` carries
    ``title``, ``module``, ``content``, ``video_url``, and ``doc_url``.
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
        for order, spec in enumerate(lesson_specs):
            conn.execute(
                """
                INSERT INTO lessons
                    (id, course_id, title, order_index,
                     content, module, video_url, doc_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    title = excluded.title,
                    order_index = excluded.order_index,
                    content = excluded.content,
                    module = excluded.module,
                    video_url = excluded.video_url,
                    doc_url = excluded.doc_url
                """,
                (
                    f"{course_id}-l{order}",
                    course_id,
                    spec.get("title", ""),
                    order,
                    spec.get("content", ""),
                    spec.get("module", ""),
                    spec.get("video_url", ""),
                    spec.get("doc_url", ""),
                ),
            )
        conn.commit()
        return

    if isinstance(repo, SupabaseRepo):
        repo._table("courses").upsert(  # noqa: SLF001
            {"id": repo._uid(course_id), "title": title, "description": description}
        ).execute()
        rows = [
            {
                "id": repo._uid(f"{course_id}-l{order}"),
                "course_id": repo._uid(course_id),
                "title": spec.get("title", ""),
                "order_index": order,
                "content": spec.get("content", ""),
                "module": spec.get("module", ""),
                "video_url": spec.get("video_url", ""),
                "doc_url": spec.get("doc_url", ""),
            }
            for order, spec in enumerate(lesson_specs)
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
