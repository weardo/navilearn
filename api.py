"""NaviLearn REST API.

A FastAPI application that exposes the same dashboard data the Streamlit UI
renders, as a real HTTP/JSON API. It is a thin consumer of :class:`core.repo`:
every route shapes repository output into a typed Pydantic response and holds no
business logic of its own.

The app builds one shared :class:`Repository` at import time via
``get_repo(get_settings())`` (the ``.env`` backend, Supabase by default). All
routes read through that single instance.

Run it with::

    .venv/bin/uvicorn api:app --reload --port 8000

FastAPI auto-serves interactive API documentation (Swagger UI) at ``/docs`` and
ReDoc at ``/redoc``; the OpenAPI schema is at ``/openapi.json``. That Swagger
page IS the API documentation for this service.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import uuid
from functools import lru_cache
from statistics import mean
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from core.config import get_settings
from core.exporters import activity_csv, progress_csv
from core.repo import Repository, get_repo

_LOG = logging.getLogger(__name__)

# One shared repository for the whole app, chosen by the .env backend.
_REPO: Repository = get_repo(get_settings())


def get_repository() -> Repository:
    """Return the process-shared :class:`Repository`.

    Exposed as a plain accessor (rather than inlining ``_REPO``) so tests can
    monkeypatch the module global and routes stay backend-agnostic.
    """

    return _REPO


app = FastAPI(
    title="NaviLearn API",
    version="1.0.0",
    description=(
        "REST access to NaviLearn dashboard data: profiles, courses, lessons, "
        "progress, activity time series, and interview reports. Interactive "
        "docs live at /docs (Swagger) and /redoc."
    ),
)


# --------------------------------------------------------------------------- #
# Response / request models
# --------------------------------------------------------------------------- #
class HealthOut(BaseModel):
    """Liveness payload."""

    status: str = "ok"
    backend: str


class ProfileOut(BaseModel):
    """A platform user."""

    id: str
    email: str
    full_name: str
    role: str
    mentor_id: Optional[str] = None


class LoginIn(BaseModel):
    """Sign-in / register request body.

    Authentication is by ``email`` + ``password`` only. The client can never
    request a ``role``: an existing account keeps its stored role, and a
    brand-new sign-in is always registered as a ``student``. ``full_name`` is
    used only when registering a new account.
    """

    email: str
    password: str
    full_name: str = ""


class LoginOut(BaseModel):
    """Successful login result: a signed bearer token plus the profile."""

    token: str
    profile: ProfileOut


class CourseOut(BaseModel):
    """A course."""

    id: str
    title: str
    description: str = ""


class LessonOut(BaseModel):
    """A lesson within a course, including its teaching material."""

    id: str
    course_id: str
    title: str
    order_index: int = 0
    content: str = ""
    module: str = ""
    video_url: str = ""
    doc_url: str = ""


class ProgressIn(BaseModel):
    """Request body to record a student's progress against one lesson."""

    lesson_id: str
    status: str = "in_progress"
    time_spent_seconds: int = 0


class ProgressOut(BaseModel):
    """A stored progress row for one student/lesson pair."""

    id: str
    student_id: str
    lesson_id: str
    status: str
    time_spent_seconds: int
    completed_at: Optional[str] = None


class ActivityIn(BaseModel):
    """Request body to record a raw learner activity event."""

    type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ProgressCourseOut(BaseModel):
    """Per-course completion summary."""

    course: str
    completed: int
    total: int
    pct: float


class TimeseriesPointOut(BaseModel):
    """One day of study activity."""

    date: str
    minutes: float


class ActivityOut(BaseModel):
    """A raw activity event."""

    id: str
    student_id: str
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""


class InterviewOut(BaseModel):
    """A scored interview report."""

    id: str
    student_id: str
    project_title: str
    scores: dict[str, Any] = Field(default_factory=dict)
    feedback: str = ""
    created_at: str = ""


class SummaryOut(BaseModel):
    """Headline metrics for a student dashboard."""

    student_id: str
    full_name: str
    lessons_completed: int
    hours_spent: float
    courses_in_progress: int
    latest_interview_overall: Optional[float] = None
    latest_interview_project: Optional[str] = None


# --------------------------------------------------------------------------- #
# Authentication: password hashing, signed tokens, and the Supabase admin client
# --------------------------------------------------------------------------- #
# Where a generated signing secret is persisted when neither API_SECRET nor
# settings.session_secret is set. Gitignored alongside .session_secret.
_API_SECRET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".api_secret"
)

# PBKDF2 work factor. High enough to be meaningful, cheap enough for a demo.
_PBKDF2_ITERATIONS = 200_000


def _hash_password(pw: str, iterations: int = _PBKDF2_ITERATIONS) -> str:
    """Hash a password with PBKDF2-HMAC-SHA256 and a fresh random salt.

    Returns a self-describing string ``pbkdf2_sha256$<iters>$<b64salt>$<b64dk>``
    so :func:`_verify_password` can recover the parameters without any external
    state.
    """

    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", (pw or "").encode("utf-8"), salt, iterations)
    return "pbkdf2_sha256${}${}${}".format(
        iterations,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(dk).decode("ascii"),
    )


def _verify_password(pw: str, stored: Optional[str]) -> bool:
    """Return ``True`` iff ``pw`` matches the ``stored`` PBKDF2 hash.

    Uses :func:`hmac.compare_digest` for a constant-time comparison. Any
    malformed or empty ``stored`` value verifies as ``False``.
    """

    if not stored:
        return False
    try:
        algo, iters_s, salt_b64, dk_b64 = stored.split("$")
    except (ValueError, AttributeError):
        return False
    if algo != "pbkdf2_sha256":
        return False
    try:
        iterations = int(iters_s)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(dk_b64)
    except (ValueError, TypeError):
        return False
    dk = hashlib.pbkdf2_hmac("sha256", (pw or "").encode("utf-8"), salt, iterations)
    return hmac.compare_digest(dk, expected)


@lru_cache(maxsize=1)
def _api_secret() -> bytes:
    """Return the HMAC signing secret for bearer tokens, stable per process.

    Resolution order: the ``API_SECRET`` env var, then
    ``settings.session_secret``, then a value generated once and persisted to a
    gitignored ``.api_secret`` file so tokens survive restarts. If the file
    cannot be read or written, a fresh ephemeral secret is used (tokens then do
    not survive a restart, which is acceptable degradation).
    """

    env = os.environ.get("API_SECRET")
    if env:
        return env.encode("utf-8")
    try:
        configured = get_settings().session_secret
    except Exception as exc:  # noqa: BLE001 - settings must not break signing.
        _LOG.warning("settings unavailable for API secret: %s", exc)
        configured = ""
    if configured:
        return configured.encode("utf-8")
    try:
        if os.path.exists(_API_SECRET_PATH):
            with open(_API_SECRET_PATH, "r", encoding="utf-8") as handle:
                existing = handle.read().strip()
            if existing:
                return existing.encode("utf-8")
        generated = secrets.token_urlsafe(32)
        with open(_API_SECRET_PATH, "w", encoding="utf-8") as handle:
            handle.write(generated)
        try:
            os.chmod(_API_SECRET_PATH, 0o600)
        except OSError:
            pass
        return generated.encode("utf-8")
    except OSError as exc:
        _LOG.warning("could not persist API secret, using ephemeral: %s", exc)
        return secrets.token_bytes(32)


def _b64url_encode(raw: bytes) -> str:
    """Return base64url text for ``raw`` with padding stripped."""

    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    """Decode base64url ``text`` that may have had its padding stripped."""

    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def _make_token(sub: str, role: str, ttl: int = 86_400) -> str:
    """Return a signed bearer token binding ``sub``/``role`` for ``ttl`` seconds.

    The token is ``<b64url(payload)>.<b64url(hmac_sha256(payload))>`` where the
    payload is a compact JSON object ``{"sub", "role", "exp"}``. It carries no
    secret material and is verified (not decrypted) by :func:`_read_token`.
    """

    payload = {"sub": sub, "role": role, "exp": int(time.time()) + int(ttl)}
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    body = _b64url_encode(raw)
    signature = hmac.new(_api_secret(), body.encode("ascii"), hashlib.sha256).digest()
    return body + "." + _b64url_encode(signature)


def _read_token(token: str) -> Optional[dict]:
    """Validate a bearer token and return its payload, or ``None``.

    Returns ``None`` when the token is malformed, the signature does not match
    (constant-time check), or it has expired.
    """

    if not token or "." not in token:
        return None
    body, _, signature_b64 = token.partition(".")
    expected = hmac.new(_api_secret(), body.encode("ascii"), hashlib.sha256).digest()
    try:
        provided = _b64url_decode(signature_b64)
    except (ValueError, TypeError):
        return None
    if not hmac.compare_digest(expected, provided):
        return None
    try:
        payload = json.loads(_b64url_decode(body).decode("utf-8"))
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    exp = payload.get("exp")
    if not isinstance(exp, (int, float)) or time.time() > exp:
        return None
    return payload


@lru_cache(maxsize=1)
def _admin_client():
    """Return a cached service-role Supabase client, or ``None``.

    Built from ``settings.supabase_url`` + ``settings.supabase_service_role_key``
    exactly like :mod:`core.classroom`. Used to read and write the
    ``profiles.password_hash`` column, which the :class:`~core.repo.Profile`
    dataclass (and therefore ``repo.upsert_profile``) does not carry. Returns
    ``None`` when Supabase is not configured so callers can degrade.
    """

    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        _LOG.warning("Auth admin client disabled: Supabase not configured.")
        return None
    try:
        from supabase import create_client  # local import: optional dependency

        return create_client(
            settings.supabase_url, settings.supabase_service_role_key
        )
    except Exception as exc:  # noqa: BLE001 - a bad client must not crash the API.
        _LOG.warning("Auth admin Supabase client unavailable: %s", exc)
        return None


def _coerce_uid(value: Optional[str]) -> Optional[str]:
    """Coerce a readable id to the same UUID :class:`SupabaseRepo` would store.

    Mirrors ``SupabaseRepo._uid``: values that already parse as UUIDs pass
    through; anything else maps to a deterministic uuid5 so a readable id such as
    ``"student-demo"`` resolves to the same row the repository wrote.
    """

    if value is None:
        return None
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"navilearn:{value}"))


def _profile_row_by_email(email: str) -> Optional[dict[str, Any]]:
    """Return the raw ``profiles`` row for ``email`` (case-insensitive), or None.

    Reads through the service-role client so ``password_hash`` is included.
    """

    client = _admin_client()
    if client is None:
        return None
    try:
        res = (
            client.table("profiles")
            .select("*")
            .ilike("email", (email or "").strip())
            .limit(1)
            .execute()
        )
        rows = res.data or []
        return rows[0] if rows else None
    except Exception as exc:  # noqa: BLE001 - reads degrade to None.
        _LOG.warning("profile lookup by email failed: %s", exc)
        return None


def _set_password_hash(email: str, password_hash: str) -> bool:
    """Persist ``password_hash`` on the profile with ``email`` (best-effort)."""

    client = _admin_client()
    if client is None:
        return False
    try:
        client.table("profiles").update({"password_hash": password_hash}).ilike(
            "email", (email or "").strip()
        ).execute()
        return True
    except Exception as exc:  # noqa: BLE001 - writes degrade to no-op.
        _LOG.warning("set password_hash failed: %s", exc)
        return False


def _row_to_profile_out(row: dict[str, Any]) -> ProfileOut:
    """Shape a raw ``profiles`` row (never the ``password_hash``) into ProfileOut."""

    mentor_id = row.get("mentor_id")
    return ProfileOut(
        id=str(row.get("id") or ""),
        email=row.get("email") or "",
        full_name=row.get("full_name") or "",
        role=row.get("role") or "student",
        mentor_id=str(mentor_id) if mentor_id else None,
    )


def seed_demo_passwords() -> dict[str, bool]:
    """Set the demo accounts' API password to ``navilearn`` (idempotent).

    Targets ``student@``, ``mentor@``, and ``teacher@navilearn.dev``. An account
    whose stored hash already verifies against ``navilearn`` is left untouched so
    re-running does not thrash the salt. Returns ``{email: has_password}``.
    """

    demo_emails = (
        "student@navilearn.dev",
        "mentor@navilearn.dev",
        "teacher@navilearn.dev",
    )
    results: dict[str, bool] = {}
    for email in demo_emails:
        row = _profile_row_by_email(email)
        if row is None:
            results[email] = False
            continue
        stored = row.get("password_hash")
        if stored and _verify_password("navilearn", stored):
            results[email] = True
            continue
        results[email] = _set_password_hash(email, _hash_password("navilearn"))
    return results


# --------------------------------------------------------------------------- #
# FastAPI auth dependencies
# --------------------------------------------------------------------------- #
def require_caller(authorization: str = Header(default="")) -> dict:
    """Resolve the caller from an ``Authorization: Bearer <token>`` header.

    Raises ``401`` when the header is missing/malformed or the token fails
    signature or expiry validation. Returns the token payload (``sub``, ``role``,
    ``exp``).
    """

    prefix = "Bearer "
    if not authorization or not authorization.startswith(prefix):
        raise HTTPException(
            status_code=401, detail="Missing or malformed Authorization header"
        )
    payload = _read_token(authorization[len(prefix):].strip())
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload


def require_student_access(
    student_id: str, caller: dict = Depends(require_caller)
) -> dict:
    """Authorize access to ``/students/{student_id}/*``.

    A caller may access a student record only when it is their own
    (``caller['sub'] == student_id``) or the caller is a ``mentor``/``teacher``.
    Raises ``403`` otherwise.
    """

    if caller.get("sub") == student_id or caller.get("role") in ("mentor", "teacher"):
        return caller
    raise HTTPException(status_code=403, detail="Forbidden: not your student record")


def require_staff(caller: dict = Depends(require_caller)) -> dict:
    """Authorize staff-only routes (the roster). Raises ``403`` for students."""

    if caller.get("role") in ("mentor", "teacher"):
        return caller
    raise HTTPException(status_code=403, detail="Forbidden: mentor or teacher only")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _require_student(student_id: str) -> None:
    """Raise 404 if ``student_id`` has no profile."""

    repo = get_repository()
    if repo.get_profile(student_id) is None:
        raise HTTPException(status_code=404, detail=f"No profile {student_id!r}")


def _latest_interview_overall(reports: list) -> Optional[float]:
    """Return the latest interview's overall score, or ``None``.

    Uses an explicit ``overall`` score if present, else the mean of numeric
    sub-scores in the most recent report.
    """

    if not reports:
        return None
    scores = reports[0].scores or {}
    overall = scores.get("overall")
    if isinstance(overall, (int, float)):
        return float(overall)
    numeric = [float(v) for v in scores.values() if isinstance(v, (int, float))]
    return round(mean(numeric), 1) if numeric else None


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthOut, tags=["system"])
def health() -> HealthOut:
    """Liveness probe: reports status and the active data backend."""

    return HealthOut(status="ok", backend=type(get_repository()).__name__)


@app.post("/auth/login", response_model=LoginOut, tags=["auth"])
def login(body: LoginIn) -> LoginOut:
    """Authenticate by email + password and return a signed bearer token.

    Three cases, resolved against the ``profiles`` table (read through the
    service-role client so ``password_hash`` is visible):

    - The email exists AND has a password set: the password is verified
      (``401`` on mismatch) and the caller is logged in with the stored role.
    - The email exists but has NO password set: ``401`` (the account must have an
      API password provisioned out of band before it can sign in here).
    - The email does not exist: a NEW profile is registered as a ``student`` with
      the given password, then logged in.

    The client-supplied role is never trusted: a new account is always a
    ``student``, and an existing account keeps its stored role. On success the
    response carries a ``token`` (send it as ``Authorization: Bearer <token>``)
    and the ``profile``.
    """

    from core.repo import Profile

    email = (body.email or "").strip()
    if not email:
        raise HTTPException(status_code=422, detail="email is required")
    password = body.password or ""
    if not password:
        raise HTTPException(status_code=422, detail="password is required")

    row = _profile_row_by_email(email)
    if row is not None:
        stored = row.get("password_hash")
        if not stored:
            raise HTTPException(
                status_code=401,
                detail="This account has no API password set. Provision one first.",
            )
        if not _verify_password(password, stored):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        profile = _row_to_profile_out(row)
        token = _make_token(profile.id, profile.role)
        return LoginOut(token=token, profile=profile)

    # New account: register as a student and set the password hash directly (the
    # Profile dataclass carries no password field, so upsert_profile cannot).
    full_name = (body.full_name or "").strip() or email
    repo = get_repository()
    saved = repo.upsert_profile(
        Profile(id="", email=email, full_name=full_name, role="student", mentor_id=None)
    )
    _set_password_hash(email, _hash_password(password))

    created = _profile_row_by_email(email)
    profile = (
        _row_to_profile_out(created)
        if created is not None
        else ProfileOut(
            id=str(_coerce_uid(saved.id) or saved.id),
            email=email,
            full_name=full_name,
            role="student",
            mentor_id=None,
        )
    )
    token = _make_token(profile.id, profile.role)
    return LoginOut(token=token, profile=profile)


@app.get("/students", response_model=list[ProfileOut], tags=["students"])
def list_students(_: dict = Depends(require_staff)) -> list[ProfileOut]:
    """List all student profiles. Mentor/teacher only (``403`` for students)."""

    repo = get_repository()
    return [ProfileOut(**vars(p)) for p in repo.list_profiles("student")]


@app.get("/students/{student_id}/summary", response_model=SummaryOut, tags=["students"])
def student_summary(
    student_id: str, _: dict = Depends(require_student_access)
) -> SummaryOut:
    """Return headline dashboard metrics for one student."""

    repo = get_repository()
    profile = repo.get_profile(student_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"No profile {student_id!r}")

    progress = repo.list_progress(student_id)
    lessons_completed = sum(1 for row in progress if row.status == "completed")
    hours = round(sum(row.time_spent_seconds for row in progress) / 3600.0, 1)

    by_course = repo.progress_by_course(student_id)
    in_progress = sum(1 for c in by_course if 0.0 < c["pct"] < 100.0)

    reports = repo.list_interview_reports(student_id)
    overall = _latest_interview_overall(reports)
    project = reports[0].project_title if reports else None

    return SummaryOut(
        student_id=student_id,
        full_name=profile.full_name,
        lessons_completed=lessons_completed,
        hours_spent=hours,
        courses_in_progress=in_progress,
        latest_interview_overall=overall,
        latest_interview_project=project,
    )


@app.get(
    "/students/{student_id}/progress",
    response_model=list[ProgressCourseOut],
    tags=["students"],
)
def student_progress(
    student_id: str, _: dict = Depends(require_student_access)
) -> list[ProgressCourseOut]:
    """Return per-course completion for one student."""

    _require_student(student_id)
    repo = get_repository()
    return [ProgressCourseOut(**row) for row in repo.progress_by_course(student_id)]


@app.get(
    "/students/{student_id}/timeseries",
    response_model=list[TimeseriesPointOut],
    tags=["students"],
)
def student_timeseries(
    student_id: str,
    days: int = Query(default=30, ge=1, le=365),
    _: dict = Depends(require_student_access),
) -> list[TimeseriesPointOut]:
    """Return the daily study-minutes time series for one student."""

    _require_student(student_id)
    repo = get_repository()
    return [
        TimeseriesPointOut(**point)
        for point in repo.activity_timeseries(student_id, days=days)
    ]


@app.get(
    "/students/{student_id}/activity",
    response_model=list[ActivityOut],
    tags=["students"],
)
def student_activity(
    student_id: str, _: dict = Depends(require_student_access)
) -> list[ActivityOut]:
    """Return the raw activity events for one student."""

    _require_student(student_id)
    repo = get_repository()
    return [ActivityOut(**vars(e)) for e in repo.list_activity(student_id)]


@app.get(
    "/students/{student_id}/interviews",
    response_model=list[InterviewOut],
    tags=["students"],
)
def student_interviews(
    student_id: str, _: dict = Depends(require_student_access)
) -> list[InterviewOut]:
    """Return interview reports for one student, newest first."""

    _require_student(student_id)
    repo = get_repository()
    return [InterviewOut(**vars(r)) for r in repo.list_interview_reports(student_id)]


@app.get("/courses", response_model=list[CourseOut], tags=["courses"])
def list_courses() -> list[CourseOut]:
    """List all courses."""

    repo = get_repository()
    return [CourseOut(**vars(c)) for c in repo.list_courses()]


@app.get("/courses/{course_id}/lessons", response_model=list[LessonOut], tags=["courses"])
def course_lessons(course_id: str) -> list[LessonOut]:
    """List the lessons of one course, ordered, with full teaching material."""

    repo = get_repository()
    lessons = repo.list_lessons(course_id)
    if not lessons and all(c.id != course_id for c in repo.list_courses()):
        raise HTTPException(status_code=404, detail=f"No course {course_id!r}")
    return [LessonOut(**vars(lesson)) for lesson in lessons]


@app.get(
    "/courses/{course_id}/lessons/{lesson_id}",
    response_model=LessonOut,
    tags=["courses"],
)
def course_lesson(course_id: str, lesson_id: str) -> LessonOut:
    """Return one lesson's full details, or 404 if it does not exist.

    The ``course_id`` scopes the lookup: a lesson that exists but belongs to a
    different course is treated as not found for this path.
    """

    repo = get_repository()
    lesson = repo.get_lesson(lesson_id)
    if lesson is None or lesson.course_id != course_id:
        raise HTTPException(
            status_code=404,
            detail=f"No lesson {lesson_id!r} in course {course_id!r}",
        )
    return LessonOut(**vars(lesson))


@app.post(
    "/students/{student_id}/progress",
    response_model=ProgressOut,
    tags=["students"],
)
def record_progress(
    student_id: str,
    body: ProgressIn,
    _: dict = Depends(require_student_access),
) -> ProgressOut:
    """Record (upsert) a student's progress against one lesson.

    ``status`` must be one of ``not_started``, ``in_progress``, or
    ``completed``; ``completed`` stamps ``completed_at`` with the current UTC
    time. The row is upserted on the (student, lesson) pair, so repeated posts
    update the existing record rather than duplicating it.
    """

    from core.repo import ProgressRow, _now_iso

    _require_student(student_id)
    status = (body.status or "in_progress").strip().lower()
    if status not in {"not_started", "in_progress", "completed"}:
        raise HTTPException(
            status_code=422,
            detail="status must be not_started, in_progress, or completed",
        )
    repo = get_repository()
    saved = repo.upsert_progress(
        ProgressRow(
            id="",
            student_id=student_id,
            lesson_id=body.lesson_id,
            status=status,
            time_spent_seconds=int(body.time_spent_seconds or 0),
            completed_at=_now_iso() if status == "completed" else None,
        )
    )
    return ProgressOut(**vars(saved))


@app.post(
    "/students/{student_id}/activity",
    response_model=ActivityOut,
    tags=["students"],
)
def record_student_activity(
    student_id: str,
    body: ActivityIn,
    _: dict = Depends(require_student_access),
) -> ActivityOut:
    """Record a raw learner activity event and return the created row.

    The event is stamped with a fresh id and UTC ``created_at`` by the
    repository. Its ``payload`` is a free-form JSON object; a ``seconds`` key,
    when present, feeds the study-minutes time series.
    """

    from core.repo import ActivityEvent

    _require_student(student_id)
    repo = get_repository()
    created = repo.record_activity(
        ActivityEvent(
            id="",
            student_id=student_id,
            type=body.type,
            payload=body.payload or {},
        )
    )
    return ActivityOut(**vars(created))
