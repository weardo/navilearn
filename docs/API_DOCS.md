# NaviLearn REST API

The NaviLearn API (`api.py`) exposes the same dashboard data the Streamlit UI
renders, as a plain HTTP/JSON service built on FastAPI. It is a thin consumer of
`core.repo`: every route shapes repository output into a typed Pydantic response
and holds no business logic of its own. One shared `Repository` is built at
import time from `.env` (Supabase by default, SQLite otherwise); all routes read
and write through that single instance.

## How to run

Use the repo venv. From the project root:

```bash
.venv/bin/uvicorn api:app --reload --port 8000
```

- Base URL: `http://127.0.0.1:8000`
- Swagger UI (interactive docs): `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- OpenAPI schema (JSON): `http://127.0.0.1:8000/openapi.json`

The Swagger page at `/docs` is generated from the route/model definitions and is
the always-current, interactive companion to this document.

## Seed data

The API serves whatever is in the configured backend. To get a small, realistic
dataset (one mentor, one student `Sam Student`, one teacher, two courses with
lessons, ~30 days of activity, progress rows, a study set, and an interview
report), seed the demo once:

```bash
.venv/bin/python -m core.repo
```

Well-known demo accounts (all use the password `navilearn`):

| email                    | role    | name         |
|--------------------------|---------|--------------|
| `student@navilearn.dev`  | student | Sam Student  |
| `mentor@navilearn.dev`   | mentor  | Maya Mentor  |
| `teacher@navilearn.dev`  | teacher | Tara Teacher |

Well-known demo course ids used by the examples below:

- Course `Python Foundations` with lessons `course-python-l0` .. `course-python-l4`.
- Course `Intro to Machine Learning` with lessons `course-ml-l0` .. `course-ml-l3`.

The demo passwords are provisioned by `api.seed_demo_passwords()` (call it once
after seeding; it is idempotent). If a demo account still reports "no API
password set" on login, run that function against the configured backend.

Note: on the Supabase backend, readable ids such as `course-python-l0` are
stored as deterministic UUIDs (uuid5). Read the ids back from the list endpoints
(`GET /courses`, `GET /courses/{course_id}/lessons`) and use those exact values
in subsequent calls rather than hardcoding the readable form.

## Authentication

Student-scoped routes are protected. The flow is real, not demo-grade:

1. `POST /auth/login` with `{email, password}`. On success it returns
   `{ "token": "<signed-bearer-token>", "profile": { ... } }`.
2. Send that token as `Authorization: Bearer <token>` on every
   `GET/POST /students/{student_id}/*` route and on `GET /students`.

Authorization is enforced per request:

- A student may access ONLY their own `student_id`. Reaching another student's
  id returns `403`.
- A mentor or teacher may access any student's routes, and only mentors/teachers
  may list the roster (`GET /students`); a student calling it gets `403`.
- A missing, malformed, tampered, or expired token returns `401`.

Tokens are stateless: an HMAC-SHA256 signature over a base64url payload
`{sub, role, exp}`, signed with a server secret (`API_SECRET`,
`settings.session_secret`, or an auto-generated `.api_secret` file). They expire
after 24 hours by default.

`/health` and the `/courses*` catalog routes are public (no token required).

Capture the token into a shell variable and reuse it:

```bash
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"student@navilearn.dev","password":"navilearn"}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["token"])')

SID=$(curl -s -X POST http://127.0.0.1:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"student@navilearn.dev","password":"navilearn"}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["profile"]["id"])')

curl -s -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:8000/students/$SID/summary"
```

## Conventions

- All request and response bodies are JSON.
- Errors use standard HTTP status codes with a FastAPI `{"detail": "..."}`
  body: `404` when a profile, course, or lesson is missing; `422` for invalid
  input (for example an unknown progress `status`).
- Every route is typed with a Pydantic response model and grouped under a tag
  (`system`, `auth`, `students`, `courses`) visible in Swagger.

---

## Endpoints

### GET /health

Liveness probe. Reports status and the active data backend class name.

Tag: `system`

```bash
curl -s http://127.0.0.1:8000/health
```

```json
{ "status": "ok", "backend": "SupabaseRepo" }
```

---

### POST /auth/login

Authenticate by email + password and return a signed bearer token plus the
profile. Resolved against the `profiles` table:

- Email exists AND has a password: the password is verified (`401` on mismatch);
  the caller is logged in with the account's stored role.
- Email exists but has NO password set: `401` (an API password must be
  provisioned out of band first).
- Email does not exist: a NEW profile is registered as a `student` with the
  given password, then logged in.

The client can never request a `role`: new accounts are always `student`, and an
existing account keeps its stored role.

Tag: `auth`

Request body:

| field       | type   | required | default | notes                                  |
|-------------|--------|----------|---------|----------------------------------------|
| `email`     | string | yes      | -       |                                        |
| `password`  | string | yes      | -       |                                        |
| `full_name` | string | no       | `email` | used only when registering a new account |

```bash
curl -s -X POST http://127.0.0.1:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"student@navilearn.dev","password":"navilearn"}'
```

```json
{
  "token": "eyJ...payload....signature",
  "profile": {
    "id": "a1b2c3d4-0000-5000-8000-000000000000",
    "email": "student@navilearn.dev",
    "full_name": "Sam Student",
    "role": "student",
    "mentor_id": null
  }
}
```

A wrong password returns `401 {"detail": "Invalid credentials"}`.

---

### GET /students

List all student profiles. Requires a bearer token for a `mentor` or `teacher`
(`403` for a student, `401` without a token).

Tag: `students`

```bash
curl -s -H "Authorization: Bearer $MENTOR_TOKEN" http://127.0.0.1:8000/students
```

```json
[
  {
    "id": "a1b2c3d4-0000-5000-8000-000000000000",
    "email": "student@navilearn.dev",
    "full_name": "Sam Student",
    "role": "student",
    "mentor_id": "b2c3d4e5-0000-5000-8000-000000000000"
  }
]
```

---

### GET /students/{student_id}/summary

Headline dashboard metrics for one student. `404` if the student has no profile.

Requires a bearer token (see Authentication): a student may read only their own
`student_id` (`403` otherwise); mentors/teachers may read any student. All the
`/students/{student_id}/*` routes below enforce the same rule.

Tag: `students`

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:8000/students/$SID/summary"
```

```json
{
  "student_id": "a1b2c3d4-0000-5000-8000-000000000000",
  "full_name": "Sam Student",
  "lessons_completed": 7,
  "hours_spent": 3.4,
  "courses_in_progress": 1,
  "latest_interview_overall": 7.0,
  "latest_interview_project": "Todo CLI in Python"
}
```

---

### GET /students/{student_id}/progress

Per-course completion for one student. `404` if the student has no profile.

Tag: `students`

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:8000/students/$SID/progress"
```

```json
[
  { "course": "Intro to Machine Learning", "completed": 2, "total": 4, "pct": 50.0 },
  { "course": "Python Foundations", "completed": 5, "total": 5, "pct": 100.0 }
]
```

---

### GET /students/{student_id}/timeseries

Daily study-minutes time series. Optional `days` query parameter (1..365,
default 30). `404` if the student has no profile.

Tag: `students`

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:8000/students/$SID/timeseries?days=7"
```

```json
[
  { "date": "2026-07-05", "minutes": 42.0 },
  { "date": "2026-07-06", "minutes": 0.0 },
  { "date": "2026-07-07", "minutes": 18.5 }
]
```

---

### GET /students/{student_id}/activity

Raw activity events for one student, oldest first. `404` if the student has no
profile.

Tag: `students`

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:8000/students/$SID/activity"
```

```json
[
  {
    "id": "e1f2a3b4-0000-5000-8000-000000000000",
    "student_id": "a1b2c3d4-0000-5000-8000-000000000000",
    "type": "lesson_view",
    "payload": { "seconds": 1200, "lesson_id": "course-python-l0" },
    "created_at": "2026-07-08T14:03:00+00:00"
  }
]
```

---

### GET /students/{student_id}/interviews

Interview reports for one student, newest first. `404` if the student has no
profile.

Tag: `students`

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:8000/students/$SID/interviews"
```

```json
[
  {
    "id": "c3d4e5f6-0000-5000-8000-000000000000",
    "student_id": "a1b2c3d4-0000-5000-8000-000000000000",
    "project_title": "Todo CLI in Python",
    "scores": { "communication": 7, "technical_depth": 6, "problem_solving": 8 },
    "feedback": "Solid structure. Practice articulating trade-offs out loud.",
    "created_at": "2026-07-10T09:00:00+00:00"
  }
]
```

---

### GET /courses

List all courses.

Tag: `courses`

```bash
curl -s http://127.0.0.1:8000/courses
```

```json
[
  {
    "id": "f0e1d2c3-0000-5000-8000-000000000000",
    "title": "Intro to Machine Learning",
    "description": "Supervised learning, evaluation, and a first model."
  },
  {
    "id": "a9b8c7d6-0000-5000-8000-000000000000",
    "title": "Python Foundations",
    "description": "Variables, control flow, functions, and data structures."
  }
]
```

---

### GET /courses/{course_id}/lessons

List the lessons of one course, ordered by `order_index`, including the full
teaching material for each lesson (`content` markdown body, `module` section
name, and optional `video_url` / `doc_url` media links). `404` if the course
does not exist.

Tag: `courses`

```bash
curl -s http://127.0.0.1:8000/courses/a9b8c7d6-0000-5000-8000-000000000000/lessons
```

```json
[
  {
    "id": "11112222-0000-5000-8000-000000000000",
    "course_id": "a9b8c7d6-0000-5000-8000-000000000000",
    "title": "Variables and Types",
    "order_index": 0,
    "content": "# Variables and Types\n\nPart of **Python Foundations**. ...",
    "module": "Fundamentals",
    "video_url": "",
    "doc_url": ""
  }
]
```

---

### GET /courses/{course_id}/lessons/{lesson_id}

Return one lesson's full details. `404` if the lesson does not exist or does not
belong to the given course.

Tag: `courses`

```bash
curl -s http://127.0.0.1:8000/courses/a9b8c7d6-0000-5000-8000-000000000000/lessons/11112222-0000-5000-8000-000000000000
```

```json
{
  "id": "11112222-0000-5000-8000-000000000000",
  "course_id": "a9b8c7d6-0000-5000-8000-000000000000",
  "title": "Variables and Types",
  "order_index": 0,
  "content": "# Variables and Types\n\nPart of **Python Foundations**. ...",
  "module": "Fundamentals",
  "video_url": "",
  "doc_url": ""
}
```

---

### POST /students/{student_id}/progress

Record (upsert) a student's progress against one lesson. The row is upserted on
the `(student, lesson)` pair, so repeated posts update the existing record
rather than duplicating it. `404` if the student has no profile; `422` if
`status` is not one of `not_started`, `in_progress`, `completed`. When `status`
is `completed`, `completed_at` is stamped with the current UTC time; otherwise
it is `null`.

Tag: `students`

Request body:

| field                | type   | required | default          |
|----------------------|--------|----------|------------------|
| `lesson_id`          | string | yes      | -                |
| `status`             | string | no       | `"in_progress"`  |
| `time_spent_seconds` | int    | no       | `0`              |

```bash
curl -s -X POST "http://127.0.0.1:8000/students/$SID/progress" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"lesson_id":"11112222-0000-5000-8000-000000000000","status":"completed","time_spent_seconds":120}'
```

```json
{
  "id": "22223333-0000-5000-8000-000000000000",
  "student_id": "a1b2c3d4-0000-5000-8000-000000000000",
  "lesson_id": "11112222-0000-5000-8000-000000000000",
  "status": "completed",
  "time_spent_seconds": 120,
  "completed_at": "2026-07-11T12:00:00+00:00"
}
```

---

### POST /students/{student_id}/activity

Record a raw learner activity event and return the created row. The event is
stamped with a fresh id and UTC `created_at` by the repository. `payload` is a
free-form JSON object; a `seconds` key, when present, feeds the study-minutes
time series. `404` if the student has no profile.

Tag: `students`

Request body:

| field     | type   | required | default |
|-----------|--------|----------|---------|
| `type`    | string | yes      | -       |
| `payload` | object | no       | `{}`    |

```bash
curl -s -X POST "http://127.0.0.1:8000/students/$SID/activity" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"type":"lesson_completed","payload":{"lesson_id":"11112222-0000-5000-8000-000000000000","seconds":120}}'
```

```json
{
  "id": "33334444-0000-5000-8000-000000000000",
  "student_id": "a1b2c3d4-0000-5000-8000-000000000000",
  "type": "lesson_completed",
  "payload": { "lesson_id": "11112222-0000-5000-8000-000000000000", "seconds": 120 },
  "created_at": "2026-07-11T12:00:00+00:00"
}
```

## Security and authentication

The IDOR / broken-auth hole is closed. The API now authenticates callers and
authorizes every student-scoped request:

- **`POST /auth/login` verifies a real credential.** Passwords are checked
  against a per-account `profiles.password_hash` (PBKDF2-HMAC-SHA256, random
  salt, 200k iterations, constant-time compare). A wrong password returns `401`;
  an account with no password set returns `401`. The client-supplied role is
  never trusted: new accounts are always `student`, existing accounts keep their
  stored role.
- **Bearer tokens gate the protected routes.** Login returns a stateless
  HMAC-SHA256 token (`{sub, role, exp}`, 24h TTL) signed with a server secret.
  `require_caller` rejects a missing/malformed/tampered/expired token with `401`.
- **Ownership is enforced (no more IDOR).** `require_student_access` allows a
  `GET/POST /students/{student_id}/*` request only when
  `caller.sub == student_id` OR the caller is a `mentor`/`teacher`; otherwise
  `403`. `GET /students` (the roster) is mentor/teacher only.
- `/health` and `/courses*` are intentionally public (liveness + read-only
  catalog).

Remaining hardening for a real multi-tenant deployment (out of scope for this
single-owner demo): rotate/persist a strong `API_SECRET` in a secret manager,
turn on Row Level Security on the underlying tables (see docs/CH5_PRIVACY.md)
rather than relying only on the service-role key, add rate limiting on
`/auth/login`, and consider refresh tokens / OIDC instead of a single long-lived
bearer.
