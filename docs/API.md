# NaviLearn REST API

A FastAPI service that exposes the same data the Streamlit dashboard renders,
as HTTP/JSON. It is a thin consumer of `core.repo`: every route shapes
repository output into a typed response and holds no business logic of its own.
The app builds one shared `Repository` via `get_repo(get_settings())` (the
`.env` backend, Supabase by default).

## Run

```bash
.venv/bin/uvicorn api:app --reload --port 8000
```

## Interactive docs (the API documentation)

FastAPI auto-generates interactive documentation from the code and models:

- Swagger UI: <http://localhost:8000/docs>  ← the primary API docs, try requests live
- ReDoc: <http://localhost:8000/redoc>
- OpenAPI schema: <http://localhost:8000/openapi.json>

## Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET  | `/health` | Liveness probe; reports the active data backend. |
| POST | `/auth/login` | Upsert a profile by email (idempotent sign-in), returns the profile. |
| GET  | `/students` | List all student profiles. |
| GET  | `/students/{id}/summary` | Headline metrics: lessons completed, hours, courses in progress, latest interview. |
| GET  | `/students/{id}/progress` | Per-course completion (`completed`/`total`/`pct`). |
| GET  | `/students/{id}/timeseries?days=30` | Daily study-minutes time series. |
| GET  | `/students/{id}/activity` | Raw activity events for the student. |
| GET  | `/students/{id}/interviews` | Interview reports, newest first. |
| GET  | `/courses` | List all courses. |
| GET  | `/courses/{id}/lessons` | Lessons of one course, ordered. |

Unknown student or course ids return `404`. `POST /auth/login` with an empty
email returns `422`.

## Example curls

```bash
BASE=http://localhost:8000

# Health
curl -s $BASE/health

# Sign in (upsert) and read a student list
curl -s -X POST $BASE/auth/login \
  -H 'content-type: application/json' \
  -d '{"email":"student@navilearn.dev","full_name":"Sam Student","role":"student"}'

curl -s $BASE/students

# Dashboard data for the demo student
SID=student-demo
curl -s $BASE/students/$SID/summary
curl -s $BASE/students/$SID/progress
curl -s "$BASE/students/$SID/timeseries?days=14"
curl -s $BASE/students/$SID/activity
curl -s $BASE/students/$SID/interviews

# Catalog
curl -s $BASE/courses
curl -s $BASE/courses/course-python/lessons
```

Note: when the `.env` backend is Supabase, the demo ids are mapped to
deterministic UUIDs, so pass the id that `GET /students` returns rather than the
literal `student-demo` string.
