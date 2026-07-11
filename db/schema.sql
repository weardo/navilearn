-- NaviLearn portable schema (SQLite dialect).
-- The Supabase/Postgres counterpart lives in
-- supabase/migrations/20260711010000_navilearn_platform.sql and models the
-- same entities with uuid PKs, timestamptz, and a pgvector chunks table.
--
-- All tables use TEXT ids so the two backends can share application code:
-- SQLite stores opaque string ids, Postgres stores uuid text.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS profiles (
    id          TEXT PRIMARY KEY,
    email       TEXT UNIQUE,
    full_name   TEXT NOT NULL DEFAULT '',
    role        TEXT NOT NULL DEFAULT 'student'
                CHECK (role IN ('student', 'mentor', 'teacher')),
    mentor_id   TEXT REFERENCES profiles (id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_profiles_role      ON profiles (role);
CREATE INDEX IF NOT EXISTS idx_profiles_mentor    ON profiles (mentor_id);

CREATE TABLE IF NOT EXISTS courses (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS lessons (
    id          TEXT PRIMARY KEY,
    course_id   TEXT NOT NULL REFERENCES courses (id) ON DELETE CASCADE,
    title       TEXT NOT NULL DEFAULT '',
    order_index INTEGER NOT NULL DEFAULT 0,
    content     TEXT NOT NULL DEFAULT '',  -- markdown lesson body
    module      TEXT NOT NULL DEFAULT '',  -- section name grouping lessons
    video_url   TEXT NOT NULL DEFAULT '',  -- optional embedded video
    doc_url     TEXT NOT NULL DEFAULT ''   -- optional attached document
);

CREATE INDEX IF NOT EXISTS idx_lessons_course ON lessons (course_id, order_index);

CREATE TABLE IF NOT EXISTS progress (
    id                 TEXT PRIMARY KEY,
    student_id         TEXT NOT NULL REFERENCES profiles (id) ON DELETE CASCADE,
    lesson_id          TEXT NOT NULL REFERENCES lessons (id) ON DELETE CASCADE,
    status             TEXT NOT NULL DEFAULT 'not_started'
                       CHECK (status IN ('not_started', 'in_progress', 'completed')),
    time_spent_seconds INTEGER NOT NULL DEFAULT 0,
    completed_at       TEXT,
    UNIQUE (student_id, lesson_id)
);

CREATE INDEX IF NOT EXISTS idx_progress_student ON progress (student_id);

CREATE TABLE IF NOT EXISTS activity_events (
    id          TEXT PRIMARY KEY,
    student_id  TEXT NOT NULL REFERENCES profiles (id) ON DELETE CASCADE,
    type        TEXT NOT NULL DEFAULT '',
    payload     TEXT NOT NULL DEFAULT '{}',  -- JSON encoded as text
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_activity_student ON activity_events (student_id, created_at);

CREATE TABLE IF NOT EXISTS study_sets (
    id          TEXT PRIMARY KEY,
    owner_id    TEXT NOT NULL REFERENCES profiles (id) ON DELETE CASCADE,
    title       TEXT NOT NULL DEFAULT '',
    source      TEXT NOT NULL DEFAULT '',
    content     TEXT NOT NULL DEFAULT '{}',  -- JSON encoded as text
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_study_sets_owner ON study_sets (owner_id, created_at);

CREATE TABLE IF NOT EXISTS interview_reports (
    id            TEXT PRIMARY KEY,
    student_id    TEXT NOT NULL REFERENCES profiles (id) ON DELETE CASCADE,
    project_title TEXT NOT NULL DEFAULT '',
    scores        TEXT NOT NULL DEFAULT '{}',  -- JSON encoded as text
    feedback      TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reports_student ON interview_reports (student_id, created_at);
