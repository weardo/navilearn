-- NaviLearn platform schema (Postgres + pgvector).
-- Postgres counterpart of db/schema.sql. Applied when a Supabase database is
-- reachable. It does not need to run against SQLite. Embedding dimension is
-- 384 to match fastembed bge-small-en-v1.5 output used across the pipeline.

create extension if not exists "pgcrypto";
create extension if not exists "vector";

-- Profiles ------------------------------------------------------------------
create table if not exists public.profiles (
    id         uuid primary key default gen_random_uuid(),
    email      text unique,
    full_name  text not null default '',
    role       text not null default 'student'
               check (role in ('student', 'mentor', 'teacher')),
    mentor_id  uuid references public.profiles (id) on delete set null
);

create index if not exists idx_profiles_role   on public.profiles (role);
create index if not exists idx_profiles_mentor on public.profiles (mentor_id);

-- Courses and lessons -------------------------------------------------------
create table if not exists public.courses (
    id          uuid primary key default gen_random_uuid(),
    title       text not null default '',
    description text not null default ''
);

create table if not exists public.lessons (
    id          uuid primary key default gen_random_uuid(),
    course_id   uuid not null references public.courses (id) on delete cascade,
    title       text not null default '',
    order_index integer not null default 0
);

create index if not exists idx_lessons_course
    on public.lessons (course_id, order_index);

-- Progress ------------------------------------------------------------------
create table if not exists public.progress (
    id                 uuid primary key default gen_random_uuid(),
    student_id         uuid not null references public.profiles (id) on delete cascade,
    lesson_id          uuid not null references public.lessons (id) on delete cascade,
    status             text not null default 'not_started'
                       check (status in ('not_started', 'in_progress', 'completed')),
    time_spent_seconds integer not null default 0,
    completed_at       timestamptz,
    unique (student_id, lesson_id)
);

create index if not exists idx_progress_student on public.progress (student_id);

-- Activity events -----------------------------------------------------------
create table if not exists public.activity_events (
    id          uuid primary key default gen_random_uuid(),
    student_id  uuid not null references public.profiles (id) on delete cascade,
    type        text not null default '',
    payload     jsonb not null default '{}'::jsonb,
    created_at  timestamptz not null default now()
);

create index if not exists idx_activity_student
    on public.activity_events (student_id, created_at);

-- Study sets ----------------------------------------------------------------
create table if not exists public.study_sets (
    id          uuid primary key default gen_random_uuid(),
    owner_id    uuid not null references public.profiles (id) on delete cascade,
    title       text not null default '',
    source      text not null default '',
    created_at  timestamptz not null default now()
);

create index if not exists idx_study_sets_owner
    on public.study_sets (owner_id, created_at);

-- Interview reports ---------------------------------------------------------
create table if not exists public.interview_reports (
    id            uuid primary key default gen_random_uuid(),
    student_id    uuid not null references public.profiles (id) on delete cascade,
    project_title text not null default '',
    scores        jsonb not null default '{}'::jsonb,
    feedback      text not null default '',
    created_at    timestamptz not null default now()
);

create index if not exists idx_reports_student
    on public.interview_reports (student_id, created_at);

-- RAG chunks (pgvector) -----------------------------------------------------
-- Embeddings live here so the Supabase backend can serve retrieval without a
-- separate Chroma store. Dimension 384 = fastembed bge-small-en-v1.5.
create table if not exists public.chunks (
    id           uuid primary key default gen_random_uuid(),
    study_set_id uuid references public.study_sets (id) on delete cascade,
    source       text not null default '',
    topic        text not null default 'general',
    content      text not null default '',
    embedding    vector(384),
    created_at   timestamptz not null default now()
);

create index if not exists idx_chunks_study_set on public.chunks (study_set_id);
create index if not exists idx_chunks_topic      on public.chunks (topic);

-- Approximate nearest neighbour index over cosine distance.
create index if not exists idx_chunks_embedding
    on public.chunks using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

-- match_chunks: cosine-similarity retrieval adapted from the soupbrain RAG
-- pattern at dim 384. Returns rows scoring above threshold, best first.
create or replace function public.match_chunks(
    query vector(384),
    k int default 5,
    threshold float default 0.2
)
returns table (
    id           uuid,
    study_set_id uuid,
    source       text,
    topic        text,
    content      text,
    similarity   float
)
language sql
stable
as $$
    select
        c.id,
        c.study_set_id,
        c.source,
        c.topic,
        c.content,
        1 - (c.embedding <=> query) as similarity
    from public.chunks c
    where c.embedding is not null
      and 1 - (c.embedding <=> query) >= threshold
    order by c.embedding <=> query
    limit greatest(k, 1);
$$;
