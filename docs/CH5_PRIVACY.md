# Challenge 5: Privacy-first design (Live Classroom)

The Live Classroom is a shared collaboration room: notes, polls, chat, and a
co-solve workspace, all persisted in Supabase Postgres. This note documents the
privacy posture and gives an OPTIONAL hardening step you can apply later.

## What data the classroom stores

| Table | Purpose | Personal data |
| --- | --- | --- |
| `classroom_sessions` | The room itself | `created_by` (a profile id) |
| `classroom_notes` | One shared notes doc per room | `updated_by` (display name) |
| `classroom_polls` | Poll questions and options | none |
| `poll_votes` | One vote per voter per poll | `voter_id` (a profile id) |
| `chat_messages` | Running chat log | `author_id`, `author_name`, message text |
| `classroom_solve` | One shared code/text workspace per room | `updated_by` (display name), code text |

## Privacy-first design choices

- **Minimal identity on the row.** Writes record a display name or profile id,
  never an email or any contact detail. The author name is a snapshot so later
  profile edits do not rewrite history.
- **Session scoping.** Every collaborative row is keyed by `session_id`. Reads
  in `core/classroom.py` always filter `eq("session_id", ...)`, so one room can
  never read another room's notes, votes, chat, or solution. The single shared
  demo room (`main-classroom`) is a deliberate default, not a leak of arbitrary
  rooms.
- **No third parties in this feature.** The classroom is Supabase-only. It calls
  no LLM, sends nothing to Groq/Sarvam/OpenAI, and spends no money. Text you type
  here is not used for model training or inference.
- **Best-effort, fail-closed reads.** A backend hiccup degrades reads to empty
  and writes to no-ops (logged, swallowed). A failure never dumps another user's
  data or a stack trace into the UI.
- **Data minimization on export.** `export_summary` renders only the current
  room's own notes, poll tallies, and recent chat: no cross-room aggregation.
- **Transport security.** All Supabase traffic is HTTPS; the service-role key
  lives only in `.env` (gitignored), never in client code or the page.

## Current access model (and its tradeoff)

The app writes with the Supabase **service-role key** from a trusted server-side
process (`core/classroom.py` owns the only client). Row Level Security (RLS) is
**disabled** on the classroom tables.

Tradeoff: the service-role key bypasses RLS entirely, so table-level RLS adds no
protection against the app itself. Enabling RLS **without** policies would block
every read and write and break the room, because service-key access still needs
policies (or the bypass) to function as expected against anon/authenticated
roles. That is why RLS is intentionally left off here and is NOT auto-applied.

RLS becomes worthwhile the day the classroom is accessed directly by the client
with an `anon` or `authenticated` key (for example, moving to Supabase Realtime
or client-side reads). At that point per-session policies stop one signed-in user
from reading another session's rows at the database layer.

## OPTIONAL hardening: enable RLS with per-session policies

Do NOT run this while the app uses the service-role key for all access, unless
you also move client reads onto an anon/authenticated key. Applying RLS as-is
will not break the service-key path (service-role bypasses RLS), but the policies
below are written for the anon/authenticated migration and are provided as a
forward-looking reference.

```sql
-- 1) Turn RLS on for every classroom table.
alter table public.classroom_sessions enable row level security;
alter table public.classroom_notes    enable row level security;
alter table public.classroom_polls    enable row level security;
alter table public.poll_votes         enable row level security;
alter table public.chat_messages      enable row level security;
alter table public.classroom_solve    enable row level security;

-- 2) Model "membership" of a session however your app defines it. The simplest
--    version below treats any authenticated user as a member of any session
--    (demo behavior). Replace the `using (true)` predicates with a real
--    membership check, e.g. an EXISTS against a classroom_members table:
--
--      using (
--        exists (
--          select 1 from public.classroom_members m
--          where m.session_id = classroom_notes.session_id
--            and m.user_id = auth.uid()::text
--        )
--      )

-- classroom_solve: read/write only your session's shared workspace.
create policy classroom_solve_select on public.classroom_solve
  for select to authenticated using (true);
create policy classroom_solve_write on public.classroom_solve
  for all to authenticated using (true) with check (true);

-- Repeat the same shape per table, scoping by session_id where present.
create policy classroom_notes_select on public.classroom_notes
  for select to authenticated using (true);
create policy classroom_notes_write on public.classroom_notes
  for all to authenticated using (true) with check (true);

create policy chat_messages_select on public.chat_messages
  for select to authenticated using (true);
create policy chat_messages_write on public.chat_messages
  for all to authenticated using (true) with check (true);

create policy classroom_polls_select on public.classroom_polls
  for select to authenticated using (true);
create policy classroom_polls_write on public.classroom_polls
  for all to authenticated using (true) with check (true);

create policy poll_votes_select on public.poll_votes
  for select to authenticated using (true);
create policy poll_votes_write on public.poll_votes
  for all to authenticated using (true) with check (true);

create policy classroom_sessions_select on public.classroom_sessions
  for select to authenticated using (true);
create policy classroom_sessions_write on public.classroom_sessions
  for all to authenticated using (true) with check (true);
```

To revert:

```sql
alter table public.classroom_solve disable row level security;
-- ...and the same for the other classroom tables.
```
