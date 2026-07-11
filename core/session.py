"""Streamlit session helpers: auth, cached repository, and demo accounts.

These functions form a thin bridge between the Streamlit UI (``Home.py`` and
the ``pages/`` multipage views) and the backend-agnostic :class:`Repository`.
They are deliberately small so every page shares one login model and one
cached data layer instead of re-wiring the repository on every rerun.

The repository is cached on ``st.session_state`` so a single SQLite
connection is reused across reruns within a user session. On first use, if the
backend has no profiles, the demo dataset is seeded so the app is never empty.

Side-effects (activity logging, seeding) are best-effort: a failure there logs
and continues, it never breaks a user flow.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
from pathlib import Path
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

from core.config import get_settings
from core.repo import (
    ActivityEvent,
    Profile,
    Repository,
    SqliteRepo,
    get_repo,
    seed_demo,
)

_LOG = logging.getLogger(__name__)

_REPO_KEY = "_navilearn_repo"
_USER_KEY = "user"
_LOGGED_OUT_KEY = "_navilearn_logged_out"

# Fixed ids used by ``seed_demo`` for the one-click demo accounts.
_DEMO_STUDENT_ID = "student-demo"
_DEMO_MENTOR_ID = "mentor-demo"
_DEMO_TEACHER_ID = "teacher-demo"

# Persistent-login cookie: stores the logged-in profile id in the browser so the
# session survives full page reloads and server restarts, not just reruns.
# The id is HMAC-signed (see _sign/_unsign) so the cookie cannot be forged to
# impersonate another profile, e.g. by guessing the "mentor-demo" id.
_COOKIE_UID = "navilearn_uid"
_COOKIE_DAYS = 14

# File holding the per-deployment HMAC secret. Kept on disk (not regenerated per
# process) so signed cookies stay valid across restarts, which is the whole point
# of persistent login. Gitignored; never committed.
_SECRET_FILE = ".session_secret"


def _session_secret() -> bytes:
    """Return a stable per-deployment secret for signing login cookies.

    Prefers ``Settings.session_secret`` (env); otherwise reads a persisted
    ``.session_secret`` file, creating it once with a random value. Falling back
    to a fixed constant only if the filesystem is read-only keeps signing working
    (best-effort) rather than breaking login entirely.
    """

    configured = getattr(get_settings(), "session_secret", "") or ""
    if configured:
        return configured.encode("utf-8")
    path = Path(_SECRET_FILE)
    try:
        if path.exists():
            data = path.read_text(encoding="utf-8").strip()
            if data:
                return data.encode("utf-8")
        token = secrets.token_urlsafe(32)
        path.write_text(token, encoding="utf-8")
        return token.encode("utf-8")
    except OSError as exc:  # noqa: BLE001 - read-only FS must not break login.
        _LOG.warning("Session secret file unavailable, using ephemeral: %s", exc)
        return b"navilearn-ephemeral-secret"


def _sign(uid: str) -> str:
    """Return ``uid.signature`` where signature is an HMAC-SHA256 of the uid."""

    mac = hmac.new(_session_secret(), uid.encode("utf-8"), hashlib.sha256)
    return f"{uid}.{mac.hexdigest()}"


def _unsign(value: str) -> Optional[str]:
    """Return the uid from a signed cookie, or ``None`` if the signature is bad.

    Uses a constant-time compare so a forged cookie (unknown or tampered id)
    cannot be used to log in as another profile.
    """

    if not value or "." not in value:
        return None
    uid, _, sig = value.rpartition(".")
    if not uid or not sig:
        return None
    expected = hmac.new(_session_secret(), uid.encode("utf-8"), hashlib.sha256)
    if hmac.compare_digest(sig, expected.hexdigest()):
        return uid
    return None


# Process-wide cache of resolved profiles, keyed by uid. Streamlit runs one
# process, so this survives every rerun AND every WebSocket reconnect. It makes a
# cookie-based session restore instant and network-independent after the first
# successful resolve, which is what stops the "Please sign in" flicker: on a
# reconnect the auto-refresh pages would otherwise re-hit Supabase on every
# restore, and a single slow or failed call would momentarily log the user out.
# Only successful lookups are cached, so a transient failure is retried later.
_PROFILE_CACHE: dict[str, "Profile"] = {}


def _resolve_profile(uid: str) -> Optional["Profile"]:
    """Resolve ``uid`` to a Profile, using a process-wide cache of successes."""

    cached = _PROFILE_CACHE.get(uid)
    if cached is not None:
        return cached
    try:
        profile = get_repo_cached().get_profile(uid)
    except Exception as exc:  # noqa: BLE001 - restore must never crash a page.
        _LOG.warning("Cookie session restore failed: %s", exc)
        return None
    if profile is not None:
        _PROFILE_CACHE[uid] = profile
    return profile


def _restore_from_cookie() -> None:
    """If not signed in, restore the session from the persistence cookie.

    Reads the cookie synchronously via ``st.context.cookies`` (no component, no
    rerun loop). The cookie is set by the browser on login (see logout/login).
    Profile resolution goes through the process-wide cache so a reconnect never
    flickers the user to a signed-out state.
    """

    if st.session_state.get(_USER_KEY) is not None:
        return
    # After an explicit logout, do not silently restore from a cookie the browser
    # may not have cleared yet: honour the sign-out until the next real login.
    if st.session_state.get(_LOGGED_OUT_KEY):
        return
    try:
        cookies = getattr(st.context, "cookies", None) or {}
        uid = _unsign(cookies.get(_COOKIE_UID) or "")
    except Exception as exc:  # noqa: BLE001 - cookie read must never crash a page.
        _LOG.warning("Cookie read failed: %s", exc)
        return
    if not uid:
        return
    profile = _resolve_profile(uid)
    if profile is not None:
        st.session_state[_USER_KEY] = profile


def _write_cookie(value: str, max_age: int) -> None:
    """Set (or, with max_age=0, clear) the login cookie via a tiny JS snippet.

    The value is JSON-encoded (not raw-interpolated) so a profile id can never
    break out of the string literal into executable JS. ``max_age`` is coerced
    to an int for the same reason. See docs/CH5_PRIVACY.md for the cookie's
    threat model (demo-grade opaque id, not a signed session token).
    """

    safe_value = json.dumps(str(value))  # yields a quoted, escaped JS string
    safe_max_age = int(max_age)
    try:
        components.html(
            "<script>document.cookie="
            f"'{_COOKIE_UID}='+encodeURIComponent({safe_value})"
            f"+'; path=/; max-age={safe_max_age}; SameSite=Lax';</script>",
            height=0,
        )
    except Exception as exc:  # noqa: BLE001 - persistence is best-effort.
        _LOG.warning("Cookie write failed: %s", exc)


def get_repo_cached() -> Repository:
    """Return a process-lived :class:`Repository`, cached on session state.

    Builds ``get_repo(get_settings())`` once per Streamlit session and stores it
    on ``st.session_state``. On first construction, if the backend reports no
    profiles, the demo dataset is seeded so dashboards have data to render.
    Seeding is best-effort: a failure is logged and the (empty) repo is still
    returned.
    """

    repo: Optional[Repository] = st.session_state.get(_REPO_KEY)
    if repo is not None:
        return repo

    settings = get_settings()
    try:
        repo = get_repo(settings)
    except Exception as exc:  # noqa: BLE001 - a bad backend must not crash the app.
        # The configured backend (for example a misconfigured or unreachable
        # Supabase project) could not be built. Fall back to the always-local
        # SQLite backend so the platform stays usable and demoable.
        _LOG.warning("Configured backend unavailable, using SQLite: %s", exc)
        repo = SqliteRepo(settings.sqlite_path)

    try:
        if not repo.list_profiles():
            seed_demo(repo)
    except Exception as exc:  # noqa: BLE001 - seeding never blocks the app.
        _LOG.warning("Demo seed skipped: %s", exc)

    st.session_state[_REPO_KEY] = repo
    return repo


def current_user() -> Optional[Profile]:
    """Return the logged-in :class:`Profile`, or ``None`` if not signed in.

    If no user is in session state, attempts to restore one from the persistent
    login cookie so a reload or server restart does not sign the user out.
    """

    user = st.session_state.get(_USER_KEY)
    if isinstance(user, Profile):
        return user
    _restore_from_cookie()
    user = st.session_state.get(_USER_KEY)
    return user if isinstance(user, Profile) else None


def login(email: str, full_name: str, role: str = "student") -> Profile:
    """Sign a user in, creating or updating their profile, and record it.

    Lookup is idempotent by email: an existing profile with the same email
    (case-insensitive) is reused so repeated logins do not create duplicates.
    The resulting profile is stored on ``st.session_state['user']`` and a
    best-effort ``'login'`` activity event is recorded.
    """

    repo = get_repo_cached()
    email = (email or "").strip()
    full_name = (full_name or "").strip() or email or "Learner"
    role = (role or "student").strip().lower()

    existing = _find_profile_by_email(repo, email)
    profile = Profile(
        id=existing.id if existing else "",
        email=email,
        full_name=full_name,
        role=role,
        mentor_id=existing.mentor_id if existing else None,
    )
    profile = repo.upsert_profile(profile)

    st.session_state[_USER_KEY] = profile
    st.session_state.pop(_LOGGED_OUT_KEY, None)  # a fresh login clears the flag
    _persist_login_cookie(profile)
    _record_login(repo, profile)
    return profile


def logout() -> None:
    """Sign the user out for good: clear session state and the persistence cookie.

    Also sets a per-session ``_LOGGED_OUT_KEY`` flag so that on the immediate
    rerun :func:`_restore_from_cookie` does NOT log the user straight back in from
    a cookie that the browser has not cleared yet. The flag is cleared on the next
    real :func:`login`, so logout reliably lands on the sign-in page.
    """

    st.session_state.pop(_USER_KEY, None)
    st.session_state[_LOGGED_OUT_KEY] = True
    _write_cookie("", 0)


def _persist_login_cookie(profile: Profile) -> None:
    """Write the signed profile id to a browser cookie so login survives reloads."""

    _write_cookie(_sign(profile.id), _COOKIE_DAYS * 86400)


def require_user() -> Profile:
    """Return the current user, or warn and stop the page if not signed in."""

    user = current_user()
    if user is None:
        st.warning("Please sign in from the NaviLearn home page to continue.")
        st.stop()
    return user  # type: ignore[return-value]  # st.stop halts before this on None.


def demo_accounts(repo: Repository) -> list[Profile]:
    """Return the seeded demo accounts for one-click login.

    Prefers the fixed seeded student and mentor ids; falls back to the first
    student and first mentor found if the ids differ. Returns an empty list if
    the backend has no matching profiles.
    """

    accounts: list[Profile] = []
    seen: set[str] = set()

    for demo_id in (_DEMO_STUDENT_ID, _DEMO_MENTOR_ID, _DEMO_TEACHER_ID):
        try:
            profile = repo.get_profile(demo_id)
        except Exception as exc:  # noqa: BLE001 - lookup is best-effort.
            _LOG.warning("Demo account lookup failed for %s: %s", demo_id, exc)
            profile = None
        if profile is not None and profile.id not in seen:
            accounts.append(profile)
            seen.add(profile.id)

    if not accounts:
        for role in ("student", "mentor", "teacher"):
            try:
                by_role = repo.list_profiles(role)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("Profile listing failed for role %s: %s", role, exc)
                by_role = []
            if by_role and by_role[0].id not in seen:
                accounts.append(by_role[0])
                seen.add(by_role[0].id)

    return accounts


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _find_profile_by_email(repo: Repository, email: str) -> Optional[Profile]:
    """Return the first profile whose email matches, case-insensitively."""

    if not email:
        return None
    target = email.lower()
    try:
        profiles = repo.list_profiles()
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("Profile listing failed during login: %s", exc)
        return None
    for profile in profiles:
        if (profile.email or "").strip().lower() == target:
            return profile
    return None


def _record_login(repo: Repository, profile: Profile) -> None:
    """Record a best-effort ``'login'`` activity event for ``profile``."""

    try:
        repo.record_activity(
            ActivityEvent(
                id="",
                student_id=profile.id,
                type="login",
                payload={"role": profile.role},
            )
        )
    except Exception as exc:  # noqa: BLE001 - telemetry never breaks login.
        _LOG.warning("Login activity not recorded: %s", exc)
