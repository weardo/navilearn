"""Supabase Storage helper for uploading course media into a public bucket.

Course authoring lets mentors/teachers attach media (docs, images, short
videos) to lessons. Those bytes live in a single PUBLIC Supabase Storage bucket
named ``course-media`` so the returned URL can be embedded directly in a lesson
(``video_url`` / ``doc_url``) and served without any signing dance.

This module owns exactly one Supabase client, built the same way
``core/classroom.py`` builds its client (from ``settings.supabase_url`` +
``settings.supabase_service_role_key``, cached for reuse across Streamlit
reruns). Every operation is best-effort: on any failure it logs and returns an
empty string rather than raising, so an upload hiccup never crashes an
authoring flow.
"""

from __future__ import annotations

import logging
import re
import uuid
from functools import lru_cache
from typing import Any

from core.config import get_settings

_LOG = logging.getLogger(__name__)

# The single public bucket every course upload lands in. It must already exist
# (created out-of-band) and be marked public so ``get_public_url`` resolves.
BUCKET = "course-media"


@lru_cache(maxsize=1)
def _client():
    """Return a cached Supabase client built from settings, or ``None``.

    Mirrors ``core/classroom._client``: cached because ``create_client`` opens a
    session worth reusing across reruns, and returns ``None`` (never raises) when
    the project is unconfigured or the client cannot be built, so callers degrade
    to a no-op instead of crashing the page.
    """

    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        _LOG.warning("Storage disabled: Supabase URL/service key not configured.")
        return None
    try:
        from supabase import create_client  # local import: optional dependency

        return create_client(
            settings.supabase_url, settings.supabase_service_role_key
        )
    except Exception as exc:  # noqa: BLE001 - a bad client must not crash the UI.
        _LOG.warning("Storage Supabase client unavailable: %s", exc)
        return None


def _sanitize_filename(filename: str) -> str:
    """Return a safe, non-empty object-name suffix derived from ``filename``.

    Keeps a conservative set of characters (letters, digits, dot, dash,
    underscore), collapses everything else to underscores, strips leading dots or
    slashes, and falls back to ``"file"`` when nothing usable remains. This keeps
    the storage key readable while avoiding path traversal or odd bytes.
    """

    base = (filename or "").strip().replace("\\", "/").split("/")[-1]
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return cleaned or "file"


def _extract_public_url(result: Any) -> str:
    """Coerce ``get_public_url``'s return value into a plain URL string.

    supabase-py has returned this as a bare string in some versions and as a dict
    (``{"publicUrl": ...}`` / ``{"data": {"publicUrl": ...}}``) or a small object
    with a ``public_url`` attribute in others. This normalises all of those to a
    string, returning ``""`` when nothing URL-shaped is found.
    """

    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        url = result.get("publicUrl") or result.get("public_url")
        if not url:
            data = result.get("data")
            if isinstance(data, dict):
                url = data.get("publicUrl") or data.get("public_url")
        return url or ""
    for attr in ("public_url", "publicUrl"):
        val = getattr(result, attr, None)
        if isinstance(val, str) and val:
            return val
    return ""


def upload_media(
    data: bytes,
    filename: str,
    content_type: str = "application/octet-stream",
) -> str:
    """Upload ``data`` into the public ``course-media`` bucket and return its URL.

    The object is stored under a unique key (a fresh ``uuid4`` prefix plus a
    sanitized ``filename``) so repeated uploads of the same name never collide,
    then its public URL is resolved and returned for embedding in a lesson.

    Best-effort: on a missing client, an upload failure, or an unresolvable URL,
    it logs and returns ``""`` rather than raising, so an authoring flow can
    continue. Defensive against supabase-py return-value quirks (upload result
    shapes and public-url shapes both vary across versions).
    """

    if not data:
        _LOG.warning("upload_media: empty data, nothing to upload.")
        return ""

    client = _client()
    if client is None:
        return ""

    path = f"{uuid.uuid4().hex}/{_sanitize_filename(filename)}"
    try:
        bucket = client.storage.from_(BUCKET)
        bucket.upload(
            path,
            data,
            {"content-type": content_type or "application/octet-stream", "upsert": "true"},
        )
    except Exception as exc:  # noqa: BLE001 - upload is best-effort.
        _LOG.warning("upload_media upload failed for %s: %s", path, exc)
        return ""

    try:
        url = _extract_public_url(bucket.get_public_url(path))
    except Exception as exc:  # noqa: BLE001 - URL resolution is best-effort.
        _LOG.warning("upload_media get_public_url failed for %s: %s", path, exc)
        return ""

    if not url:
        _LOG.warning("upload_media: could not resolve public URL for %s", path)
        return ""
    return url
