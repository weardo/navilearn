"""Robust extraction of JSON from noisy LLM output.

LLMs frequently wrap JSON in prose or Markdown fences. These helpers pull the
first balanced JSON object or array out of a string and parse it, returning a
fallback on any failure so callers never crash on malformed output.
"""

from __future__ import annotations

import json
from typing import Any


def _find_balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    """Return the first balanced ``open_ch``..``close_ch`` span, or None."""

    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json(text: str, default: Any = None) -> Any:
    """Extract and parse the first balanced JSON value from ``text``.

    Tries a direct parse first, then a fenced code block, then the first
    balanced object, then the first balanced array. Returns ``default`` if
    nothing parses.
    """

    if text is None:
        return default
    stripped = text.strip()

    # Direct parse.
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strip a Markdown code fence if present.
    if "```" in stripped:
        body = stripped.split("```", 2)
        if len(body) >= 2:
            candidate = body[1]
            if candidate.lower().startswith("json"):
                candidate = candidate[4:]
            candidate = candidate.strip()
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                pass

    # First balanced object, then array.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        span = _find_balanced(stripped, open_ch, close_ch)
        if span:
            try:
                return json.loads(span)
            except (json.JSONDecodeError, ValueError):
                continue

    return default
