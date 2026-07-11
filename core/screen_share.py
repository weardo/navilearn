"""Browser screen-share capture as a Streamlit custom component.

Unlike :mod:`core.capture` (which grabs the *server's* display and is only
correct for a locally self-hosted run), this reads the *candidate's own screen*
from their browser via the ``getDisplayMedia`` API. The tiny no-build frontend
in ``components/screen_share/index.html`` requests screen sharing, grabs a frame
every few seconds, and hands it back to Python as a base64 JPEG data URL. This
is the honest capture path for a hosted deployment: the interviewer sees what the
candidate is actually looking at, on the candidate's machine.

Returns ``None`` until the user has granted sharing and a frame has arrived. On
error the frontend returns a string prefixed with ``"ERROR:"`` which the helpers
here translate to ``None`` plus a reason, so callers never crash.
"""

from __future__ import annotations

import base64
import binascii
import os
from typing import Optional

import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "components", "screen_share")

# Declared once at import time. ``path`` points at the static frontend directory.
_screen_share_component = components.declare_component("navilearn_screen_share", path=_COMPONENT_DIR)


def screen_share_widget(interval_ms: int = 5000, key: Optional[str] = None) -> Optional[str]:
    """Render the screen-share widget and return the latest frame data URL.

    ``interval_ms`` is how often the browser grabs a fresh frame. Returns the
    ``data:image/jpeg;base64,...`` string of the most recent frame, ``None`` if
    the user has not shared yet, or a string starting with ``"ERROR:"`` if the
    browser reported a problem (declined permission, unsupported, and so on).
    """

    return _screen_share_component(interval_ms=interval_ms, default=None, key=key)


def data_url_to_jpeg(data_url: Optional[str]) -> Optional[bytes]:
    """Decode a ``data:image/jpeg;base64,...`` string into raw JPEG bytes.

    Returns ``None`` for a missing value, an error sentinel, or malformed data,
    so a bad frame never breaks the interview flow.
    """

    if not data_url or not isinstance(data_url, str) or data_url.startswith("ERROR:"):
        return None
    if "," not in data_url:
        return None
    b64 = data_url.split(",", 1)[1]
    try:
        return base64.b64decode(b64)
    except (binascii.Error, ValueError):
        return None
