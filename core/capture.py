"""On-device capture for the AI Interviewer: grab the screen and the mic.

Challenge 1's interviewer runs on the candidate's own machine, so it needs two
local inputs: a screenshot of the shared screen (fed to :mod:`core.vision`) and
a short microphone recording of a spoken answer (fed to :mod:`core.stt` or
:mod:`core.sarvam`). This module wraps ``mss`` (screen) and ``sounddevice``
(mic) behind a tiny, typed surface.

Both capture functions raise :class:`CaptureError` with a clear, human message
when the underlying device is missing (no display, no input device, no
PortAudio) instead of surfacing a raw backend exception or hard-crashing. A UI
can catch it and fall back to pasted text, keeping the demo runnable headless.
The ``*_available`` probes answer the same question without side effects and
never raise, so a page can decide whether to show a capture button at all.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

# Default microphone sample rate (Hz). 16 kHz mono matches what the STT layer
# (Groq Whisper / Sarvam Saarika) expects, so no resample is needed downstream.
MIC_SAMPLE_RATE = 16_000


class CaptureError(RuntimeError):
    """Raised when a local capture device is unavailable or fails.

    Carries a message a UI can show directly. Callers are expected to catch
    this and offer a paste-text fallback rather than let it propagate.
    """


def _new_temp_path(suffix: str) -> str:
    """Return a fresh temp file path with ``suffix`` (the file is not opened)."""

    fd, path = tempfile.mkstemp(suffix=suffix, prefix="navilearn_capture_")
    import os

    os.close(fd)
    return path


# External screenshot tools we know how to drive, tried in this order by
# :func:`capture_screen`. Each entry is (binary name, builder). The builder maps
# an output PNG path to the argv that writes a full-screen shot to it.
#
# Order matters: native Wayland compositors expose ``grim`` (wlroots: Sway,
# Hyprland) first; ``gnome-screenshot`` covers GNOME; ``spectacle`` covers KDE
# Plasma; ``import`` (ImageMagick) is an X11 tool used before mss because it
# tolerates some setups mss's raw XGetImage does not. The freedesktop and GNOME
# Shell DBus portals are tried between the CLI tools and the mss fallback.
_SCREENSHOT_TOOLS: tuple[tuple[str, "object"], ...] = (
    ("grim", lambda out: ["grim", out]),
    ("gnome-screenshot", lambda out: ["gnome-screenshot", "-f", out]),
    # -b background, -n no notification, -o write to file.
    ("spectacle", lambda out: ["spectacle", "-b", "-n", "-o", out]),
    ("import", lambda out: ["import", "-silent", "-window", "root", out]),
)


def _is_valid_png(path: str) -> bool:
    """True if ``path`` exists and starts with the PNG magic signature."""

    try:
        if os.path.getsize(path) < 8:
            return False
        with open(path, "rb") as handle:
            return handle.read(8) == b"\x89PNG\r\n\x1a\n"
    except OSError:
        return False


def _screenshot_methods() -> list[str]:
    """Names of screen-capture methods that look usable on this machine.

    Cheap, side-effect-free probes only (``which`` for CLI tools, a DBus name
    check for the portals, monitor enumeration for mss). A name appearing here
    means the method is worth *trying*, not that a grab is guaranteed: some
    backends (notably mss under XWayland) enumerate a monitor yet fail the
    actual pixel grab, which :func:`capture_screen` handles by falling through.
    """

    methods: list[str] = []
    for name, _builder in _SCREENSHOT_TOOLS:
        if shutil.which(name):
            methods.append(name)
    if _portal_available():
        methods.append("portal")
    try:
        import mss

        with mss.mss() as sct:
            # monitors[0] is the virtual "all monitors" entry; a real primary
            # monitor means the list has at least two members.
            if len(sct.monitors) >= 2:
                methods.append("mss")
    except Exception:  # noqa: BLE001 - a probe must never raise.
        pass
    return methods


def _portal_available() -> bool:
    """True if a DBus screenshot portal looks reachable (best-effort probe)."""

    if not shutil.which("gdbus"):
        return False
    # A session bus is required for any portal call.
    return bool(os.environ.get("DBUS_SESSION_BUS_ADDRESS")) or os.path.exists(
        f"/run/user/{os.getuid()}/bus"
    )


def screen_available() -> bool:
    """True if any working screen-capture method is present on this machine.

    Detects the Wayland CLI tools (``grim``/``gnome-screenshot``/``spectacle``),
    ImageMagick's ``import``, a DBus screenshot portal, and the mss X11 backend.
    Never raises: any error is reported as ``False`` so a caller can hide the
    capture button safely.
    """

    try:
        return bool(_screenshot_methods())
    except Exception:  # noqa: BLE001 - a probe must never raise.
        return False


def mic_available() -> bool:
    """True if a default input (microphone) device is present.

    Never raises: missing PortAudio or no input device yields ``False``.
    """

    try:
        import sounddevice as sd

        devices = sd.query_devices()
        return any(
            int(device.get("max_input_channels", 0)) > 0 for device in devices
        )
    except Exception:  # noqa: BLE001 - a probe must never raise.
        return False


def _run_screenshot_tool(name, builder, target: str) -> bool:
    """Run one CLI screenshot tool; return True if it wrote a valid PNG.

    Never raises: a missing binary, non-zero exit, timeout, or an output file
    that is not a real PNG all yield ``False`` so :func:`capture_screen` can try
    the next method.
    """

    if not shutil.which(name):
        return False
    try:
        # Some tools refuse to overwrite an existing file; start clean.
        try:
            os.unlink(target)
        except OSError:
            pass
        proc = subprocess.run(
            builder(target),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
        return proc.returncode == 0 and _is_valid_png(target)
    except Exception:  # noqa: BLE001 - a failed tool must not raise.
        return False


def _capture_via_portal(target: str) -> bool:
    """Capture via a DBus screenshot portal (GNOME Shell or freedesktop).

    Tries the synchronous ``org.gnome.Shell.Screenshot`` interface (which writes
    straight to a file), then falls back to a best-effort freedesktop portal
    call. Returns True only when a valid PNG lands at ``target``. Never raises.
    """

    if not _portal_available():
        return False
    # GNOME Shell: Screenshot(include_cursor, flash, filename) -> (ok, path).
    try:
        os.unlink(target)
    except OSError:
        pass
    try:
        proc = subprocess.run(
            [
                "gdbus",
                "call",
                "--session",
                "--dest",
                "org.gnome.Shell.Screenshot",
                "--object-path",
                "/org/gnome/Shell/Screenshot",
                "--method",
                "org.gnome.Shell.Screenshot.Screenshot",
                "true",
                "false",
                target,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
        if proc.returncode == 0 and _is_valid_png(target):
            return True
    except Exception:  # noqa: BLE001 - fall through to the freedesktop portal.
        pass
    return False


def capture_screen(out_path: str | None = None) -> str:
    """Capture the primary screen to a PNG and return its path.

    Tries every known method in order so the same call works across desktops:
    the Wayland tools ``grim`` (wlroots), ``gnome-screenshot`` (GNOME) and
    ``spectacle`` (KDE), then ImageMagick's ``import``, then a DBus screenshot
    portal, and finally the mss X11 backend as a last resort. The first method
    that writes a valid PNG wins.

    Raises :class:`CaptureError` with a clear, human message when no method
    produces an image (for example headless, or Wayland with no screenshot tool
    installed), so a caller can fall back to pasted screen text.
    """

    target = out_path or _new_temp_path(".png")

    # 1) External CLI tools, in preference order (Wayland-first, then X11).
    for name, builder in _SCREENSHOT_TOOLS:
        if _run_screenshot_tool(name, builder, target):
            return target

    # 2) DBus screenshot portal (GNOME Shell / freedesktop).
    if _capture_via_portal(target):
        return target

    # 3) mss X11 fallback. Enumerates then grabs the primary monitor. On many
    #    Wayland setups this raises an X protocol error, which we swallow so the
    #    combined "no method worked" message below is what the caller sees.
    mss_error: str | None = None
    try:
        import mss
        import mss.tools

        with mss.mss() as sct:
            monitors = sct.monitors
            if len(monitors) >= 2:
                # monitors[1] is the primary physical monitor.
                shot = sct.grab(monitors[1])
                mss.tools.to_png(shot.rgb, shot.size, output=target)
                if _is_valid_png(target):
                    return target
    except Exception as exc:  # noqa: BLE001 - remember, then report uniformly.
        mss_error = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__

    detail = f" (last error: {mss_error})" if mss_error else ""
    raise CaptureError(
        "Screen capture is not available on this machine: no working method "
        "(grim, gnome-screenshot, spectacle, import, DBus portal, or mss) "
        f"produced an image{detail}. Are you headless, or is no screenshot "
        "tool installed? Paste the screen text instead."
    )


def record_mic(seconds: float = 6.0, out_path: str | None = None) -> str:
    """Record the default microphone to a 16 kHz mono WAV and return its path.

    Blocks for ``seconds`` while recording via ``sounddevice``. Raises
    :class:`CaptureError` with a clear message when no input device or
    PortAudio backend is available, so a caller can fall back to typed text.
    """

    target = out_path or _new_temp_path(".wav")
    try:
        import numpy as np
        import sounddevice as sd

        frames = int(seconds * MIC_SAMPLE_RATE)
        audio = sd.rec(
            frames,
            samplerate=MIC_SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        data = np.asarray(audio, dtype=np.int16)
        with wave.open(target, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # int16 -> 2 bytes/sample
            wav.setframerate(MIC_SAMPLE_RATE)
            wav.writeframes(data.tobytes())
        return target
    except CaptureError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize to a catchable error.
        raise CaptureError(
            "Microphone capture is not available on this machine "
            f"({exc}). Type your answer instead."
        ) from exc
