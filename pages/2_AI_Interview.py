"""AI Interview: a driveable demo of the autonomous technical interviewer.

In the full NaviLearn product the interviewer (Challenge 1) runs on the
candidate's own machine, capturing their screen (OCR) and speech (STT) live and
defending their project in real time. That capture loop is out of scope for a
web demo, so this page exposes the same brain (:mod:`core.interview`) in a way a
reviewer can drive by hand: paste a project summary plus optional "screen text"
(what OCR would have read), then run an adaptive interview one question at a
time and score it against the rubric.

Every LLM call is gated behind an explicit button so the page loads instantly
(and spends nothing) under AppTest, which never clicks. This module holds no
interview logic of its own: it is a thin UI over
:class:`core.interview.InterviewEngine`.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

import streamlit as st

from core import capture, sarvam, stt, vision, voice_screen
from core.interview import (
    InterviewEngine,
    InterviewReport as RubricReport,
    Turn,
    run_scripted,
)
from core.notes import create_note
from core.repo import ActivityEvent, InterviewReport as ReportEntity
from core.sarvam import SarvamSTT, SarvamTTS
from core.session import current_user, get_repo_cached, require_user
from core.summarizer import get_summarizer

# How many question/answer turns a session may run before we force scoring.
_MAX_TURNS = 5

# Session-state keys, namespaced so they never collide with other pages.
_K_QUESTION = "iv_question"
_K_HISTORY = "iv_history"
_K_SUMMARY = "iv_summary"
_K_SCREEN = "iv_screen"
_K_REPORT = "iv_report"
_K_LANG = "iv_lang"

# Live (auto) mode state, namespaced apart from the manual flow above so the two
# modes never clobber each other's question/history/report.
_K_LIVE_QUESTION = "iv_live_question"
_K_LIVE_HISTORY = "iv_live_history"
_K_LIVE_SCREEN = "iv_live_screen"
_K_LIVE_REPORT = "iv_live_report"

# Conversational (browser) mode state. This is the FEATURED, honest-on-a-hosted-
# deploy flow: it reads the CANDIDATE's own browser screen-share and browser mic,
# not the server's display. All keys share the 'ivc_' prefix so they never
# collide with the manual (_K_*) or legacy live (_K_LIVE_*) state above.
_K_CV_STARTED = "ivc_started"
_K_CV_SUMMARY = "ivc_summary"
_K_CV_QUESTION = "ivc_question"
_K_CV_HISTORY = "ivc_history"
_K_CV_SCREEN_TEXT = "ivc_screen_text"
# Legacy hash keys from the previous st.audio_input flow, kept only so a reset
# clears them for sessions that still carry the old state.
_K_CV_SCREEN_HASH = "ivc_screen_hash"
_K_CV_AUDIO_HASH = "ivc_last_audio_hash"
_K_CV_TYPED = "ivc_typed"
_K_CV_REPORT = "ivc_report"
_K_CV_SPEAK = "ivc_speak"
# Hands-free voice + screen component state. ``_K_CV_LAST_SEQ`` is the last
# utterance sequence number we processed (the component increments seq per
# utterance, so a higher seq means a genuinely new answer, not the same one
# persisting across reruns). ``_K_CV_WAV`` caches the current question's
# synthesized speech so it survives reruns, and ``_K_CV_AUTOPLAY`` marks the one
# render on which that speech should auto-play.
_K_CV_LAST_SEQ = "ivc_last_seq"
_K_CV_WAV = "ivc_wav"
_K_CV_AUTOPLAY = "ivc_autoplay"

# Language choices for the Indic-voice layer. The label is what the selector
# shows; the value maps to the Sarvam BCP-47 code used for both TTS and STT.
_LANG_ENGLISH = "English"
_LANG_HINDI = "हिन्दी"
_LANG_OPTIONS = (_LANG_ENGLISH, _LANG_HINDI)


def _is_hindi() -> bool:
    """True when the learner picked Hindi in the language selector."""

    return st.session_state.get(_K_LANG) == _LANG_HINDI


def _tts_lang() -> str:
    """Sarvam target-language code for the current selection."""

    return "hi-IN" if _is_hindi() else "en-IN"


def _stt_lang() -> str:
    """Sarvam STT language hint for the current selection.

    Hindi is hinted explicitly; English leaves auto-detection ("unknown") so a
    learner who answers in a mix of English and Indic words still transcribes.
    """

    return "hi-IN" if _is_hindi() else "unknown"


def _first_line(text: str) -> str:
    """Return the first non-empty line of ``text``, trimmed, or a fallback."""

    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:120]
    return "Untitled project"


def _reset_session() -> None:
    """Clear all interview state so a fresh interview can start."""

    for key in (_K_QUESTION, _K_HISTORY, _K_REPORT):
        st.session_state.pop(key, None)


def _history() -> list[Turn]:
    """Return the running list of completed turns from session state."""

    hist = st.session_state.get(_K_HISTORY)
    return hist if isinstance(hist, list) else []


def _live_history() -> list[Turn]:
    """Return the running list of completed turns for the live (auto) mode."""

    hist = st.session_state.get(_K_LIVE_HISTORY)
    return hist if isinstance(hist, list) else []


def _read_screen_now(state_key: str) -> str:
    """Capture the screen right now, read it with vision, store and return it.

    This is the real Challenge 1 perception step: grab a fresh frame off the
    candidate's screen and turn it into grounding text. The description is
    written to ``state_key`` (used as the live screen context) and returned. Any
    missing display or empty reading degrades to an ``st.info`` note and an empty
    string, so the caller can fall back to whatever context it already has.
    """

    try:
        path = capture.capture_screen()
    except capture.CaptureError as exc:
        st.info(str(exc))
        return ""
    with st.spinner("Reading your screen with the vision model..."):
        description = vision.describe_screen(path)
    if description:
        st.session_state[state_key] = description
        return description
    st.info(
        "Captured the screen but the vision model returned nothing. The "
        "interview will continue on the previous context."
    )
    return ""


def _live_grab_answer(answer_key: str, seconds: float) -> None:
    """Fill ``answer_key`` from the mic for live mode, honouring the language.

    Routes to Sarvam Saarika for Hindi and Groq Whisper for English, matching
    the manual flow. A missing mic degrades to an ``st.info`` note so the
    candidate can type into the live answer box instead.
    """

    if _is_hindi():
        _record_indic_answer_into_state(answer_key, seconds)
    else:
        _record_answer_into_state(answer_key, seconds)


def _transcript_so_far(
    summary: str, history: list[Turn], current_q: str, current_a: str
) -> str:
    """Assemble the running interview transcript as plain interviewer/candidate text.

    Combines the project summary, every completed turn, and the in-progress
    question and answer into a single block the on-device summarizer can digest.
    """

    lines: list[str] = []
    if summary and summary.strip():
        lines.append(f"Project: {summary.strip()}")
    for turn in history:
        if turn.question:
            lines.append(f"Interviewer: {turn.question.strip()}")
        if turn.answer and turn.answer.strip():
            lines.append(f"Candidate: {turn.answer.strip()}")
    if current_q and current_q.strip():
        lines.append(f"Interviewer: {current_q.strip()}")
    if current_a and current_a.strip():
        lines.append(f"Candidate: {current_a.strip()}")
    return "\n".join(lines)


def _render_live_summary(transcript: str) -> None:
    """On-device live summary of the interview transcript so far (Ch2, no LLM).

    This is Challenge 2's stated purpose: real-time summaries for automated
    voice interviews. The quantized T5-small runs through onnxruntime with no
    Groq, no LLM and no rate limit. Gated behind a button so headless AppTest
    runs (which never click) skip the model load entirely.
    """

    if not (transcript and transcript.strip()):
        st.caption("Nothing to summarize yet: start the interview first.")
        return
    with st.spinner("Summarizing the transcript on-device (ONNX)..."):
        out = get_summarizer().summarize_timed(transcript)
    if out.text:
        st.info(out.text)
        m1, m2, m3 = st.columns(3)
        m1.metric("Model size", f"{out.model_size_mb:.0f} MB")
        m2.metric("Latency", f"{out.latency_ms:.0f} ms")
        m3.metric("Load time", f"{out.load_time_ms:.0f} ms")
    else:
        st.info(
            "The offline model is unavailable on this machine, so no on-device "
            "summary was produced."
        )


def _persist_report(
    user_id: str, summary: str, report: RubricReport
) -> None:
    """Save a scored report and log an 'interview' activity, best-effort.

    Failures here never break the UI: the on-screen report is the source of
    truth for the reviewer, and persistence only feeds the dashboard.
    """

    repo = get_repo_cached()
    scores = {
        "technical_depth": report.technical_depth,
        "clarity": report.clarity,
        "originality": report.originality,
        "implementation_understanding": report.implementation_understanding,
        "overall": report.overall,
    }
    try:
        repo.save_interview_report(
            ReportEntity(
                id="",
                student_id=user_id,
                project_title=_first_line(summary),
                scores=scores,
                feedback=report.feedback,
            )
        )
        repo.record_activity(
            ActivityEvent(
                id="",
                student_id=user_id,
                type="interview",
                payload={"overall": report.overall, "project": _first_line(summary)},
            )
        )
    except Exception as exc:  # noqa: BLE001 - persistence must not break the demo.
        st.info(f"Report shown below but could not be saved: {exc}")


def _report_markdown(report: RubricReport, project_title: str) -> str:
    """Assemble a Markdown body for the scored report, for saving as a Note.

    Includes the project title, every rubric score and the feedback text, plus
    any recorded strengths and gaps, so the saved note is a self-contained
    record of the interview outcome.
    """

    lines: list[str] = [
        f"# Interview: {project_title}",
        "",
        "## Scores",
        f"- Overall: {report.overall:.1f} / 10",
        f"- Technical depth: {report.technical_depth:.1f} / 10",
        f"- Clarity: {report.clarity:.1f} / 10",
        f"- Originality: {report.originality:.1f} / 10",
        f"- Implementation understanding: {report.implementation_understanding:.1f} / 10",
    ]
    if report.feedback and report.feedback.strip():
        lines += ["", "## Feedback", "", report.feedback.strip()]
    if report.strengths:
        lines += ["", "## Strengths", ""] + [f"- {item}" for item in report.strengths]
    if report.gaps:
        lines += ["", "## Gaps", ""] + [f"- {item}" for item in report.gaps]
    return "\n".join(lines)


def _render_save_to_notes(
    report: RubricReport, user, project_title: str, note_key: str
) -> None:
    """Render a 'Save to Notes' button that stores the report as a Markdown note.

    Builds the note body from the project title, each rubric score and the
    feedback text, then creates an owner-scoped note tagged with the
    ``ai-interview`` source. Best-effort: :func:`core.notes.create_note` already
    logs and swallows backend failures, so a click never crashes the page. Falls
    back to :func:`core.session.current_user` when no user is threaded in.
    """

    resolved = user if user is not None else current_user()
    if resolved is None:
        return
    if not st.button("Save to Notes", key=f"iv_save_notes_{note_key}"):
        return
    title = (project_title or "").strip() or "Untitled project"
    body = _report_markdown(report, title)
    try:
        create_note(resolved.id, f"Interview: {title}", body, source="ai-interview")
    except Exception as exc:  # noqa: BLE001 - saving must not crash the demo.
        st.info(f"Could not save to your Notes right now: {exc}")
        return
    st.success("Saved to your Notes.")


def _render_report(
    report: RubricReport,
    user=None,
    project_title: str = "",
    note_key: str = "report",
) -> None:
    """Render a scored :class:`RubricReport` as metrics, bars and notes.

    When a signed-in ``user`` is available, a 'Save to Notes' button is shown so
    the reviewer can persist the report as a Markdown study note. ``note_key``
    keeps that button's widget key unique across the several places a report can
    render in one run.
    """

    st.subheader("Scored report")
    st.metric("Overall", f"{report.overall:.1f} / 10")

    subs = [
        ("Technical depth", report.technical_depth),
        ("Clarity", report.clarity),
        ("Originality", report.originality),
        ("Implementation understanding", report.implementation_understanding),
    ]
    cols = st.columns(len(subs))
    for col, (label, value) in zip(cols, subs):
        col.metric(label, f"{value:.1f}")
    for label, value in subs:
        st.caption(label)
        st.progress(min(1.0, max(0.0, value / 10.0)))

    if report.feedback:
        st.markdown("**Feedback**")
        st.write(report.feedback)

    left, right = st.columns(2)
    with left:
        st.markdown("**Strengths**")
        if report.strengths:
            for item in report.strengths:
                st.markdown(f"- {item}")
        else:
            st.caption("None recorded.")
    with right:
        st.markdown("**Gaps**")
        if report.gaps:
            for item in report.gaps:
                st.markdown(f"- {item}")
        else:
            st.caption("None recorded.")

    _render_save_to_notes(report, user, project_title, note_key)


def _run_engine(fn, spinner: str):
    """Call an engine method under a spinner, surfacing errors inline.

    Returns the result, or ``None`` if the call raised (for example when no LLM
    provider is configured). The interview engine already degrades to sensible
    defaults, so a raise here is genuinely a configuration or network problem.
    """

    try:
        with st.spinner(spinner):
            return fn()
    except Exception as exc:  # noqa: BLE001 - report clearly, do not crash.
        st.error(
            "The interviewer could not reach the language model. Check the "
            f"configured LLM provider and try again. Details: {exc}"
        )
        return None


def _capture_screen_into_state() -> None:
    """Grab the primary screen, read it with vision, load it as screen text.

    Writes the vision description into ``_K_SCREEN`` (before the screen_text
    widget is instantiated this run, so Streamlit accepts the assignment). Any
    missing display or empty reading degrades to an ``st.info`` note.
    """

    try:
        path = capture.capture_screen()
    except capture.CaptureError as exc:
        st.info(str(exc))
        return
    with st.spinner("Reading your screen with the vision model..."):
        description = vision.describe_screen(path)
    if description:
        st.session_state[_K_SCREEN] = description
        st.success(
            "Read your screen. The description is loaded as screen text below "
            "and will drive the interview."
        )
    else:
        st.info(
            "Captured the screen but the vision model returned nothing. Paste "
            "the screen text below instead."
        )


def _speak_question(question: str) -> None:
    """Synthesize ``question`` with Sarvam Bulbul and play it inline.

    Uses the current language selection to pick the target voice. Any failure
    (missing key, network, rate limit) degrades to an ``st.info`` note and never
    crashes the interview.
    """

    try:
        with st.spinner("Synthesizing the question with Sarvam..."):
            wav = SarvamTTS().synthesize(question, lang=_tts_lang())
    except Exception as exc:  # noqa: BLE001 - voice is optional, never crash.
        st.info(f"Could not speak the question right now: {exc}")
        return
    if wav:
        st.audio(wav, format="audio/wav")
    else:
        st.info("The voice service returned no audio. Read the question above.")


def _record_indic_answer_into_state(answer_key: str, seconds: float) -> None:
    """Record the mic and transcribe it with Sarvam Saarika into the answer box.

    This is the Indic sibling of :func:`_record_answer_into_state`: it uses
    Sarvam speech-to-text (which handles Hindi and other Indian languages better
    than Groq Whisper) with a language hint from the current selection. A
    missing mic or empty transcript degrades to an ``st.info`` note so the
    learner can type instead.
    """

    try:
        with st.spinner(f"Recording {seconds:.0f}s from your microphone..."):
            wav_path = capture.record_mic(seconds)
    except capture.CaptureError as exc:
        st.info(str(exc))
        return
    with st.spinner("Transcribing your spoken answer with Sarvam..."):
        transcript = SarvamSTT().transcribe(wav_path, language_code=_stt_lang())
    if transcript:
        st.session_state[answer_key] = transcript
        st.success("Transcribed your spoken answer into the box below.")
    else:
        st.info(
            "Could not transcribe the recording with Sarvam. Type your answer "
            "below instead."
        )


def _record_answer_into_state(answer_key: str, seconds: float) -> None:
    """Record the mic, transcribe it, and load it into the answer box.

    Writes the transcript into ``answer_key`` before the answer widget is
    instantiated this run. A missing mic or failed transcription degrades to an
    ``st.info`` note so the reviewer can type instead.
    """

    try:
        with st.spinner(f"Recording {seconds:.0f}s from your microphone..."):
            wav = capture.record_mic(seconds)
    except capture.CaptureError as exc:
        st.info(str(exc))
        return
    transcript = _run_engine(
        lambda: stt.transcribe(wav), "Transcribing your spoken answer..."
    )
    if transcript:
        st.session_state[answer_key] = transcript
        st.success("Transcribed your spoken answer into the box below.")
    else:
        st.info(
            "Could not transcribe the recording. Type your answer below instead."
        )


def _reset_live() -> None:
    """Clear all live-mode state so a fresh live interview can start."""

    for key in (_K_LIVE_QUESTION, _K_LIVE_HISTORY, _K_LIVE_SCREEN, _K_LIVE_REPORT):
        st.session_state.pop(key, None)


def _render_live_interview(user, summary: str) -> None:
    """Real-time, autonomous interview driven by live screen + speech.

    This is the Challenge 1 loop as an actual loop rather than a turn-by-turn
    form. "Start live interview" captures the screen, reads it with the vision
    model, and asks a grounded opening question. Each "Capture screen + next
    question" press re-captures the (now changed) screen, transcribes the spoken
    answer, and asks the next adaptive question grounded in the FRESH screen and
    that answer, appending to the running history. "Finish + score" produces the
    rubric report.

    Every step is behind a button so headless AppTest (which never clicks) loads
    the page instantly and spends nothing. If the screen or mic is unavailable
    the mode degrades cleanly to pasted screen text and a typed answer box.
    """

    st.subheader("Legacy live interview (local self-host only)")
    st.caption(
        "Local self-host only: this captures the SERVER's display and mic, so on "
        "a hosted deploy it reads the host machine, not your screen. It is honest "
        "only when you run NaviLearn on your own machine; otherwise use the "
        "conversational mode at the top, which shares YOUR browser screen. Press "
        "Start and the interviewer captures the screen, reads it, and asks an "
        "opening question, then each round re-captures, listens to your spoken "
        "answer, and asks the next question. No display or mic (headless)? It "
        "falls back to the screen text above and a typed answer box."
    )

    screen_ok = capture.screen_available()
    mic_ok = capture.mic_available()
    cols = st.columns(2)
    cols[0].caption(
        "Screen capture: on" if screen_ok else "Screen capture: off (paste text)"
    )
    cols[1].caption(
        "Microphone: on" if mic_ok else "Microphone: off (type your answer)"
    )

    live_q = st.session_state.get(_K_LIVE_QUESTION)
    live_hist = _live_history()

    start_col, stop_col = st.columns(2)
    with start_col:
        start_live = st.button(
            "Start live interview", key="iv_live_start", width="stretch"
        )
    with stop_col:
        if st.button("Reset live", key="iv_live_reset", width="stretch"):
            _reset_live()
            st.rerun()

    if start_live:
        engine = InterviewEngine()
        # Seed the interviewer with a fresh screen read layered on the summary,
        # so the opening question is grounded in what is actually on screen.
        screen_desc = ""
        if screen_ok:
            screen_desc = _read_screen_now(_K_LIVE_SCREEN)
        if not screen_desc:
            screen_desc = st.session_state.get(_K_SCREEN, "")
        seed_parts = [p for p in (summary, screen_desc) if p and p.strip()]
        seed = "\n\n".join(seed_parts) or summary
        question = _run_engine(
            lambda: engine.opening_question(seed),
            "Preparing a grounded opening question...",
        )
        if question is not None:
            st.session_state[_K_LIVE_HISTORY] = []
            st.session_state[_K_LIVE_QUESTION] = question
            st.session_state.pop(_K_LIVE_REPORT, None)
            st.rerun()

    live_q = st.session_state.get(_K_LIVE_QUESTION)
    live_hist = _live_history()

    if live_q:
        turn_no = len(live_hist) + 1
        st.markdown(f"**Live question {turn_no} of up to {_MAX_TURNS}**")
        st.info(live_q)

        if st.button("🔊 Speak question", key=f"iv_live_speak_{len(live_hist)}"):
            _speak_question(live_q)

        # Mic buttons sit above the answer widget so a transcript can be written
        # to the answer key before that widget is instantiated this run.
        answer_key = f"iv_live_answer_{len(live_hist)}"
        if mic_ok:
            rec_col, sec_col = st.columns([3, 1])
            with sec_col:
                seconds = st.number_input(
                    "Record seconds",
                    min_value=2.0,
                    max_value=30.0,
                    value=8.0,
                    step=1.0,
                    key=f"iv_live_sec_{len(live_hist)}",
                )
            with rec_col:
                if st.button(
                    "🎤 Record spoken answer",
                    key=f"iv_live_rec_{len(live_hist)}",
                    width="stretch",
                ):
                    _live_grab_answer(answer_key, float(seconds))
        else:
            st.caption("No microphone detected: type your answer below.")

        answer = st.text_area(
            "Your answer (spoken transcript, or type here)",
            key=answer_key,
            height=110,
            placeholder="Explain the concrete choice on screen, not the textbook definition.",
        )

        # Live on-device summary of the running transcript (Ch2, no LLM).
        if st.button(
            "Live summary (on-device)", key=f"iv_live_sum_{len(live_hist)}"
        ):
            live_screen = st.session_state.get(_K_LIVE_SCREEN, "") or st.session_state.get(
                _K_SCREEN, ""
            )
            transcript = _transcript_so_far(
                _live_seed_text(summary, live_screen), live_hist, live_q, answer
            )
            _render_live_summary(transcript)

        at_cap = turn_no >= _MAX_TURNS
        next_col, finish_col = st.columns(2)
        with next_col:
            next_click = st.button(
                "Capture screen + next question",
                key=f"iv_live_next_{len(live_hist)}",
                width="stretch",
                disabled=at_cap,
            )
        with finish_col:
            finish_click = st.button(
                "Finish + score",
                key=f"iv_live_finish_{len(live_hist)}",
                width="stretch",
                type="primary",
            )
        if at_cap:
            st.caption("Turn limit reached: finish and score the live session.")

        if next_click:
            engine = InterviewEngine()
            # Re-capture the (now changed) screen so the follow-up is grounded in
            # the FRESH frame, not the seed. Fall back to the last known context.
            fresh = ""
            if screen_ok:
                fresh = _read_screen_now(_K_LIVE_SCREEN)
            live_screen = fresh or st.session_state.get(
                _K_LIVE_SCREEN, ""
            ) or st.session_state.get(_K_SCREEN, "")
            new_hist = live_hist + [Turn(question=live_q, answer=answer)]
            follow_up = _run_engine(
                lambda: engine.next_question(
                    new_hist, live_screen, latest_answer=answer
                ),
                "Reading the fresh screen and thinking of the next question...",
            )
            if follow_up is not None:
                st.session_state[_K_LIVE_HISTORY] = new_hist
                st.session_state[_K_LIVE_QUESTION] = follow_up
                st.rerun()

        if finish_click:
            engine = InterviewEngine()
            live_screen = st.session_state.get(_K_LIVE_SCREEN, "") or st.session_state.get(
                _K_SCREEN, ""
            )
            new_hist = live_hist + [Turn(question=live_q, answer=answer)]
            report = _run_engine(
                lambda: engine.score(new_hist, live_screen),
                "Scoring the live interview...",
            )
            if report is not None:
                st.session_state[_K_LIVE_HISTORY] = new_hist
                st.session_state[_K_LIVE_REPORT] = report
                st.session_state.pop(_K_LIVE_QUESTION, None)
                _persist_report(user.id, summary, report)
                st.rerun()

    live_report = st.session_state.get(_K_LIVE_REPORT)
    if isinstance(live_report, RubricReport):
        _render_report(
            live_report, user, _first_line(summary), note_key="iv_live"
        )
    elif not live_q:
        st.caption(
            "Press Start live interview to capture your screen and begin the "
            "autonomous, real-time interview."
        )


def _live_seed_text(summary: str, screen: str) -> str:
    """Combine summary + live screen context for the live-summary transcript."""

    parts = [p for p in (summary, screen) if p and p.strip()]
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Conversational (browser) interview: the featured, hosted-honest flow.
#
# This reads the candidate's OWN browser screen-share (via the getDisplayMedia
# component in core.screen_share) and their OWN browser microphone (st.audio_
# input), then advances like a real conversation: whenever the candidate finishes
# speaking (a new recording arrives), it transcribes, appends the turn, and asks
# the next grounded question, speaking it aloud. No per-question clicking.
#
# Everything is gated behind an explicit Start button plus real browser inputs
# (a shared frame, a recording, a typed answer), so a headless AppTest run (which
# never shares, records, or clicks) loads instantly and spends nothing: it never
# reaches capture, vision, STT, TTS, or the interview LLM.
# --------------------------------------------------------------------------- #


def _conv_history() -> list[Turn]:
    """Return the running list of completed turns for the conversational mode."""

    hist = st.session_state.get(_K_CV_HISTORY)
    return hist if isinstance(hist, list) else []


def _reset_conv() -> None:
    """Clear conversational-mode state so a fresh browser interview can start."""

    for key in (
        _K_CV_STARTED,
        _K_CV_QUESTION,
        _K_CV_HISTORY,
        _K_CV_SCREEN_TEXT,
        _K_CV_SCREEN_HASH,
        _K_CV_AUDIO_HASH,
        _K_CV_REPORT,
        _K_CV_SPEAK,
        _K_CV_LAST_SEQ,
        _K_CV_WAV,
        _K_CV_AUTOPLAY,
    ):
        st.session_state.pop(key, None)


def _describe_jpeg(jpeg: bytes) -> str:
    """Read one browser screen frame with the vision model, as plain text.

    Writes the JPEG bytes to a temporary ``.jpg`` file (what
    :func:`core.vision.describe_screen` expects) and returns the description.
    Any failure degrades to ``""`` so a bad frame never breaks the interview.
    """

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as fh:
            fh.write(jpeg)
            tmp_path = fh.name
        return vision.describe_screen(tmp_path) or ""
    except Exception:  # noqa: BLE001 - vision is best-effort, never crash a frame.
        return ""
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _synth_question_wav(question: str) -> Optional[bytes]:
    """Synthesize ``question`` with Sarvam and return the WAV bytes, or ``None``.

    Honours the current voice-language selection. Voice is optional: any failure
    (missing key, network, rate limit) degrades to a small caption and returns
    ``None`` so the conversation continues on the written question alone.
    """

    try:
        wav = sarvam.synthesize(question, lang=_tts_lang())
    except Exception as exc:  # noqa: BLE001 - voice is optional, never crash.
        st.caption(f"Voice unavailable right now: {exc}")
        return None
    return wav or None


def _transcribe_audio_bytes(data: bytes) -> str:
    """Transcribe raw browser-mic bytes to text, Groq first then Sarvam.

    The bytes are written to a temporary ``.webm`` file (the container the
    hands-free component records; ffmpeg inside :func:`core.stt.transcribe` reads
    it regardless of suffix). Groq Whisper is tried first; if it yields nothing,
    Sarvam Saarika is the fallback for Indic speech. Returns ``""`` when neither
    produces a transcript.
    """

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as fh:
            fh.write(data)
            tmp_path = fh.name
        text = ""
        try:
            text = stt.transcribe(tmp_path) or ""
        except Exception:  # noqa: BLE001 - fall through to the Indic path.
            text = ""
        if not text.strip():
            try:
                text = sarvam.transcribe(tmp_path, language_code=_stt_lang()) or ""
            except Exception:  # noqa: BLE001 - both paths best-effort.
                text = ""
        return text.strip()
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _conv_advance(summary: str, answer_text: str) -> None:
    """Record the answer to the current question and ask the next one.

    Appends ``Turn(current_question, answer_text)`` to the running history, then
    asks an adaptive follow-up grounded in the latest cached screen reading and
    that answer, speaking it aloud on the next render. The history and question
    are only committed if the follow-up call succeeds, so a transient LLM error
    leaves the candidate on the same question to retry. At the turn cap the
    follow-up is skipped and the flow invites the candidate to finish and score.
    """

    history = _conv_history()
    question = st.session_state.get(_K_CV_QUESTION, "")
    new_hist = history + [Turn(question=question, answer=answer_text)]

    if len(new_hist) >= _MAX_TURNS:
        st.session_state[_K_CV_HISTORY] = new_hist
        st.session_state[_K_CV_QUESTION] = ""
        st.rerun()
        return

    engine = InterviewEngine()
    screen_text = st.session_state.get(_K_CV_SCREEN_TEXT, "")
    follow_up = _run_engine(
        lambda: engine.next_question(new_hist, screen_text, latest_answer=answer_text),
        "Listening, and thinking of the next question...",
    )
    if follow_up is None:
        return
    st.session_state[_K_CV_HISTORY] = new_hist
    st.session_state[_K_CV_QUESTION] = follow_up
    st.session_state[_K_CV_SPEAK] = True
    st.rerun()


def _conv_process_utterance(value: dict, summary: str) -> None:
    """Advance the interview when a NEW hands-free utterance arrives.

    ``value`` is the component payload
    (``{"seq": int, "audio": data-url, "screen": data-url, "error"?: str}``).
    Utterances are keyed by ``seq``: only a ``seq`` greater than the last one we
    processed triggers work, so the same utterance persisting across reruns is
    never transcribed twice, and ``seq == -1`` surfaces a capture error. The seq
    is marked processed before transcription so a failed transcript does not loop.

    Each utterance carries a fresh screen frame grabbed at the same instant: it
    is read with the vision model and cached as the interview's grounding text so
    the next question can refer to what is on the candidate's screen right now.
    The transcript then advances the conversation exactly as a typed answer would.
    """

    seq = value.get("seq")
    if not isinstance(seq, int):
        return
    if seq == -1:
        reason = str(value.get("error") or "").strip() or "mic or screen access failed"
        st.caption(f"Voice capture unavailable ({reason}). Use the typed fallback below.")
        return
    if seq <= 0 or seq == st.session_state.get(_K_CV_LAST_SEQ):
        return
    st.session_state[_K_CV_LAST_SEQ] = seq

    # Read the screen frame captured with this utterance and cache the grounding
    # text, so the follow-up references what is currently on screen.
    jpeg = voice_screen.decode_screen(value.get("screen"))
    if jpeg:
        description = _describe_jpeg(jpeg)
        if description:
            st.session_state[_K_CV_SCREEN_TEXT] = description

    audio_bytes = voice_screen.decode_audio(value.get("audio"))
    if not audio_bytes:
        st.info("Could not read that recording. Speak again, or type your answer below.")
        return
    transcript = _transcribe_audio_bytes(audio_bytes)
    if not transcript:
        st.info(
            "Could not transcribe that. Speak again, or type your answer below."
        )
        return
    _conv_advance(summary, transcript)


def _render_conv_transcript(history: list[Turn], pending_question: str = "") -> None:
    """Render the full running interview as a persistent chat log.

    Each completed turn renders the interviewer's question in an assistant chat
    bubble followed by the candidate's answer in a user chat bubble, so the
    conversation reads top to bottom like a real transcript. When a question is
    awaiting an answer, ``pending_question`` is appended as a trailing assistant
    bubble so the candidate always sees the live conversation above the answer
    control. The log is derived from ``_K_CV_HISTORY`` plus the current question
    in session state, so it persists across the reruns that each recording,
    screen frame and answer trigger.
    """

    if not history and not pending_question:
        return
    st.markdown("#### Conversation")
    for turn in history:
        if turn.question:
            with st.chat_message("assistant"):
                st.markdown(turn.question)
        answered = turn.answer.strip() if turn.answer and turn.answer.strip() else ""
        with st.chat_message("user"):
            st.markdown(answered or "_(no answer)_")
    if pending_question:
        with st.chat_message("assistant"):
            st.markdown(pending_question)


def _conv_finish(user, summary: str) -> None:
    """Score the conversation, render it, persist it, and end the session.

    Scoring uses the completed turns (every asked-and-answered pair) plus the
    latest screen reading. The report is persisted best-effort via the shared
    path so it appears on the dashboard 'My interviews' card, then the current
    question is cleared so the finished report renders in place.
    """

    engine = InterviewEngine()
    history = _conv_history()
    screen_text = st.session_state.get(_K_CV_SCREEN_TEXT, "")
    report = _run_engine(
        lambda: engine.score(history, screen_text),
        "Scoring your interview...",
    )
    if report is not None:
        st.session_state[_K_CV_REPORT] = report
        st.session_state[_K_CV_QUESTION] = ""
        _persist_report(user.id, summary, report)
        st.rerun()


def _lock_navigation() -> None:
    """Hide the sidebar page nav while an interview is in progress.

    Navigating to another page tears down the live mic and screen component (its
    media streams stop and the conversation state is lost). Hiding the multipage
    nav links keeps the candidate on the page, and a warning explains why. The
    lock is only applied during an active interview and lifts once it is scored.
    """

    st.markdown(
        '<style>[data-testid="stSidebarNav"]{display:none;}</style>',
        unsafe_allow_html=True,
    )
    st.warning(
        "Interview in progress. Do not switch pages or refresh: that ends the "
        "live session. Use Finish and score when you are done."
    )


def _render_conversational_interview(user) -> None:
    """The featured, hands-free conversational interview over the browser.

    Setup asks for a project summary and one Start press. From there the
    hands-free voice + screen component takes over: the candidate clicks "Enable
    mic and screen" once, and then just talks. When they go quiet the component
    hands back one utterance (their speech plus a screen frame grabbed at that
    instant); the interviewer transcribes the answer, reads the screen, asks and
    speaks the next question, and resumes listening, all without a per-answer
    click. A typed-answer fallback stays available for anyone without a working
    mic. Finish and score writes a rubric report to the dashboard.

    Everything is gated behind Start plus real browser input, so a headless
    AppTest run (which never clicks Start nor enables the component) loads the
    page instantly and never reaches capture, vision, STT, TTS, or the LLM.
    """

    st.subheader("Conversational interview")
    st.caption(
        "The honest hosted flow: your own browser screen and microphone. Press "
        "Start, click Enable mic and screen once, then just talk. The "
        "interviewer listens, reads your screen, and asks the next question on "
        "its own the moment you pause, so you barely touch the keyboard. No "
        "working mic? A typed-answer fallback stays available."
    )

    # Lock navigation while a started interview has not yet been scored, so
    # switching pages cannot kill the live mic and screen mid-session.
    interview_active = bool(st.session_state.get(_K_CV_STARTED)) and not isinstance(
        st.session_state.get(_K_CV_REPORT), RubricReport
    )
    if interview_active:
        _lock_navigation()

    if not st.session_state.get(_K_CV_STARTED):
        if _K_CV_SUMMARY not in st.session_state:
            st.session_state[_K_CV_SUMMARY] = st.session_state.get(_K_SUMMARY, "")
        st.text_area(
            "Project summary",
            key=_K_CV_SUMMARY,
            height=140,
            placeholder=(
                "A CLI todo app in Python. Tasks persist to a JSON file; commands "
                "are add, list, done. I used argparse for parsing and dataclasses "
                "for the Task model."
            ),
        )
        st.caption(
            "When you start, the interviewer asks an opening question aloud. Click "
            "Enable mic and screen once in the panel that appears, then answer by "
            "speaking; it goes hands-free from there."
        )
        if st.button(
            "Start conversational interview",
            key="ivc_start",
            type="primary",
            width="stretch",
        ):
            summary = st.session_state.get(_K_CV_SUMMARY, "")
            engine = InterviewEngine()
            question = _run_engine(
                lambda: engine.opening_question(summary),
                "Preparing your opening question...",
            )
            if question is not None:
                st.session_state[_K_CV_STARTED] = True
                st.session_state[_K_CV_HISTORY] = []
                st.session_state[_K_CV_QUESTION] = question
                st.session_state[_K_CV_SPEAK] = True
                st.session_state[_K_CV_LAST_SEQ] = 0
                st.session_state.pop(_K_CV_REPORT, None)
                st.session_state.pop(_K_CV_WAV, None)
                st.session_state.pop(_K_CV_AUTOPLAY, None)
                st.session_state.pop(_K_CV_SCREEN_TEXT, None)
                st.rerun()

        prior = st.session_state.get(_K_CV_REPORT)
        if isinstance(prior, RubricReport):
            _render_report(
                prior,
                user,
                _first_line(st.session_state.get(_K_CV_SUMMARY, "")),
                note_key="ivc_prior",
            )
        return

    # --- Active conversation. ---------------------------------------------- #
    summary = st.session_state.get(_K_CV_SUMMARY, "")
    question = st.session_state.get(_K_CV_QUESTION, "")
    history = _conv_history()
    has_report = isinstance(st.session_state.get(_K_CV_REPORT), RubricReport)

    # Whether the interviewer is producing its line this render. On that render we
    # pause the mic (allow=False) so the component does not capture the
    # interviewer's synthesized voice as an answer; on every other render the mic
    # listens (allow=True). This flag is set whenever a new question is generated
    # (the opening and each follow-up) and cleared once we have synthesized it.
    speak_now = bool(st.session_state.get(_K_CV_SPEAK))

    # The hands-free voice + screen component: one click enables mic and screen,
    # then it returns an utterance whenever the candidate goes quiet.
    value = voice_screen.voice_screen_widget(
        allow=not speak_now,
        silence_ms=1100,
        speech_rms=0.02,
        key="ivc_voice",
    )

    # Prominently surface the latest reading of the candidate's screen so they can
    # see what the interviewer is grounding on this turn.
    screen_text = st.session_state.get(_K_CV_SCREEN_TEXT, "")
    if screen_text:
        st.info(f"On your screen I can see: {screen_text}")

    # Persistent running chat log, always visible ABOVE the answer control so the
    # candidate watches the conversation grow turn by turn. The pending question
    # is rendered as the trailing interviewer bubble.
    _render_conv_transcript(history, pending_question=question)

    if question and speak_now:
        # Speak-prep render: synthesize the question while the mic is paused
        # (allow=False above), cache the audio, then rerun so it auto-plays on the
        # next render with the mic listening again. Synthesizing here (not on the
        # playback render) keeps the audio out of the pause window and avoids
        # leaving the mic paused with no way to resume.
        st.session_state[_K_CV_WAV] = _synth_question_wav(question)
        st.session_state[_K_CV_AUTOPLAY] = True
        st.session_state.pop(_K_CV_SPEAK, None)
        st.rerun()

    if question:
        turn_no = len(history) + 1
        st.caption(
            f"Question {turn_no} of up to {_MAX_TURNS}. Just speak your answer; "
            "the interviewer replies the moment you pause."
        )

        # Play the current question aloud once (Sarvam TTS, cached across reruns).
        wav = st.session_state.get(_K_CV_WAV)
        if wav:
            st.audio(
                wav,
                format="audio/wav",
                autoplay=bool(st.session_state.pop(_K_CV_AUTOPLAY, False)),
            )

        # Advance whenever a NEW utterance arrives (and the mic was listening).
        if not speak_now and isinstance(value, dict):
            _conv_process_utterance(value, summary)

        # Typed-answer fallback for anyone without a working microphone.
        with st.form("ivc_typed_form", clear_on_submit=True):
            typed = st.text_input(
                "Prefer to type? Enter your answer and press Send",
                key=_K_CV_TYPED,
                placeholder="Explain the concrete choice on screen, not the textbook definition.",
            )
            sent = st.form_submit_button("Send answer")
        if sent and typed and typed.strip():
            _conv_advance(summary, typed.strip())
    elif history and not has_report:
        st.info(
            "You have answered the maximum number of questions. Finish and score "
            "when you are ready."
        )

    if history and not has_report:
        if st.button(
            "Finish and score",
            key="ivc_finish",
            type="primary",
            width="stretch",
        ):
            _conv_finish(user, summary)

    report = st.session_state.get(_K_CV_REPORT)
    if isinstance(report, RubricReport):
        _render_report(report, user, _first_line(summary), note_key="ivc")

    if st.button("Start a new conversational interview", key="ivc_reset"):
        _reset_conv()
        st.rerun()


def main() -> None:
    """Render the AI Interview page."""

    st.set_page_config(page_title="AI Interview", page_icon="🎤", layout="wide")

    user = require_user()

    st.title("AI Interview")
    st.caption(
        "A live project defense with an AI interviewer. The featured mode below "
        "reads your OWN browser screen-share and microphone and advances like a "
        "real conversation. A manual, step-by-step mode is kept below it as a "
        "fallback, and a legacy self-host mode (server-side capture) is tucked "
        "away for people running NaviLearn on their own machine."
    )

    # --- Indic voice: language selection drives TTS and Sarvam STT. --------- #
    # A NavGurukul-style differentiator for regional learners. The choice is
    # stored in session state (via the widget key) and read by the speak and
    # Indic-record helpers below. It costs nothing until a voice button is
    # pressed, so AppTest (which never clicks) stays fast and offline.
    st.radio(
        "Voice language",
        options=_LANG_OPTIONS,
        key=_K_LANG,
        horizontal=True,
        help=(
            "Pick हिन्दी to hear questions spoken and answer aloud in Hindi via "
            "Sarvam. English keeps the existing text and Whisper flow."
        ),
    )

    # --- Featured mode: the conversational, browser-driven interview. ------- #
    # This is the honest hosted flow (candidate's own screen + mic), so it leads
    # the page. It is fully self-contained and gated behind Start plus real
    # browser inputs, so headless AppTest never spends here.
    _render_conversational_interview(user)

    st.divider()
    st.markdown("### Manual interview (paste and step through)")
    st.caption(
        "The fallback flow: paste a project summary and, optionally, some screen "
        "text (what OCR would read from an editor or running app), then run an "
        "adaptive interview one question at a time and score it against the "
        "rubric. Use this when browser screen share or mic is not available."
    )

    summary = st.text_area(
        "Project summary",
        value=st.session_state.get(_K_SUMMARY, ""),
        key=_K_SUMMARY,
        height=140,
        placeholder=(
            "A CLI todo app in Python. Tasks persist to a JSON file; commands "
            "are add, list, done. I used argparse for parsing and dataclasses "
            "for the Task model."
        ),
    )
    # --- Live capture: the real Challenge 1 loop, gated behind buttons. ----- #
    # These buttons must sit above the screen_text widget so a capture can
    # write _K_SCREEN before that widget is instantiated this run.
    with st.expander(
        "Server-side screen capture (local self-host only)", expanded=False
    ):
        st.caption(
            "Local self-host only: this grabs the SERVER's display and mic, not "
            "your browser's, so on a hosted deploy it would read the machine "
            "NaviLearn runs on, not your screen. It is honest only when you run "
            "NaviLearn on your own machine. For a hosted deploy use the "
            "conversational mode at the top, which shares YOUR browser screen. "
            "If a display or mic is missing (for example headless), use the text "
            "boxes instead."
        )
        if capture.screen_available():
            if st.button("Capture my screen", key="iv_cap_screen"):
                _capture_screen_into_state()
        else:
            st.info(
                "No display detected (headless). Paste the screen text below "
                "to simulate what OCR would read."
            )

    screen_text = st.text_area(
        "Screen text (optional, what OCR would capture)",
        value=st.session_state.get(_K_SCREEN, ""),
        key=_K_SCREEN,
        height=120,
        placeholder="def add(task: str) -> None: ...  # visible source, README, or app UI",
    )

    # --- Legacy live (auto) mode: server-side capture, self-host only. ------ #
    with st.expander(
        "Legacy live interview (local self-host only): server-side screen + mic",
        expanded=False,
    ):
        _render_live_interview(user, summary)

    st.divider()
    st.markdown("### Manual interview (step by step)")

    start_col, demo_col = st.columns(2)
    with start_col:
        start = st.button("Start / restart interview", width="stretch")
    with demo_col:
        demo = st.button("Run scripted demo", width="stretch")

    st.divider()

    # --- One-click scripted demo: full session + report in a single click. -- #
    if demo:
        _reset_session()
        sample_summary = (
            "A CLI todo app in Python. Tasks persist to a JSON file; commands "
            "are add, list and done. Parsing uses argparse and the Task model "
            "is a dataclass."
        )
        sample_screen = (
            "class Task:\n    title: str\n    done: bool = False\n\n"
            "def save(tasks): json.dump([asdict(t) for t in tasks], open(PATH,'w'))"
        )
        qa_pairs = [
            ("design", "I keep tasks in a list of Task dataclasses and serialise them to JSON on every change so nothing is lost between runs."),
            ("tradeoff", "Rewriting the whole file each time is simple but O(n); for a personal todo list the file is tiny so it is fine, and I avoid partial-write corruption by writing then renaming."),
            ("edge case", "If the JSON file is missing or corrupt I catch the error on load and start from an empty list, then the next save recreates a clean file."),
        ]
        report = _run_engine(
            lambda: run_scripted(sample_summary, qa_pairs, sample_screen),
            "Running scripted interview and scoring...",
        )
        if report is not None:
            st.success("Scripted interview complete.")
            _persist_report(user.id, sample_summary, report)
            _render_report(
                report, user, _first_line(sample_summary), note_key="iv_demo"
            )
        return

    # --- Manual flow: start, then answer question by question. -------------- #
    if start:
        engine = InterviewEngine()
        question = _run_engine(
            lambda: engine.opening_question(summary),
            "Preparing the opening question...",
        )
        if question is not None:
            st.session_state[_K_HISTORY] = []
            st.session_state[_K_QUESTION] = question
            st.session_state.pop(_K_REPORT, None)

    current: Optional[str] = st.session_state.get(_K_QUESTION)
    history = _history()

    if current:
        turn_no = len(history) + 1
        st.markdown(f"**Question {turn_no} of up to {_MAX_TURNS}**")
        st.info(current)

        # Speak the question aloud (Sarvam Bulbul, gated so AppTest never spends).
        if st.button(
            "🔊 Speak question",
            key=f"iv_speak_{len(history)}",
        ):
            _speak_question(current)

        # Mic capture sits above the answer widget so a transcript can be
        # written to answer_key before that widget is instantiated this run.
        answer_key = f"iv_answer_{len(history)}"
        if capture.mic_available():
            rec_col, sec_col = st.columns([3, 1])
            with sec_col:
                seconds = st.number_input(
                    "Record seconds",
                    min_value=2.0,
                    max_value=30.0,
                    value=6.0,
                    step=1.0,
                    key=f"iv_sec_{len(history)}",
                )
            with rec_col:
                # English path (Groq Whisper) and Indic path (Sarvam Saarika)
                # both feed the same answer box; the learner picks by button.
                eng_col, indic_col = st.columns(2)
                with eng_col:
                    if st.button(
                        "Record spoken answer",
                        key=f"iv_rec_{len(history)}",
                        width="stretch",
                    ):
                        _record_answer_into_state(answer_key, float(seconds))
                with indic_col:
                    if st.button(
                        "🎤 Record answer (Indic)",
                        key=f"iv_rec_indic_{len(history)}",
                        width="stretch",
                    ):
                        _record_indic_answer_into_state(
                            answer_key, float(seconds)
                        )
        else:
            st.caption("No microphone detected: type your answer below.")

        answer = st.text_area(
            "Your answer",
            key=answer_key,
            height=120,
            placeholder="Explain the concrete choice, not the textbook definition.",
        )

        # On-device live summary of the transcript so far (Ch2, no Groq/LLM).
        # This is exactly Challenge 2's purpose: real-time summaries for
        # automated voice interviews. Gated behind a button so AppTest stays
        # fast (it never clicks, so the ONNX model is never loaded).
        if st.button(
            "Live summary (on-device)",
            key=f"iv_live_summary_{len(history)}",
        ):
            transcript = _transcript_so_far(summary, history, current, answer)
            _render_live_summary(transcript)

        at_cap = turn_no >= _MAX_TURNS
        next_col, finish_col = st.columns(2)
        with next_col:
            next_click = st.button(
                "Next question", width="stretch", disabled=at_cap
            )
        with finish_col:
            finish_click = st.button(
                "Finish + score", width="stretch", type="primary"
            )

        if at_cap:
            st.caption("Turn limit reached: finish and score the session.")

        if next_click:
            engine = InterviewEngine()
            history = history + [Turn(question=current, answer=answer)]
            follow_up = _run_engine(
                lambda: engine.next_question(
                    history, screen_text, latest_answer=answer
                ),
                "Thinking of a follow-up...",
            )
            if follow_up is not None:
                st.session_state[_K_HISTORY] = history
                st.session_state[_K_QUESTION] = follow_up
                st.rerun()

        if finish_click:
            engine = InterviewEngine()
            history = history + [Turn(question=current, answer=answer)]
            report = _run_engine(
                lambda: engine.score(history, screen_text),
                "Scoring the interview...",
            )
            if report is not None:
                st.session_state[_K_HISTORY] = history
                st.session_state[_K_REPORT] = report
                st.session_state.pop(_K_QUESTION, None)
                _persist_report(user.id, summary, report)
                st.rerun()

    # --- Show the last scored report (if any) below the flow. --------------- #
    report = st.session_state.get(_K_REPORT)
    if isinstance(report, RubricReport):
        _render_report(report, user, _first_line(summary), note_key="iv_manual")
    elif not current:
        st.caption(
            "Paste a project summary above and press Start, or click Run "
            "scripted demo to see a full scored report in one click."
        )


main()
