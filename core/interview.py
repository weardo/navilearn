"""Autonomous AI interviewer engine (Challenge 1 core logic).

This is the brain of the on-device technical interviewer, not the capture loop.
It is driven by two multimodal streams that the host app supplies as plain text:

- SCREEN text: what is visible on the candidate's screen (OCR of code, diagrams,
  a running app, a README), giving the interviewer real context to ground on.
- SPEECH text: the candidate's spoken answers (STT), one turn at a time.

The engine reasons over that context to ask a context-aware opening question,
then adaptive follow-ups that probe depth, and finally scores the whole session
against a rubric using an LLM judge. Every LLM call goes through the shared
gateway (core.llm) so the same code runs on Groq, Sarvam, Ollama, or the mock.

The engine never re-implements parsing or STT: it consumes text the pipeline
already produced. It is deliberately robust to messy model output, degrading to
sensible defaults rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.config import Settings, get_settings
from core.jsonutil import extract_json
from core.llm import JUDGE_MARKER, LLMClient, get_llm

# Keep prompts modest: cap how much captured context we forward per call.
_MAX_SCREEN_CHARS = 4000
_MAX_HISTORY_TURNS = 8


@dataclass
class Turn:
    """One question/answer exchange in the interview."""

    question: str
    answer: str


@dataclass
class InterviewReport:
    """Rubric scoring of a completed interview. Each score is on a 0..10 scale."""

    technical_depth: float
    clarity: float
    originality: float
    implementation_understanding: float
    overall: float
    feedback: str
    strengths: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)


_INTERVIEWER_SYSTEM = (
    "You are a senior technical interviewer conducting a live project defense. "
    "You can see the candidate's screen and hear their answers. You ask one "
    "sharp, specific question at a time, grounded in what is actually on screen "
    "and in what the candidate just said. You probe for real understanding: "
    "design trade-offs, edge cases, why-not-alternatives, and implementation "
    "detail. You never lecture and never ask more than one question at once. "
    "Reply with the question text only, no preamble and no numbering."
)

_OPENING_PROMPT = """A candidate is about to defend the project shown on their screen.

Project / screen summary:
\"\"\"
{summary}
\"\"\"

Ask a single, specific opening question that shows you have looked at their
work and invites them to explain a real, concrete part of it. Question only."""

_FOLLOWUP_PROMPT = """You are mid-interview. Below is the conversation so far, the
candidate's current screen, and their most recent answer.

Conversation so far:
{history}

Current screen text:
\"\"\"
{screen}
\"\"\"

Candidate's latest answer:
\"\"\"
{answer}
\"\"\"

Ask ONE adaptive follow-up question. It must build on the latest answer and stay
grounded in the screen content, pushing for deeper technical detail (a trade-off,
an edge case, a specific implementation choice, or a claim that needs evidence).
If the answer was vague or wrong, zero in on that gap.

When the current screen text above is non-empty you can see the candidate's
screen, so you MUST refer to something specific you actually see on it in your
question (a named function, file, class, error, or piece of UI), making it clear
you are looking at their real work right now. Question only."""

_JUDGE_SYSTEM = (
    "You are a rigorous, fair technical-interview evaluator. You grade only on "
    "the evidence in the transcript. You reply with a single JSON object and no "
    "other text."
)

_JUDGE_PROMPT = """Evaluate the candidate's performance in the technical interview
below. Use the screen context to judge whether their answers are accurate and
grounded in their actual work.

Screen context:
\"\"\"
{screen}
\"\"\"

Interview transcript:
{transcript}

Scoring rubric (each 0 to 10, where 0 is no evidence and 10 is exceptional):
- technical_depth: correctness and sophistication of the technical reasoning.
- clarity: how clearly and precisely the candidate communicates.
- originality: independent thought, insight, non-boilerplate reasoning.
- implementation_understanding: genuine grasp of how their own project works,
  consistent with what is on screen (penalise answers that do not match the code).

{marker}: Return ONLY a JSON object with EXACTLY these keys and no others:
{{
  "technical_depth": <number 0..10>,
  "clarity": <number 0..10>,
  "originality": <number 0..10>,
  "implementation_understanding": <number 0..10>,
  "overall": <number 0..10>,
  "feedback": "<2 to 4 sentence overall assessment>",
  "strengths": ["<short strength>", ...],
  "gaps": ["<short gap or area to improve>", ...]
}}
Do not wrap the JSON in prose or code fences."""


def _clamp_score(value: object, default: float = 0.0) -> float:
    """Coerce ``value`` to a float clamped into the 0..10 rubric range."""

    try:
        score = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if score != score:  # NaN guard.
        return default
    return max(0.0, min(10.0, score))


def _clamp_str_list(value: object) -> list[str]:
    """Coerce ``value`` into a clean list of non-empty strings."""

    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _truncate_screen(screen_text: str) -> str:
    text = (screen_text or "").strip()
    if len(text) > _MAX_SCREEN_CHARS:
        return text[:_MAX_SCREEN_CHARS]
    return text


def _format_history(history: list[Turn]) -> str:
    """Render recent turns as a compact transcript for prompts."""

    recent = history[-_MAX_HISTORY_TURNS:]
    if not recent:
        return "(no questions asked yet)"
    lines: list[str] = []
    for i, turn in enumerate(recent, start=1):
        lines.append(f"Q{i}: {turn.question}")
        lines.append(f"A{i}: {turn.answer}")
    return "\n".join(lines)


class InterviewEngine:
    """Adaptive, context-aware AI interviewer built on the shared LLM gateway."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm if llm is not None else get_llm(get_settings())

    def opening_question(self, project_summary: str) -> str:
        """Return a context-aware first question from what is on screen."""

        summary = (project_summary or "").strip() or "(no project summary provided)"
        messages = [
            {"role": "system", "content": _INTERVIEWER_SYSTEM},
            {"role": "user", "content": _OPENING_PROMPT.format(summary=summary)},
        ]
        response = self._llm.chat(messages)
        question = response.text.strip()
        if not question:
            return "Walk me through what this project does and how you built it."
        return question

    def next_question(
        self, history: list[Turn], screen_text: str, latest_answer: str
    ) -> str:
        """Return an adaptive follow-up grounded in the screen and last answer."""

        messages = [
            {"role": "system", "content": _INTERVIEWER_SYSTEM},
            {
                "role": "user",
                "content": _FOLLOWUP_PROMPT.format(
                    history=_format_history(history),
                    screen=_truncate_screen(screen_text),
                    answer=(latest_answer or "").strip() or "(no answer given)",
                ),
            },
        ]
        response = self._llm.chat(messages)
        question = response.text.strip()
        if not question:
            return "Can you go one level deeper on the part you just described?"
        return question

    def score(self, history: list[Turn], screen_text: str) -> InterviewReport:
        """Score the interview against the rubric using an LLM judge.

        The prompt carries the :data:`core.llm.JUDGE_MARKER` so a deterministic
        mock can recognise the evaluation request. Output is parsed robustly and
        every score is clamped into 0..10.
        """

        transcript = _format_history(history)
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {
                "role": "user",
                "content": _JUDGE_PROMPT.format(
                    screen=_truncate_screen(screen_text),
                    transcript=transcript,
                    marker=JUDGE_MARKER,
                ),
            },
        ]
        response = self._llm.chat(messages)
        data = extract_json(response.text, default={})
        if not isinstance(data, dict):
            data = {}

        technical_depth = _clamp_score(data.get("technical_depth"))
        clarity = _clamp_score(data.get("clarity"))
        originality = _clamp_score(data.get("originality"))
        implementation_understanding = _clamp_score(
            data.get("implementation_understanding")
        )

        subs = [
            technical_depth,
            clarity,
            originality,
            implementation_understanding,
        ]
        # Use the model's overall if it gave one, else average the sub-scores.
        if "overall" in data:
            overall = _clamp_score(data.get("overall"))
        else:
            overall = round(sum(subs) / len(subs), 2)

        feedback = str(data.get("feedback", "")).strip()
        if not feedback:
            feedback = "No structured feedback was produced for this session."

        return InterviewReport(
            technical_depth=technical_depth,
            clarity=clarity,
            originality=originality,
            implementation_understanding=implementation_understanding,
            overall=overall,
            feedback=feedback,
            strengths=_clamp_str_list(data.get("strengths")),
            gaps=_clamp_str_list(data.get("gaps")),
        )

    def run_scripted(
        self,
        project_summary: str,
        qa_pairs: list[tuple[str, str]],
        screen_text: str,
    ) -> InterviewReport:
        """Play a fixed set of answers through the engine, then score.

        Convenience for demos and tests: the engine still generates each question
        adaptively (opening, then follow-ups grounded in the running history and
        screen), but the answers come from ``qa_pairs``. Each pair is
        ``(hint, answer)``; the ``hint`` is unused for control flow and only kept
        so callers can label answers. The produced :class:`Turn` list pairs the
        engine's actual questions with the scripted answers, then is scored.
        """

        history: list[Turn] = []
        answers = [answer for _hint, answer in qa_pairs]
        for index, answer in enumerate(answers):
            if index == 0:
                question = self.opening_question(project_summary)
            else:
                question = self.next_question(
                    history, screen_text, latest_answer=answers[index - 1]
                )
            history.append(Turn(question=question, answer=answer))
        return self.score(history, screen_text)


def run_scripted(
    project_summary: str,
    qa_pairs: list[tuple[str, str]],
    screen_text: str,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
) -> InterviewReport:
    """Module-level convenience wrapper around :meth:`InterviewEngine.run_scripted`.

    Builds an engine (using ``llm`` if given, else the configured or provided
    settings) and plays ``qa_pairs`` through it, returning the scored report.
    """

    if llm is None and settings is not None:
        llm = get_llm(settings)
    engine = InterviewEngine(llm)
    return engine.run_scripted(project_summary, qa_pairs, screen_text)
