"""Offline tests for the AI interviewer engine (core.interview).

These use the deterministic MockClient so they never touch the network. The mock
recognises the JUDGE_JSON marker embedded in the scoring prompt and returns a
fixed verdict, letting us exercise parsing, clamping and report shape.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Settings
from core.interview import (
    InterviewEngine,
    InterviewReport,
    Turn,
    run_scripted,
)
from core.llm import MockClient, get_llm

_SUB_SCORE_FIELDS = (
    "technical_depth",
    "clarity",
    "originality",
    "implementation_understanding",
    "overall",
)


def _mock_engine() -> InterviewEngine:
    return InterviewEngine(MockClient())


def test_opening_question_non_empty():
    engine = _mock_engine()
    question = engine.opening_question("A CLI todo app in Python with a JSON store.")
    assert isinstance(question, str)
    assert question.strip()


def test_next_question_non_empty():
    engine = _mock_engine()
    history = [Turn(question="What does it do?", answer="It stores todos.")]
    question = engine.next_question(
        history, screen_text="def add(task): ...", latest_answer="It stores todos."
    )
    assert isinstance(question, str)
    assert question.strip()


def test_get_llm_mock_provider_builds_engine():
    settings = Settings(llm_provider="mock")
    engine = InterviewEngine(get_llm(settings))
    assert engine.opening_question("Tiny app").strip()


def test_score_returns_report_with_scores_in_range():
    engine = _mock_engine()
    history = [
        Turn(question="What does it do?", answer="It manages a todo list."),
        Turn(question="How is data stored?", answer="In a JSON file."),
    ]
    report = engine.score(history, screen_text="todo.json + cli.py")
    assert isinstance(report, InterviewReport)
    for field_name in _SUB_SCORE_FIELDS:
        value = getattr(report, field_name)
        assert 0.0 <= value <= 10.0, f"{field_name}={value} out of range"
    assert isinstance(report.feedback, str) and report.feedback.strip()
    assert isinstance(report.strengths, list)
    assert isinstance(report.gaps, list)


def test_run_scripted_produces_valid_report():
    report = run_scripted(
        project_summary="A todo app.",
        qa_pairs=[("intro", "It tracks tasks."), ("storage", "Saved as JSON.")],
        screen_text="cli.py, store.json",
        llm=MockClient(),
    )
    assert isinstance(report, InterviewReport)
    for field_name in _SUB_SCORE_FIELDS:
        assert 0.0 <= getattr(report, field_name) <= 10.0
