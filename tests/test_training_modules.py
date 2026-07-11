"""Offline tests for the training-module generator (core.training_modules).

These use the deterministic MockClient so they never touch the network. They
verify Markdown rendering of a hand-built module, that generation always yields
at least a fallback module under the mock, theme extraction shape, and that the
version tracker writes JSON + Markdown and appends ordinal version entries.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Settings
from core.llm import MockClient, get_llm
from core.training_modules import (
    TrainingModule,
    extract_themes,
    generate_modules,
    render_all,
    render_markdown,
    save_modules,
)

_SAMPLE_NOTES = [
    "Standup: mentees cannot push to Git because SSH keys are not set up. "
    "Standup format is yesterday / today / blockers.",
    "Onboarding: walked the batch through SSH key setup again. Deploy to "
    "staging by building first, then deploying from main.",
]


def _hand_built_module() -> TrainingModule:
    return TrainingModule(
        title="Set up SSH keys for Git",
        overview="Learn to authenticate to the class repo with an SSH key.",
        steps=["Generate a key with ssh-keygen", "Add the public key", "Test the connection"],
        faqs=[{"q": "Why does push ask for a password?", "a": "You are on HTTPS, not SSH."}],
        role="new mentee",
        source_theme="Git SSH setup",
    )


def test_render_markdown_contains_title_and_steps():
    module = _hand_built_module()
    markdown = render_markdown(module)
    assert module.title in markdown
    for step in module.steps:
        assert step in markdown
    # The FAQ question and role surface too.
    assert "Why does push ask for a password?" in markdown
    assert "new mentee" in markdown


def test_render_all_wraps_multiple_modules():
    modules = [_hand_built_module(), _hand_built_module()]
    doc = render_all(modules)
    assert "# Training Modules" in doc
    assert doc.count(_hand_built_module().title) >= 2


def test_generate_modules_returns_list_under_mock():
    modules = generate_modules(MockClient(), _SAMPLE_NOTES, max_modules=3)
    assert isinstance(modules, list)
    assert len(modules) >= 1
    for module in modules:
        assert isinstance(module, TrainingModule)
        assert module.title.strip()


def test_generate_modules_empty_notes_returns_empty():
    assert generate_modules(MockClient(), []) == []
    assert generate_modules(MockClient(), ["   ", ""]) == []


def test_extract_themes_returns_list():
    themes = extract_themes(MockClient(), _SAMPLE_NOTES)
    assert isinstance(themes, list)


def test_get_llm_mock_provider_generates():
    llm = get_llm(Settings(llm_provider="mock"))
    modules = generate_modules(llm, _SAMPLE_NOTES)
    assert isinstance(modules, list) and modules


def test_save_modules_writes_files_and_tracks_versions(tmp_path):
    out_dir = str(tmp_path / "modules")
    modules = [_hand_built_module()]

    entry1 = save_modules(modules, out_dir)
    assert entry1["version"] == 1
    assert os.path.exists(os.path.join(out_dir, "modules_v1.json"))
    assert os.path.exists(os.path.join(out_dir, "modules_v1.md"))
    assert os.path.exists(os.path.join(out_dir, "versions.json"))

    # A second run appends an incremented ordinal.
    entry2 = save_modules(modules, out_dir)
    assert entry2["version"] == 2
    assert os.path.exists(os.path.join(out_dir, "modules_v2.md"))

    with open(os.path.join(out_dir, "versions.json"), "r", encoding="utf-8") as handle:
        versions = json.load(handle)
    assert isinstance(versions, list) and len(versions) == 2
    assert [v["version"] for v in versions] == [1, 2]

    # The saved JSON round-trips the module structure.
    with open(os.path.join(out_dir, "modules_v1.json"), "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload[0]["title"] == _hand_built_module().title
