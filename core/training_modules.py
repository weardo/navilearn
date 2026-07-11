"""Turn recurring meeting notes into reusable, LMS-ready training modules.

Mentors and teachers accumulate meeting notes (standups, onboarding sessions,
office hours) that repeat the same themes over and over. This module reads a
batch of those notes and, using the shared LLM gateway (:mod:`core.llm`), does
two things:

1. :func:`extract_themes` detects the recurring themes across every note.
2. :func:`generate_modules` distills the notes into role-based training modules
   (tutorial / how-to / FAQ) with structured steps and short definitions.

The output is a list of :class:`TrainingModule` dataclasses that render to clean
Markdown (:func:`render_markdown` / :func:`render_all`) ready to paste into an
LMS. :func:`save_modules` persists a run to disk (JSON + Markdown) and appends an
ordinal version entry to ``versions.json`` so how the modules evolve over time is
tracked.

Every LLM call goes through the vendor-agnostic gateway and every model reply is
parsed defensively via :func:`core.jsonutil.extract_json`, so malformed output
degrades to a sensible fallback instead of crashing a mentor's workflow.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from core.jsonutil import extract_json
from core.llm import LLMClient

if TYPE_CHECKING:  # avoid a runtime import cycle; used for annotations only.
    from core.repo import Lesson, Repository

_LOG = logging.getLogger(__name__)

# Cap total note text sent to the model so a big batch stays within context and
# cost stays predictable. Notes beyond this are truncated, newest logic first.
_MAX_CHARS = 12000


@dataclass
class TrainingModule:
    """A single reusable training module distilled from meeting notes.

    Attributes:
        title: Short, action-oriented module title.
        overview: One or two sentences on what the learner will be able to do.
        steps: Ordered how-to / tutorial steps.
        faqs: List of ``{"q": ..., "a": ...}`` question/answer pairs.
        role: The audience this module serves (for example "new mentee",
            "onboarding engineer", "teacher").
        source_theme: The recurring theme this module was built from.
    """

    title: str
    overview: str
    steps: list[str] = field(default_factory=list)
    faqs: list[dict] = field(default_factory=list)
    role: str = "learner"
    source_theme: str = ""


# --------------------------------------------------------------------------- #
# Note preparation
# --------------------------------------------------------------------------- #
def _join_notes(notes_texts: list[str]) -> str:
    """Concatenate notes into one delimited block, truncated to the char cap.

    Each note is separated by a visible divider so the model can tell where one
    meeting ends and the next begins, which helps it spot themes that recur
    across separate meetings rather than within a single one.
    """

    cleaned = [str(note).strip() for note in notes_texts if str(note).strip()]
    blocks: list[str] = []
    for index, note in enumerate(cleaned, start=1):
        blocks.append(f"--- MEETING NOTE {index} ---\n{note}")
    joined = "\n\n".join(blocks)
    if len(joined) > _MAX_CHARS:
        joined = joined[:_MAX_CHARS]
    return joined


# --------------------------------------------------------------------------- #
# Theme extraction
# --------------------------------------------------------------------------- #
_THEME_SYSTEM = (
    "You analyze recurring meeting notes for a learning organization. You "
    "identify themes that repeat across multiple separate meetings, not "
    "one-off items. You reply with JSON only, no prose."
)

_THEME_PROMPT = """Read the meeting notes below (each note is a separate meeting).
Identify the themes that RECUR across two or more meetings: shared topics,
repeated questions, common blockers, or onboarding steps that keep coming up.

Return ONLY a JSON array of short theme strings, most recurring first, for
example: ["Git workflow", "Standup etiquette", "Deploying to staging"].
Return at most 8 themes.

Meeting notes:
\"\"\"
{notes}
\"\"\"
"""


def extract_themes(llm: LLMClient, notes_texts: list[str]) -> list[str]:
    """Detect recurring themes across a batch of meeting notes.

    Args:
        llm: Any client satisfying :class:`core.llm.LLMClient`.
        notes_texts: One string per meeting note.

    Returns:
        A de-duplicated list of short theme strings (most recurring first),
        possibly empty if the notes are empty or the model returns nothing
        usable.
    """

    notes = _join_notes(notes_texts)
    if not notes:
        return []

    messages = [
        {"role": "system", "content": _THEME_SYSTEM},
        {"role": "user", "content": _THEME_PROMPT.format(notes=notes)},
    ]
    response = llm.chat(messages)
    data = extract_json(response.text, default=[])

    # Accept a bare array or an object wrapping one.
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                data = value
                break

    themes: list[str] = []
    seen: set[str] = set()
    if isinstance(data, list):
        for item in data:
            theme = str(item).strip() if not isinstance(item, dict) else str(
                item.get("theme") or item.get("name") or ""
            ).strip()
            key = theme.lower()
            if theme and key not in seen:
                themes.append(theme)
                seen.add(key)
    return themes


# --------------------------------------------------------------------------- #
# Module generation
# --------------------------------------------------------------------------- #
_MODULE_SYSTEM = (
    "You are an instructional designer. You turn recurring meeting notes into "
    "reusable, self-contained training modules for a learning management "
    "system. Each module is practical, role-targeted, and combines a short "
    "tutorial overview, concrete how-to steps, and a small FAQ. You reply with "
    "JSON only, no prose."
)

_MODULE_PROMPT = """From the meeting notes below, produce up to {max_modules}
reusable training modules covering the themes that recur across meetings.

Return ONLY a JSON array. Each element is an object with this exact shape:
{{
  "title": "short action-oriented title",
  "overview": "1-2 sentences on what the learner will be able to do",
  "steps": ["ordered how-to step", "next step", "..."],
  "faqs": [{{"q": "a common question from the notes", "a": "a concise answer"}}],
  "role": "who this is for, e.g. new mentee or onboarding engineer",
  "source_theme": "the recurring theme this module is built from"
}}

Make each module self-contained: someone who missed the meetings could learn
from it. Prefer 3-6 concrete steps and 2-4 FAQs per module. Do not invent facts
that are not supported by the notes.

Meeting notes:
\"\"\"
{notes}
\"\"\"
"""


def _coerce_str_list(value: object) -> list[str]:
    """Normalize an arbitrary JSON value into a clean list of strings."""

    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip() if value is not None else ""
    return [text] if text else []


def _coerce_faqs(value: object) -> list[dict]:
    """Normalize the FAQ field into a list of ``{"q": ..., "a": ...}`` dicts."""

    faqs: list[dict] = []
    if not isinstance(value, list):
        return faqs
    for item in value:
        if not isinstance(item, dict):
            continue
        question = str(item.get("q") or item.get("question") or "").strip()
        answer = str(item.get("a") or item.get("answer") or "").strip()
        if question and answer:
            faqs.append({"q": question, "a": answer})
    return faqs


def _module_from_dict(item: dict) -> TrainingModule | None:
    """Build a :class:`TrainingModule` from one parsed JSON object, or None."""

    title = str(item.get("title", "")).strip()
    overview = str(item.get("overview", "")).strip()
    if not title:
        return None
    return TrainingModule(
        title=title,
        overview=overview,
        steps=_coerce_str_list(item.get("steps")),
        faqs=_coerce_faqs(item.get("faqs")),
        role=str(item.get("role", "")).strip() or "learner",
        source_theme=str(item.get("source_theme", "")).strip(),
    )


def _fallback_module(notes_texts: list[str], themes: list[str]) -> TrainingModule:
    """Return a minimal module when the model yields no usable structure.

    This keeps :func:`generate_modules` non-empty (and the mock/offline path
    working) so the UI always has something to render and download.
    """

    theme = themes[0] if themes else "Team onboarding"
    return TrainingModule(
        title=f"Training module: {theme}",
        overview=(
            "A starter module distilled from recurring meeting notes. Review the "
            "steps and FAQs, then refine them for your learners."
        ),
        steps=[
            "Read the source meeting notes for context.",
            "Confirm the recurring theme this module should cover.",
            "Add the concrete actions your learners must take.",
        ],
        faqs=[
            {
                "q": "What is this module based on?",
                "a": "Recurring themes detected across your meeting notes.",
            }
        ],
        role="learner",
        source_theme=theme,
    )


def generate_modules(
    llm: LLMClient, notes_texts: list[str], max_modules: int = 5
) -> list[TrainingModule]:
    """Generate role-based training modules from a batch of meeting notes.

    Args:
        llm: Any client satisfying :class:`core.llm.LLMClient`.
        notes_texts: One string per meeting note.
        max_modules: Upper bound on the number of modules returned.

    Returns:
        A list of :class:`TrainingModule`. Always at least one item when there
        is any note text: if the model returns nothing structured (for example
        under the deterministic mock client), a single fallback module is
        returned so callers never get an empty result from real input.
    """

    notes = _join_notes(notes_texts)
    if not notes:
        return []

    messages = [
        {"role": "system", "content": _MODULE_SYSTEM},
        {
            "role": "user",
            "content": _MODULE_PROMPT.format(max_modules=max_modules, notes=notes),
        },
    ]
    response = llm.chat(messages)
    data = extract_json(response.text, default=[])

    # Accept a bare array or an object wrapping one.
    if isinstance(data, dict):
        wrapped = None
        for value in data.values():
            if isinstance(value, list):
                wrapped = value
                break
        # A single module object (not wrapped in a list) is also valid.
        data = wrapped if wrapped is not None else [data]

    modules: list[TrainingModule] = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            module = _module_from_dict(item)
            if module is not None:
                modules.append(module)
            if len(modules) >= max_modules:
                break

    if not modules:
        themes = extract_themes(llm, notes_texts)
        modules = [_fallback_module(notes_texts, themes)]
    return modules[:max_modules]


# --------------------------------------------------------------------------- #
# Markdown rendering (LMS-ready)
# --------------------------------------------------------------------------- #
def render_markdown(module: TrainingModule) -> str:
    """Render one :class:`TrainingModule` as a self-contained Markdown section."""

    lines: list[str] = [f"# {module.title}"]

    meta: list[str] = []
    if module.role:
        meta.append(f"**Role:** {module.role}")
    if module.source_theme:
        meta.append(f"**Theme:** {module.source_theme}")
    if meta:
        lines.append("")
        lines.append("  ".join(meta))

    if module.overview:
        lines.append("")
        lines.append("## Overview")
        lines.append(module.overview)

    if module.steps:
        lines.append("")
        lines.append("## Steps")
        for index, step in enumerate(module.steps, start=1):
            lines.append(f"{index}. {step}")

    if module.faqs:
        lines.append("")
        lines.append("## FAQ")
        for faq in module.faqs:
            question = str(faq.get("q", "")).strip()
            answer = str(faq.get("a", "")).strip()
            if not question:
                continue
            lines.append(f"**Q: {question}**")
            if answer:
                lines.append(f"A: {answer}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_all(modules: list[TrainingModule]) -> str:
    """Render a list of modules as one LMS-ready Markdown document."""

    if not modules:
        return "# Training Modules\n\nNo modules were generated.\n"

    header = ["# Training Modules", "", f"_{len(modules)} module(s) generated._", ""]
    sections = [render_markdown(module).rstrip() for module in modules]
    return "\n".join(header) + "\n" + "\n\n---\n\n".join(sections) + "\n"


# --------------------------------------------------------------------------- #
# Publishing to a course (real lessons students can take)
# --------------------------------------------------------------------------- #
# The section name every published training lesson is grouped under in Learn.
PUBLISH_MODULE_LABEL = "Training modules"


def publish_modules_as_lessons(
    repo: "Repository",
    course_id: str,
    modules: list[TrainingModule],
    module_label: str = PUBLISH_MODULE_LABEL,
) -> list["Lesson"]:
    """Create one real lesson per generated module in the given course.

    Each module becomes a :class:`core.repo.Lesson` whose ``content`` is the
    module rendered as self-contained Markdown (:func:`render_markdown`) and
    whose ``module`` groups it under ``module_label`` in the Learn surface. A
    blank ``order_index`` lets the repository append the lesson after any
    existing ones. Publishing is the primary action here, so a per-lesson
    failure is logged and skipped rather than aborting the whole batch.

    Args:
        repo: Any :class:`core.repo.Repository` backend.
        course_id: The id of the course to publish into.
        modules: The modules produced by :func:`generate_modules`.
        module_label: The section name lessons are grouped under.

    Returns:
        The list of :class:`core.repo.Lesson` rows that were created.
    """

    from core.repo import Lesson  # local import: avoids a module import cycle.

    created: list[Lesson] = []
    for module in modules:
        try:
            lesson = repo.create_lesson(
                Lesson(
                    id="",
                    course_id=course_id,
                    title=module.title,
                    content=render_markdown(module),
                    module=module_label,
                )
            )
            created.append(lesson)
        except Exception as exc:  # noqa: BLE001 - one bad lesson must not stop the rest.
            _LOG.warning("Could not publish module %r: %s", module.title, exc)
    return created


# --------------------------------------------------------------------------- #
# Light version tracking
# --------------------------------------------------------------------------- #
def _next_ordinal(versions_path: str) -> int:
    """Return the next 1-based ordinal given an existing ``versions.json``."""

    if not os.path.exists(versions_path):
        return 1
    try:
        with open(versions_path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if isinstance(existing, list) and existing:
            last = existing[-1]
            if isinstance(last, dict):
                return int(last.get("version", len(existing))) + 1
            return len(existing) + 1
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return 1


def save_modules(
    modules: list[TrainingModule], out_dir: str, version: int | None = None
) -> dict:
    """Persist a run of modules and append an ordinal version entry.

    Writes three things into ``out_dir`` (created if missing):

    * ``modules_v<N>.json`` - the structured modules for this run.
    * ``modules_v<N>.md`` - the LMS-ready Markdown for this run.
    * ``versions.json`` - a growing list of run entries so changes over time are
      tracked. Each entry records its ordinal ``version``, an ISO ``created_at``
      timestamp, the module count, and the two file names.

    Args:
        modules: The modules produced by :func:`generate_modules`.
        out_dir: Directory to write into.
        version: Optional explicit ordinal; when omitted the next ordinal is
            derived from any existing ``versions.json``.

    Returns:
        The version entry dict that was appended to ``versions.json``.
    """

    os.makedirs(out_dir, exist_ok=True)
    versions_path = os.path.join(out_dir, "versions.json")
    ordinal = int(version) if version is not None else _next_ordinal(versions_path)

    json_name = f"modules_v{ordinal}.json"
    md_name = f"modules_v{ordinal}.md"

    payload = [asdict(module) for module in modules]
    with open(os.path.join(out_dir, json_name), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, md_name), "w", encoding="utf-8") as handle:
        handle.write(render_all(modules))

    entry = {
        "version": ordinal,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_count": len(modules),
        "titles": [module.title for module in modules],
        "json_file": json_name,
        "markdown_file": md_name,
    }

    versions: list = []
    if os.path.exists(versions_path):
        try:
            with open(versions_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, list):
                versions = loaded
        except (OSError, ValueError, json.JSONDecodeError):
            versions = []
    versions.append(entry)

    with open(versions_path, "w", encoding="utf-8") as handle:
        json.dump(versions, handle, indent=2, ensure_ascii=False)

    return entry
