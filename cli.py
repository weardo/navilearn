"""NaviLearn command-line interface.

Ingest a single learning source (document, transcript file or YouTube URL),
run the full NaviLearn pipeline and write a folder of structured study
artifacts: a concept map, flashcards (JSON and CSV), a summary, a concept
graph (DOT and JSON) and a prerequisite-ordered learning path.

Usage:
    python cli.py <source> [--out OUTDIR] [--n-flashcards N]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

from core.artifacts import (
    Summary,
    flashcards_to_csv,
    flashcards_to_json,
    learning_path,
)
from core.pipeline import ProcessResult, get_store, process


def _slugify(title: str) -> str:
    """Turn a title into a filesystem-friendly folder name.

    Lowercases, replaces runs of non-alphanumeric characters with a single
    hyphen and trims stray hyphens. Falls back to "source" when empty.
    """

    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "source"


def _summary_markdown(title: str, summary: Summary) -> str:
    """Render a Summary as a Markdown document with per-topic sections."""

    lines: list[str] = [f"# {title}", "", "## Overview", "", summary.overall, ""]
    if summary.per_topic:
        lines.append("## Topics")
        lines.append("")
        for topic, text in summary.per_topic.items():
            lines.append(f"### {topic}")
            lines.append("")
            lines.append(text)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_text(path: Path, text: str) -> int:
    """Write text to ``path`` and return the number of bytes written."""

    path.write_text(text, encoding="utf-8")
    return path.stat().st_size


def _run(source: str, out: str | None, n_flashcards: int) -> Path:
    """Process ``source`` and write all artifacts, returning the output dir."""

    # A shared store lets topic-tagged chunks accumulate for later retrieval.
    store = get_store()
    result: ProcessResult = process(source, store=store, n_flashcards=n_flashcards)

    title = result.title or _slugify(source)
    out_dir = Path(out) if out else Path("outputs") / _slugify(title)
    out_dir.mkdir(parents=True, exist_ok=True)

    concepts_payload = dataclasses.asdict(result.concept_map)
    path_payload = learning_path(result.concept_map)

    written: list[tuple[str, int]] = []
    written.append(
        (
            "concepts.json",
            _write_text(
                out_dir / "concepts.json",
                json.dumps(concepts_payload, indent=2, ensure_ascii=False),
            ),
        )
    )
    written.append(
        (
            "flashcards.json",
            _write_text(
                out_dir / "flashcards.json", flashcards_to_json(result.flashcards)
            ),
        )
    )
    written.append(
        (
            "flashcards.csv",
            _write_text(
                out_dir / "flashcards.csv", flashcards_to_csv(result.flashcards)
            ),
        )
    )
    written.append(
        (
            "summary.md",
            _write_text(
                out_dir / "summary.md", _summary_markdown(title, result.summary)
            ),
        )
    )
    written.append(("graph.dot", _write_text(out_dir / "graph.dot", result.graph_dot)))
    written.append(
        (
            "graph.json",
            _write_text(
                out_dir / "graph.json",
                json.dumps(result.graph_json, indent=2, ensure_ascii=False),
            ),
        )
    )
    written.append(
        (
            "learning_path.json",
            _write_text(
                out_dir / "learning_path.json",
                json.dumps(path_payload, indent=2, ensure_ascii=False),
            ),
        )
    )

    # Concise report of what the pipeline produced.
    print(f"NaviLearn processed: {source}")
    print(f"  title:       {title}")
    print(f"  chunks:      {result.n_chunks}")
    print(f"  concepts:    {len(result.concept_map.concepts)}")
    print(f"  topics:      {len(result.concept_map.topics)}")
    print(f"  flashcards:  {len(result.flashcards)}")
    print(f"  output dir:  {out_dir}")
    print("  files:")
    for name, size in written:
        print(f"    - {name} ({size} bytes)")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for the NaviLearn CLI."""

    parser = argparse.ArgumentParser(
        prog="navilearn",
        description=(
            "Ingest a learning source and generate structured study artifacts: "
            "concept map, flashcards, summary, concept graph and learning path."
        ),
    )
    parser.add_argument(
        "source",
        help="Path to a document/transcript (pdf, docx, txt, md, srt, vtt) or a YouTube URL.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: ./outputs/<title>/).",
    )
    parser.add_argument(
        "--n-flashcards",
        type=int,
        default=12,
        help="Number of flashcards to generate (default: 12).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse arguments, run the pipeline, report the result."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _run(args.source, args.out, args.n_flashcards)
    except FileNotFoundError as exc:
        print(f"error: source not found: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - surface any failure cleanly to the CLI user
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
