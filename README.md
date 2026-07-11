# NaviLearn

**Challenge 6: Multi-Source Learning Content Ingestion and Structured Output Generation.**

NaviLearn turns raw learning material (documents, transcripts, video captions)
into structured, ready-to-study outputs: a concept map, a browsable concept
graph, flashcards you can import anywhere (JSON and CSV), topic-scoped
summaries, a prerequisite-ordered learning path and a searchable topic store.
One engine, one call, many artifacts.

## What it is

Feed NaviLearn a single source and it runs a full ingestion pipeline that:

1. Parses the source into clean text (PDF, DOCX, TXT, Markdown, SRT/VTT captions, or a YouTube transcript).
2. Extracts a structured concept map (concepts, topics, a topic hierarchy and typed edges) with an LLM.
3. Generates study artifacts from that map and text: flashcards, summaries, a concept graph and a learning path.
4. Embeds and stores every chunk in a persistent vector store, tagged by topic, so content is retrievable later.

It ships as both a Streamlit web tool and a command-line interface, backed by a
reusable `core/` engine.

## Architecture

```
                 +----------------------------------------------------------+
                 |                      NaviLearn engine                    |
                 |                                                           |
  source  --->   |  parse  --->  concepts  --->  artifacts + graph          |
 (pdf/docx/      |  (extract_text,   (extract_        (flashcards,           |
  txt/md/srt/    |   chunk_text)      concept_map)     summary, graph,       |
  vtt/YouTube)   |                                     learning_path)        |
                 |       |                                                   |
                 |       +------>  topic store (embed + chroma, per-topic)   |
                 +----------------------------------------------------------+
                                            |
                                            v
                     concepts.json  flashcards.json/.csv  summary.md
                     graph.dot  graph.json  learning_path.json
```

- **parse** (`core/parsers.py`): detects the input type, extracts text and splits it into chunks.
- **concepts** (`core/concepts.py`): asks the LLM for a strict-JSON concept map and coerces it into typed dataclasses.
- **artifacts** (`core/artifacts.py`): flashcards, summaries and the prerequisite-ordered learning path.
- **graph** (`core/graph.py`): Graphviz DOT and plain node/edge JSON renderings of the concept map.
- **topic store** (`core/store.py`): embeds chunks with FastEmbed and persists them to Chroma, each tagged with its best-matching topic for later retrieval.
- **pipeline** (`core/pipeline.py`): ties it all together behind a single `process(source)` call.

## Supported inputs

| Input | How it is read |
| --- | --- |
| PDF (`.pdf`) | `pypdf` text extraction |
| Word (`.docx`) | `python-docx` |
| Plain text (`.txt`) | read directly |
| Markdown (`.md`) | read directly |
| Subtitles (`.srt`, `.vtt`) | caption cues parsed into plain transcript text |
| YouTube URL | transcript fetched via `youtube-transcript-api` |

**Video support is via transcripts.** NaviLearn treats video as its caption or
transcript track: paste a YouTube URL (captions are fetched automatically) or
drop in an exported `.srt` / `.vtt` file. Audio-only or caption-less video is a
documented extension point: wiring a Whisper speech-to-text step
(`core/stt.py`) to emit a transcript would let any media file flow through the
exact same pipeline unchanged.

## Quickstart

### 1. Environment

Reuse the provided virtual environment (nothing to install), or create your own
and `pip install -e ".[dev]"`.

```bash
# Reuse the kit venv
PYTHON=/mnt/data/astra/projects/jobprep/kit/.venv/bin/python

# Configure secrets (Groq LLM key, embedding model, chroma dir)
cp .env.example .env   # then edit, if an example is present; a working .env may already exist
```

### 2. Web tool

```bash
$PYTHON -m streamlit run app.py
```

Upload a file or paste a YouTube URL, then browse the concept graph, flashcards,
summaries, learning path and topic search in the browser.

### 3. Command-line interface

```bash
$PYTHON cli.py <source> [--out OUTDIR] [--n-flashcards N]

# Example
$PYTHON cli.py data/samples/lesson_photosynthesis.md --out ./outputs/photosynthesis --n-flashcards 6
```

This writes, into `OUTDIR` (default `./outputs/<title>/`):

- `concepts.json` — the full concept map (concepts, topics, hierarchy, edges).
- `flashcards.json` / `flashcards.csv` — study cards, ready to import (for example into Anki).
- `summary.md` — an overall summary plus one section per topic.
- `graph.dot` / `graph.json` — the concept graph as Graphviz DOT and as node/edge JSON.
- `learning_path.json` — concepts ordered to respect prerequisite edges.

Run the test suite with `$PYTHON -m pytest`.

## Deliverables mapping

| Challenge 6 deliverable | Where it lives |
| --- | --- |
| Web tool for multi-source ingestion | `app.py` (Streamlit), engine in `core/` |
| Structured flashcards (JSON + CSV) | `flashcards.json`, `flashcards.csv` via `core/artifacts.py` |
| Concept / knowledge graph | `graph.dot`, `graph.json` via `core/graph.py` |
| Topic-based retrieval | `core/store.py` (FastEmbed embeddings + Chroma, per-topic tags) |
| Summaries and learning path | `summary.md`, `learning_path.json` via `core/artifacts.py` |

## How it maps to Challenge 7

Challenge 7 (turning meeting notes into training modules) is the same engine
pointed at a different source. Meeting notes or a call transcript flow through
the identical `parse -> concepts -> artifacts + graph -> topic store` pipeline:
the concept map becomes the module outline, the summaries become module
overviews, the flashcards become knowledge checks and the learning path becomes
the module sequence. Only the input adapter and output framing change; the
structured-extraction core is shared.
