#!/usr/bin/env bash
# Launch the NaviLearn platform (Streamlit multipage app, entry = Home.py).
#
# ARROW_DEFAULT_MEMORY_POOL=system forces pyarrow onto the system allocator.
# This build's pyarrow can segfault when Apache Arrow serialization runs on a
# thread other than the one that first imported it (Streamlit spins a fresh
# thread per rerun). The UI already avoids Arrow-backed chart elements, and this
# env var is the belt-and-braces guard so a stray Arrow path never crashes.
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/streamlit"
if [ ! -x "$PY" ]; then
  PY="streamlit"
fi

export ARROW_DEFAULT_MEMORY_POOL=system
exec "$PY" run Home.py "$@"
