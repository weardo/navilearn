"""Offline on-device summarizer: a quantized T5-small served via onnxruntime.

This is the Challenge 2 ultra-light summarizer, ported into NaviLearn as a
fully offline, on-device summary capability. Inference runs entirely through
onnxruntime (via optimum's ``ORTModelForSeq2SeqLM``); there is no LLM call, no
Groq, no network and no rate limit. The same quantized graphs are what would
ship to a browser via ``onnxruntime-web`` (WASM) for a client-side deployment.

The public surface is deliberately small and thin:

* :class:`OnnxSummarizer` lazily loads the quantized model once (cached as a
  module singleton via :func:`get_summarizer`) and exposes
  :meth:`OnnxSummarizer.summarize` (returns the summary string) and
  :meth:`OnnxSummarizer.summarize_timed` (returns a :class:`SummaryResult`
  carrying the summary plus load/latency/size telemetry).
* :func:`summarize` is a module-level convenience over the singleton.

Graceful degradation is a hard requirement: if the model directory or the
onnxruntime/optimum stack is missing, loading and summarizing never raise. The
summary comes back empty ("") so the rest of the platform keeps running.
"""

from __future__ import annotations

import glob
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_LOG = logging.getLogger(__name__)

# core/summarizer.py -> repo root is parent.parent, model lives under models/.
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "t5-small-onnx-int8"


def model_size_mb(model_dir: Path = MODEL_DIR) -> float:
    """Return the summed on-disk size (MB) of the quantized ``.onnx`` graphs."""

    try:
        total_bytes = sum(
            os.path.getsize(p) for p in glob.glob(str(model_dir / "*.onnx"))
        )
    except OSError:
        return 0.0
    return round(total_bytes / (1024 * 1024), 1)


@dataclass
class SummaryResult:
    """Result of a single summarization request, with on-device telemetry.

    ``text`` is empty when the model or its runtime is unavailable, so callers
    can render the empty state without special-casing exceptions.
    """

    text: str
    load_time_ms: float
    latency_ms: float
    model_size_mb: float


class OnnxSummarizer:
    """Quantized ONNX seq2seq summarizer served through onnxruntime.

    The heavy model + tokenizer load lazily on first use and never raises: a
    missing model directory or a missing onnxruntime/optimum stack leaves the
    instance in a disabled state where :meth:`summarize` returns ``""``.
    """

    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        """Record the model directory. No I/O happens until first summarize."""

        self.model_dir = model_dir
        self.load_time_ms: float = 0.0
        self.model_size_mb: float = model_size_mb(model_dir)
        self._tokenizer = None
        self._model = None
        self._load_attempted = False
        self._available = False

    # ------------------------------------------------------------------ #
    # Lazy loading
    # ------------------------------------------------------------------ #
    def _ensure_loaded(self) -> bool:
        """Load the model + tokenizer once; return whether inference is usable.

        Any failure (missing directory, missing dependency, corrupt graph) is
        logged and swallowed so the platform runs without the offline model.
        """

        if self._load_attempted:
            return self._available

        self._load_attempted = True

        if not self.model_dir.exists():
            _LOG.warning(
                "Offline summarizer model not found at %s; summaries disabled.",
                self.model_dir,
            )
            return False

        try:
            from optimum.onnxruntime import ORTModelForSeq2SeqLM
            from transformers import AutoTokenizer
        except Exception as exc:  # noqa: BLE001 - optional dependency, degrade.
            _LOG.warning(
                "onnxruntime/optimum stack unavailable; summaries disabled: %s",
                exc,
            )
            return False

        start = time.perf_counter()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            # Auto-detect the merged decoder (decoder_model_merged.onnx). Do NOT
            # pass decoder_file_name: that forces the non-merged path which then
            # needs a separate decoder_with_past graph we intentionally dropped.
            self._model = ORTModelForSeq2SeqLM.from_pretrained(
                self.model_dir,
                use_cache=True,
                use_merged=True,
                use_io_binding=False,
            )
        except Exception as exc:  # noqa: BLE001 - never crash the platform.
            _LOG.warning("Offline summarizer failed to load: %s", exc)
            self._tokenizer = None
            self._model = None
            return False

        self.load_time_ms = round((time.perf_counter() - start) * 1000, 1)
        self._available = True
        return True

    @property
    def available(self) -> bool:
        """True once the model is loaded and ready for inference."""

        return self._ensure_loaded()

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def summarize_timed(
        self,
        text: str,
        *,
        max_new_tokens: int = 80,
        min_length: int = 10,
        num_beams: int = 1,
    ) -> SummaryResult:
        """Summarize ``text`` and return the summary plus on-device telemetry.

        T5 uses the ``summarize:`` task prefix. Greedy decoding (``num_beams=1``)
        keeps latency low, matching the lightweight/real-time target. When the
        model is unavailable, or ``text`` is empty, the returned
        :class:`SummaryResult` has an empty ``text`` and zero latency.
        """

        if not text or not text.strip():
            return SummaryResult(
                text="",
                load_time_ms=self.load_time_ms,
                latency_ms=0.0,
                model_size_mb=self.model_size_mb,
            )

        if not self._ensure_loaded():
            return SummaryResult(
                text="",
                load_time_ms=self.load_time_ms,
                latency_ms=0.0,
                model_size_mb=self.model_size_mb,
            )

        prompt = "summarize: " + text.strip()
        start = time.perf_counter()
        try:
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                num_beams=num_beams,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
            summary = self._tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            ).strip()
        except Exception as exc:  # noqa: BLE001 - degrade to empty, never crash.
            _LOG.warning("Offline summarization failed: %s", exc)
            return SummaryResult(
                text="",
                load_time_ms=self.load_time_ms,
                latency_ms=0.0,
                model_size_mb=self.model_size_mb,
            )

        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return SummaryResult(
            text=summary,
            load_time_ms=self.load_time_ms,
            latency_ms=latency_ms,
            model_size_mb=self.model_size_mb,
        )

    def summarize(self, text: str, max_new_tokens: int = 80) -> str:
        """Return only the summary string, or ``""`` when unavailable."""

        return self.summarize_timed(text, max_new_tokens=max_new_tokens).text


# --------------------------------------------------------------------------- #
# Module singleton
# --------------------------------------------------------------------------- #
_SINGLETON: Optional[OnnxSummarizer] = None


def get_summarizer() -> OnnxSummarizer:
    """Return the process-wide :class:`OnnxSummarizer` singleton (lazy load)."""

    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = OnnxSummarizer()
    return _SINGLETON


def summarize(text: str, max_new_tokens: int = 80) -> str:
    """Summarize ``text`` via the shared singleton; ``""`` when unavailable."""

    return get_summarizer().summarize(text, max_new_tokens=max_new_tokens)
