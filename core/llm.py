"""Unified LLM gateway: one interface, any vendor or offline model.

Built on LiteLLM so a feature never binds to a vendor. The provider and model come
from Settings, so the SAME code runs against Groq, OpenAI, Sarvam (Indic), or a
fully offline Ollama model by changing one env var. Adds automatic retries and
optional cross-provider fallbacks (e.g. hosted primary, offline backup) for
production resilience. MockClient stays network-free and deterministic for tests.

Every feature imports get_llm(settings) and calls .chat(messages) -> LLMResponse.
The gateway is the single AI entry point (the moat), so swapping vendors or going
offline is a config change, not a code change.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import litellm
from litellm import completion

from core.config import Settings

# Keep LiteLLM quiet and portable: drop params a given provider does not support
# instead of erroring, and do not phone home.
litellm.drop_params = True
litellm.telemetry = False
litellm.suppress_debug_info = True

# Marker embedded in judge prompts so the deterministic MockClient can recognise
# an evaluation request and return a valid JSON verdict without any network.
JUDGE_MARKER = "JUDGE_JSON"

_TIMEOUT = 120.0
_RETRIES = 2


@dataclass
class LLMResponse:
    """Normalized result of a single chat completion, provider-independent."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str = ""


@runtime_checkable
class LLMClient(Protocol):
    """Protocol every client satisfies (gateway or mock)."""

    def chat(self, messages: list[dict], **opts: Any) -> LLMResponse:
        ...


def _word_count(text: str) -> int:
    return len(str(text).split())


def _messages_word_count(messages: list[dict]) -> int:
    return sum(_word_count(m.get("content", "")) for m in messages)


def _provider_kwargs(settings: Settings) -> tuple[str, dict[str, Any]]:
    """Map the configured provider to a LiteLLM model string + call kwargs.

    LiteLLM model strings are "<provider>/<model>". Any hosted vendor with an
    OpenAI-compatible endpoint (Sarvam included) is reached via api_base + api_key.
    """
    provider = settings.llm_provider.lower()
    if provider == "groq":
        return f"groq/{settings.groq_model}", {"api_key": settings.groq_api_key}
    if provider == "openai":
        return f"openai/{settings.openai_model}", {"api_key": settings.openai_api_key}
    if provider == "ollama":
        return f"ollama/{settings.llm_model}", {"api_base": settings.ollama_base_url}
    if provider == "sarvam":
        # Sarvam exposes an OpenAI-compatible chat endpoint; reach it as a custom
        # openai provider so the Indic model swaps in with no code change.
        return f"openai/{settings.llm_model}", {
            "api_key": settings.sarvam_api_key,
            "api_base": "https://api.sarvam.ai/v1",
        }
    raise ValueError(f"Unknown llm_provider: {settings.llm_provider!r}")


class LiteLLMGateway:
    """Vendor-agnostic chat client. Same call shape for every provider."""

    def __init__(self, settings: Settings) -> None:
        self._model, self._kwargs = _provider_kwargs(settings)
        # Optional cross-provider fallbacks, e.g. "ollama/llama3.2:3b" as an offline
        # backup when a hosted provider is unreachable. Comma-separated in config.
        self._fallbacks = [
            m.strip() for m in settings.llm_fallback_models.split(",") if m.strip()
        ]

    def chat(self, messages: list[dict], **opts: Any) -> LLMResponse:
        start = time.perf_counter()
        call: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "timeout": _TIMEOUT,
            "num_retries": _RETRIES,
            **self._kwargs,
            **opts,
        }
        if self._fallbacks:
            call["fallbacks"] = self._fallbacks
        resp = completion(**call)
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or _messages_word_count(messages))
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or _word_count(text))
        return LLMResponse(
            text=text.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            model=getattr(resp, "model", self._model),
        )


class MockClient:
    """Deterministic, network-free client for tests and eval wiring."""

    _LATENCY_MS = 1.0

    def chat(self, messages: list[dict], **opts: Any) -> LLMResponse:
        joined = "\n".join(str(m.get("content", "")) for m in messages)
        if JUDGE_MARKER in joined:
            text = '{"correctness": 1.0, "groundedness": 1.0, "relevance": 1.0}'
            return LLMResponse(text, _messages_word_count(messages), _word_count(text), self._LATENCY_MS, "mock")
        last_user = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                last_user = str(message.get("content", ""))
                break
        text = f"Based on the provided context, here is a simple answer to: {last_user}"
        return LLMResponse(text, _messages_word_count(messages), _word_count(text), self._LATENCY_MS, "mock")


def get_llm(settings: Settings) -> LLMClient:
    """Return the LLM client for the configured provider (mock stays offline)."""
    if settings.llm_provider.lower() == "mock":
        return MockClient()
    return LiteLLMGateway(settings)
