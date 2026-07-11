"""LLM client abstraction with interchangeable providers.

Providers implement a single ``chat`` method returning a normalized
``LLMResponse``. The default is local Ollama; Groq and OpenAI use the same
OpenAI-compatible chat-completions shape; ``MockClient`` is deterministic and
network-free for tests and evaluation harness wiring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

from core.config import Settings

# Marker embedded in judge prompts so the deterministic MockClient can recognise
# an evaluation request and return a valid JSON verdict without any network.
JUDGE_MARKER = "JUDGE_JSON"

_HTTP_TIMEOUT = 120.0


@dataclass
class LLMResponse:
    """Normalized result of a single chat completion."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


@runtime_checkable
class LLMClient(Protocol):
    """Protocol every provider client satisfies."""

    def chat(self, messages: list[dict], **opts: Any) -> LLMResponse:
        """Return a completion for the given chat messages."""
        ...


def _word_count(text: str) -> int:
    return len(text.split())


def _messages_word_count(messages: list[dict]) -> int:
    return sum(_word_count(str(m.get("content", ""))) for m in messages)


class OllamaClient:
    """Local Ollama provider (POST /api/chat, non-streaming)."""

    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.llm_model

    def chat(self, messages: list[dict], **opts: Any) -> LLMResponse:
        start = time.perf_counter()
        payload = {"model": self._model, "messages": messages, "stream": False}
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            resp = client.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        latency_ms = (time.perf_counter() - start) * 1000.0
        text = data.get("message", {}).get("content", "")
        prompt_tokens = int(
            data.get("prompt_eval_count", _messages_word_count(messages))
        )
        completion_tokens = int(data.get("eval_count", _word_count(text)))
        return LLMResponse(text, prompt_tokens, completion_tokens, latency_ms)


class _OpenAICompatClient:
    """Shared implementation for OpenAI-compatible chat-completions APIs."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model

    def chat(self, messages: list[dict], **opts: Any) -> LLMResponse:
        start = time.perf_counter()
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload: dict[str, Any] = {"model": self._model, "messages": messages}
        payload.update(opts)
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            resp = client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        latency_ms = (time.perf_counter() - start) * 1000.0
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", _messages_word_count(messages)))
        completion_tokens = int(usage.get("completion_tokens", _word_count(text)))
        return LLMResponse(text, prompt_tokens, completion_tokens, latency_ms)


class GroqClient(_OpenAICompatClient):
    """Groq provider (OpenAI-compatible, hosted, free tier)."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.groq_api_key,
            model=settings.groq_model,
        )


class OpenAIClient(_OpenAICompatClient):
    """OpenAI provider (chat completions)."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(
            base_url="https://api.openai.com/v1",
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )


class MockClient:
    """Deterministic, network-free client for tests and eval wiring.

    If any message content contains the JUDGE marker it returns a perfect-score
    JSON verdict. Otherwise it echoes the last user question in a short canned
    answer. Token counts are word counts; latency is a small constant.
    """

    _LATENCY_MS = 1.0

    def chat(self, messages: list[dict], **opts: Any) -> LLMResponse:
        joined = "\n".join(str(m.get("content", "")) for m in messages)
        if JUDGE_MARKER in joined:
            text = '{"correctness": 1.0, "groundedness": 1.0, "relevance": 1.0}'
            return LLMResponse(
                text,
                _messages_word_count(messages),
                _word_count(text),
                self._LATENCY_MS,
            )
        last_user = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                last_user = str(message.get("content", ""))
                break
        text = f"Based on the provided context, here is a simple answer to: {last_user}"
        return LLMResponse(
            text,
            _messages_word_count(messages),
            _word_count(text),
            self._LATENCY_MS,
        )


def get_llm(settings: Settings) -> LLMClient:
    """Select and construct the LLM client for the configured provider."""

    provider = settings.llm_provider.lower()
    if provider == "ollama":
        return OllamaClient(settings)
    if provider == "groq":
        return GroqClient(settings)
    if provider == "openai":
        return OpenAIClient(settings)
    if provider == "mock":
        return MockClient()
    raise ValueError(f"Unknown llm_provider: {settings.llm_provider!r}")
