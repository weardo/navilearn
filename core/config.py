"""Settings for NaviLearn. Reads .env (Groq key reused from the kit)."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "groq"
    llm_model: str = "llama-3.1-8b-instant"
    ollama_base_url: str = "http://localhost:11434"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    sarvam_api_key: str = ""
    # Comma-separated LiteLLM model strings used as offline/backup fallbacks,
    # e.g. "ollama/llama3.2:3b". Empty means no fallback (retries only).
    llm_fallback_models: str = ""
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chroma_dir: str = ".chroma"


@lru_cache
def get_settings() -> Settings:
    return Settings()
