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
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chroma_dir: str = ".chroma"


@lru_cache
def get_settings() -> Settings:
    return Settings()
