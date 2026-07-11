"""Settings for NaviLearn. Reads .env (Groq key reused from the kit)."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
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

    # Swappable vector store. "chroma" (default, local, offline) or "supabase"
    # (pgvector via the namespaced navilearn_chunks table + navilearn_match_chunks).
    vector_backend: str = "chroma"

    # Swappable data layer. "sqlite" (default, stdlib) or "supabase" (Postgres).
    db_backend: str = "sqlite"
    sqlite_path: str = "navilearn.db"
    supabase_url: str = ""
    supabase_service_role_key: str = ""
    supabase_anon_key: str = ""

    # Secret for signing the persistent-login cookie (HMAC). Empty means the app
    # generates and persists one in .session_secret so cookies survive restarts.
    session_secret: str = ""

    # Co-solve "Run (Python)" executes code on the host, which is a remote-code-
    # execution risk on a public deploy. Off by default so the hosted demo is
    # safe; set env NAVI_ENABLE_CODE_RUN=true only in a trusted/local environment.
    enable_code_run: bool = Field(
        default=False, validation_alias="NAVI_ENABLE_CODE_RUN"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
