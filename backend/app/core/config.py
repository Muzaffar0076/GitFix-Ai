"""
app/core/config.py
──────────────────
Central configuration for GitFix AI.

HOW IT WORKS:
  - `BaseSettings` (from pydantic-settings) automatically reads your .env file.
  - Each variable defined here must exist in .env, or have a default value.
  - `@lru_cache` ensures the .env file is only read ONCE at startup, not on
    every import. This is a standard Python performance pattern.

USAGE (in any other file):
  from app.core.config import get_settings
  settings = get_settings()
  print(settings.GROQ_API_KEY)
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All environment variables for the application.
    pydantic-settings reads these automatically from the .env file.
    If a required variable is missing, the app will CRASH at startup
    with a clear error message — which is exactly what you want.
    """

    # ── Groq (Free LLM) ──────────────────────────────────────────────────────
    GROQ_API_KEY: str      # Required — get free key at console.groq.com

    # ── GitHub ────────────────────────────────────────────────────────────────
    GITHUB_PAT: str  # Required — Personal Access Token

    # ── File Paths ────────────────────────────────────────────────────────────
    REPOS_CLONE_PATH: str = "./cloned_repos"   # Where repos get cloned to
    CHROMA_DB_PATH: str = "./chroma_store"      # Where ChromaDB persists data

    # ── App Behaviour ─────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LLM_MODEL: str = "llama-3.3-70b-versatile"  # Best Groq model for code
    MAX_RETRIES: int = 3

    # ── Pydantic Config ───────────────────────────────────────────────────────
    model_config = SettingsConfigDict(
        # Look for .env in current dir first, then parent dir.
        # This handles being launched from either /backend or the project root.
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=True,      # GROQ_API_KEY ≠ groq_api_key
    )


@lru_cache()  # ← This decorator means: "call this once, cache the result forever"
def get_settings() -> Settings:
    """
    Returns the singleton Settings instance.
    Use this function everywhere instead of creating Settings() directly.
    """
    return Settings()
