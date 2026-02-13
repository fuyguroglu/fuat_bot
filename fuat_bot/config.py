"""
Configuration management for Fuat_bot.
Uses pydantic-settings to load from environment variables and .env files.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Provider settings
    llm_provider: Literal["anthropic", "openai", "gemini", "ollama"] = "anthropic"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    model_name: str = "claude-sonnet-4-20250514"

    # Ollama settings (for local LLMs)
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.3:70b"

    # Paths
    workspace_dir: Path = Field(default=Path("./workspace"))
    sessions_dir: Path = Field(default=Path("./sessions"))

    # Memory settings
    memory_dir: Path = Field(default=Path("./memory"))
    memory_enabled: bool = True
    memory_injection_enabled: bool = True
    memory_working_limit: int = 10  # Max working memories to inject
    memory_facts_limit: int = 20    # Max facts to inject
    memory_semantic_limit: int = 5  # Max semantic search results

    # Embedding settings (for Phase 4 - semantic memory)
    embedding_provider: Literal["sentence-transformers", "openai"] = "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"  # sentence-transformers model

    # RAG System Settings
    rag_enabled: bool = True
    rag_chunk_size: int = 500  # Target chunk size in tokens
    rag_chunk_overlap: int = 50  # Overlap between chunks in tokens
    rag_min_chunk_size: int = 100  # Minimum chunk size to keep
    rag_retrieval_limit: int = 5  # Default number of results to return
    rag_max_retrieval_limit: int = 20  # Maximum allowed retrieval limit
    rag_rerank_enabled: bool = True  # Enable cross-encoder re-ranking
    rag_rerank_multiplier: int = 3  # Retrieve 3x results before re-ranking
    rag_rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rag_default_category: str = "documents"
    rag_categories: list[str] = Field(
        default=["regulations", "course_materials", "policies", "syllabi", "documents"]
    )

    # Google Calendar
    google_calendar_credentials_file: Path = Field(default=Path("./credentials.json"))
    google_calendar_token_file: Path = Field(default=Path("./token.json"))
    google_calendar_id: str = "primary"
    # IANA timezone name used for all calendar events, e.g. "Europe/Nicosia", "UTC", "America/New_York"
    calendar_timezone: str = "UTC"

    # Google Contacts (People API) — separate token, same credentials.json
    google_contacts_token_file: Path = Field(default=Path("./contacts_token.json"))

    # Email (SMTP + IMAP) — multi-account
    # EMAIL_ACCOUNTS is a JSON dict of named accounts, e.g.:
    # {"personal": {"address": "me@gmail.com", "password": "app-pass",
    #               "smtp_host": "smtp.gmail.com", "smtp_port": 587,
    #               "imap_host": "imap.gmail.com", "imap_port": 993}}
    email_enabled: bool = False
    email_accounts: dict = Field(default_factory=dict)
    email_default_account: str | None = None

    # Telegram Bot
    telegram_enabled: bool = False
    telegram_bot_token: str | None = None
    telegram_allowed_users: list[int] = Field(default_factory=list)
    # When True, anyone can use the bot (no allowlist). Use with caution.
    telegram_open_access: bool = False

    @field_validator("telegram_allowed_users", mode="before")
    @classmethod
    def _coerce_allowed_users(cls, v: object) -> list[int]:
        """Accept a bare int, a comma-separated string, or a JSON array.

        This lets users write either:
            TELEGRAM_ALLOWED_USERS=123456789          (single ID, no brackets)
            TELEGRAM_ALLOWED_USERS=123456789,987654   (comma-separated)
            TELEGRAM_ALLOWED_USERS=[123456789]        (JSON array — preferred)
        """
        if isinstance(v, int):
            return [v]
        if isinstance(v, list):
            return [int(x) for x in v]
        if isinstance(v, str):
            s = v.strip()
            # JSON array
            if s.startswith("["):
                import json
                return [int(x) for x in json.loads(s)]
            # Comma-separated (or single value)
            return [int(x.strip()) for x in s.split(",") if x.strip()]
        return []

    # Logging
    log_level: str = "INFO"

    def get_api_key(self) -> str:
        """Get the API key for the configured provider.

        Note: Ollama doesn't require an API key, returns 'ollama' as placeholder.
        """
        if self.llm_provider == "anthropic":
            if not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            return self.anthropic_api_key
        elif self.llm_provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY not set")
            return self.openai_api_key
        elif self.llm_provider == "gemini":
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY not set")
            return self.gemini_api_key
        elif self.llm_provider == "ollama":
            # Ollama doesn't require an API key
            return "ollama"
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")


# Global settings instance
settings = Settings()
