"""Single source of truth for all environment-driven configuration. I am not going for os/load_dotenv everywhere it sucks. 
"""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Required API credentials ────────────────────────────────────────────
    azure_openai_api_key: str = Field(..., alias="AZURE_OPENAI_API_KEY")
    azure_openai_api_version: str =Field(..., alias="AZURE_OPENAI_API_VERSION")
    azure_openai_endpoint: str = Field(..., alias="AZURE_OPENAI_ENDPOINT")
    tavily_api_key: str = Field(..., alias="TAVILY_API_KEY")
    qdrant_url: str = Field(..., alias="QDRANT_URL")
    qdrant_api_key: str = Field(..., alias="QDRANT_API_KEY")

    # ── Model configuration──────────────────────────────────────────────────
    azure_openai_chat_deployment: str = Field(default="gpt-5.4-mini", alias="AZURE_OPENAI_CHAT_DEPLOYMENT")
    azure_openai_embedding_deployment: str = Field(default="text-embedding-3-small", alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    # ── Storage paths ────────────────────────────────────────────────────────
    sessions_file: Path = Field(default=Path("sessions.json"), alias="RePaAs_SESSIONS_FILE")
    checkpoint_db_path: Path = Field(default=Path("checkpoints.db"), alias="RePaAs_CHECKPOINT_DB")
    embedding_cache_dir: Path = Field(
        default=Path("./embedding_cache/"), alias="RePaAs_EMBEDDING_CACHE_DIR"
    )

    # ── Feature toggles ──────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="RePaAs_LOG_LEVEL")
    log_json: bool = Field(default=False, alias="RePaAs_LOG_JSON")


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — construct once, reuse everywhere."""
    return Settings()
