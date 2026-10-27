"""Settings read from environment variables (a local .env is used only for development)."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = ""
    openai_api_key: str = ""
    admin_token: str = ""
    db_connect_timeout: int = 30  # Neon may need a few seconds to wake from suspend


@lru_cache
def get_settings() -> Settings:
    return Settings()
