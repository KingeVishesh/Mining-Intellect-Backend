"""
Mining Intellect Backend v2 — Configuration
All settings loaded from environment variables via pydantic-settings.
"""
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    grok_api_key: Optional[str] = Field(default=None, alias="GROK_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")

    # Exa
    exa_api_key: Optional[str] = Field(default=None, alias="EXA_API_KEY")

    # Supabase
    supabase_url: Optional[str] = Field(default=None, alias="SUPABASE_URL")
    supabase_service_role_key: Optional[str] = Field(default=None, alias="SUPABASE_SERVICE_ROLE_KEY")

    # LangSmith
    langchain_api_key: Optional[str] = Field(default=None, alias="LANGCHAIN_API_KEY")
    langchain_tracing_v2: bool = Field(default=True, alias="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field(default="mining-intellect-v2", alias="LANGCHAIN_PROJECT")
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        alias="LANGCHAIN_ENDPOINT",
    )

    # App
    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


settings = Settings()
