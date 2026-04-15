import os
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # openai | anthropic
    intent_provider: str = "openai"
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = Field(
        default=None,
        description="OpenAI 兼容接口 Base URL，环境变量：OPENAI_BASE_URL",
    )
    openai_model: str = "gpt-4o-mini"

    @field_validator("openai_base_url", mode="before")
    @classmethod
    def strip_openai_base_url(cls, v: object) -> object:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s or None
        return v
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # TMDB — https://developer.themoviedb.org/docs
    tmdb_api_key: Optional[str] = None
    tmdb_base_url: str = "https://api.themoviedb.org/3"
    tmdb_image_base_url: str = "https://image.tmdb.org/t/p/w500"

    # Web search: tavily | serper
    search_provider: str = "tavily"
    tavily_api_key: Optional[str] = None
    serper_api_key: Optional[str] = None
    # 搜索后用 LLM 从候选中精选播放入口；关闭则仅用规则重排后取前 5 条
    search_llm_curate: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_openai_base_url() -> Optional[str]:
    """
    供 OpenAI 官方 SDK 与 LangChain ChatOpenAI 使用的 base_url。
    优先使用配置中的 openai_base_url（来自 OPENAI_BASE_URL）；
    若未写入 .env 但进程环境中存在 OPENAI_BASE_URL / OPENAI_API_BASE，则作为后备（兼容部分部署方式）。
    """
    configured = get_settings().openai_base_url
    if configured:
        return configured
    for key in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        raw = os.getenv(key)
        if raw and str(raw).strip():
            return str(raw).strip()
    return None
