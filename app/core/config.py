from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "NEOGULMAP Shared AI Server"
    APP_VERSION: str = "1.0.0"
    
    OPENAI_API_KEY: str

    CHAT_MODEL: str = "gpt-5.4-mini"
    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    EMBED_MODEL: str = "text-embedding-3-small"

    MYSQL_HOST: str
    MYSQL_PORT: int = 3306
    MYSQL_USER: str
    MYSQL_PASSWORD: str
    MYSQL_DB: str

    DATA_DIR: str = "./app/service/policy_chatbot/original_data"
    INDEX_DIR: str = "./app/service/policy_chatbot/vector_data"

    LOG_LEVEL: str = "INFO"

    BACKEND_CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["*"])

    SEARCH_SEMANTIC_TOP_K: int = 20
    SEARCH_BM25_TOP_K: int = 20
    SEARCH_FUSED_TOP_K: int = 20
    SEARCH_RERANK_TOP_K: int = 3
    SEARCH_RESPONSE_TOP_K: int = 3


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()