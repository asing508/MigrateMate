"""MigrateMate Configuration."""

from functools import lru_cache
from typing import Literal
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore")
    
    # Application
    APP_NAME: str = "MigrateMate"
    APP_ENV: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = True
    SECRET_KEY: str = Field(default="change-me-in-production")
    
    # PostgreSQL
    POSTGRES_USER: str = "migratemate"
    POSTGRES_PASSWORD: str = "migratemate_secret"
    POSTGRES_DB: str = "migratemate_db"
    POSTGRES_HOST: str = "127.0.0.1"
    POSTGRES_PORT: int = 5433
    
    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @computed_field
    @property
    def DATABASE_URL_SYNC(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_HTTP_PORT: int = 7333
    QDRANT_GRPC_PORT: int = 6334
    
    @computed_field
    @property
    def QDRANT_URL(self) -> str:
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_HTTP_PORT}"
    
    # Neo4j
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "migratemate_neo4j"
    NEO4J_HOST: str = "localhost"
    NEO4J_BOLT_PORT: int = 9687
    NEO4J_HTTP_PORT: int = 9474
    
    @computed_field
    @property
    def NEO4J_URI(self) -> str:
        return f"bolt://{self.NEO4J_HOST}:{self.NEO4J_BOLT_PORT}"
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6389
    REDIS_PASSWORD: str = "migratemate_redis"
    
    @computed_field
    @property
    def REDIS_URL(self) -> str:
        return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/0"
    
    # Embedding
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    GEMINI_API_KEY: str = ""

@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
