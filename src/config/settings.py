from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

CONFIG_PATH = Path("config/config.yaml")
ENV_PATH = Path(".env")


class Secrets(BaseSettings):
    """
    Environment-based configuration loaded from .env.
    """

    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    cohere_api_key: str = Field(..., env="COHERE_API_KEY")

    qdrant_api_key: str | None = Field(None, env="QDRANT_API_KEY")
    qdrant_url: str = Field("http://localhost:6333", env="QDRANT_URL")

    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    elasticsearch_url: str = Field("http://localhost:9200", env="ELASTICSEARCH_URL")
    elasticsearch_api_key: str | None = Field(None, env="ELASTICSEARCH_API_KEY")

    encryption_key: str | None = Field(None, env="ENCRYPTION_KEY")
    jwt_secret_key: str | None = Field(None, env="JWT_SECRET_KEY")

    neo4j_uri: str = Field("neo4j://localhost:7687", env="NEO4J_URI")
    neo4j_username: str = Field("neo4j", env="NEO4J_USERNAME")
    neo4j_password: str = Field("password", env="NEO4J_PASSWORD")
    neo4j_database: str = Field("neo4j", env="NEO4J_DATABASE")

    app_env: str = Field("development", env="APP_ENV")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = str(ENV_PATH)
        env_file_encoding = "utf-8"
        case_sensitive = False


class AppConfig:
    """
    YAML config with typed section access.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @property
    def app(self) -> dict[str, Any]:
        return self._data["app"]

    @property
    def knowledge_base(self) -> dict[str, Any]:
        return self._data["knowledge_base"]

    @property
    def ingestion(self) -> dict[str, Any]:
        return self._data["ingestion"]

    @property
    def chunking(self) -> dict[str, Any]:
        return self._data["chunking"]

    @property
    def embeddings(self) -> dict[str, Any]:
        return self._data["embeddings"]

    @property
    def vector_store(self) -> dict[str, Any]:
        return self._data["vector_store"]

    @property
    def query(self) -> dict[str, Any]:
        return self._data["query"]

    @property
    def retrieval(self) -> dict[str, Any]:
        return self._data["retrieval"]

    @property
    def generation(self) -> dict[str, Any]:
        return self._data["generation"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self._data["evaluation"]

    @property
    def operations(self) -> dict[str, Any]:
        return self._data["operations"]

    @property
    def log(self) -> dict[str, Any]:
        return self._data["log"]

    @property
    def graphrag(self) -> dict[str, Any]:
        return self._data.get("graphrag", {})

    @property
    def rlhf(self) -> dict[str, Any]:
        return self._data.get("rlhf", {})

    @property
    def tuning(self) -> dict[str, Any]:
        return self._data.get("tuning", {})

    def get(self, key: str, default: Any | None = None) -> Any:
        """Dot-path lookup (e.g. 'ingestion.versioning.ttl_days')."""
        keys = key.split(".")
        node = self._data
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, default)
        return node


def load_yaml_config() -> AppConfig:
    """Load YAML configuration file."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.yaml not found at: {CONFIG_PATH}")

    with open(CONFIG_PATH, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    return AppConfig(raw)


@lru_cache(maxsize=1)
def get_secrets() -> Secrets:
    """Load and cache environment-based secrets."""
    return Secrets()


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load and cache application configuration."""
    return load_yaml_config()
