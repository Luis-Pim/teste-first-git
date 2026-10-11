from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuração via variáveis de ambiente (ou arquivo .env). Nunca coloque chaves no código."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="INVESTAI_", extra="ignore")

    database_url: str = "postgresql+psycopg://investai:investai@localhost:5432/investai"
    cors_origins: list[str] = ["http://localhost:3000"]

    # Provedores de mercado: "manual" (padrão, sem rede), "brapi"
    market_provider: str = "manual"
    brapi_token: str | None = None
    brapi_base_url: str = "https://brapi.dev/api"
    bcb_base_url: str = "https://api.bcb.gov.br/dados/serie"
    http_timeout_seconds: float = 10.0
    quote_cache_minutes: int = 15

    # Alertas
    stale_quote_days: int = 3
    upcoming_payment_days: int = 15


@lru_cache
def get_settings() -> Settings:
    return Settings()
