from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


NASDAQ_10 = (
    "NVDA",
    "GOOGL",
    "AAPL",
    "MSFT",
    "AMZN",
    "META",
    "AVGO",
    "TSLA",
    "COST",
    "NFLX",
)


class Settings(BaseSettings):
    aws_region: str = "eu-west-2"
    bedrock_model_id: str = "amazon.nova-lite-v1:0"
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    reports_bucket: str = ""
    reports_prefix: str = "financial-reports"
    sec_user_agent: str = "StockAgent/0.1 contact@example.com"
    database_url: str = ""
    log_level: str = "INFO"
    stock_universe: str = ",".join(NASDAQ_10)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(item.strip().upper() for item in self.stock_universe.split(",") if item.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
