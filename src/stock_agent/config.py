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
    embedding_dimension: int = 1024
    bedrock_input_cost_per_1m_tokens: float = 0.06
    bedrock_output_cost_per_1m_tokens: float = 0.24
    bedrock_embedding_cost_per_1m_tokens: float = 0.02
    opensearch_endpoint: str = ""
    opensearch_index: str = "financial-report-chunks"
    reports_bucket: str = ""
    reports_prefix: str = "financial-reports"
    sec_user_agent: str = "StockAgent/0.1 contact@example.com"
    database_url: str = ""
    log_level: str = "INFO"
    stock_universe: str = ",".join(NASDAQ_10)
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_environment: str = "development"
    langfuse_release: str = ""
    langfuse_capture_content: bool = False
    langfuse_sample_rate: float = 1.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(item.strip().upper() for item in self.stock_universe.split(",") if item.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
