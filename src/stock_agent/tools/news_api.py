from datetime import UTC, datetime, timedelta
from typing import Any

import requests

from stock_agent.config import Settings, get_settings


class NewsApiTool:
    endpoint = "https://newsapi.org/v2/everything"

    def __init__(
        self,
        settings: Settings | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.session = session or requests.Session()

    def search(self, query: str, page_size: int = 10) -> list[dict[str, Any]]:
        if not self.settings.news_api_key:
            raise ValueError("NEWS_API_KEY is not configured")
        response = self.session.get(
            self.endpoint,
            params={
                "q": query,
                "from": (datetime.now(UTC) - timedelta(days=14)).date().isoformat(),
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": min(page_size, 20),
                "apiKey": self.settings.news_api_key,
            },
            timeout=20,
        )
        response.raise_for_status()
        return response.json().get("articles", [])
