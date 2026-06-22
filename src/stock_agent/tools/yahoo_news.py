from typing import Any

import yfinance as yf

from stock_agent.config import Settings, get_settings


class YahooNewsTool:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def search(self, query: str, page_size: int = 10) -> list[dict[str, Any]]:
        requested = {token.strip().upper() for token in query.split("OR")}
        tickers = [ticker for ticker in self.settings.tickers if ticker in requested]
        articles: list[dict[str, Any]] = []
        seen: set[str] = set()

        for ticker in tickers or list(self.settings.tickers[:3]):
            for item in yf.Ticker(ticker).news:
                content = item.get("content") or item
                canonical = content.get("canonicalUrl") or {}
                click_through = content.get("clickThroughUrl") or {}
                url = (
                    click_through.get("url")
                    or canonical.get("url")
                    or content.get("previewUrl")
                    or item.get("link")
                    or ""
                )
                article_id = item.get("id") or url or content.get("title", "")
                if not article_id or article_id in seen:
                    continue
                seen.add(article_id)
                provider = content.get("provider") or {}
                articles.append(
                    {
                        "title": content.get("title") or item.get("title") or "Untitled",
                        "source": {
                            "name": provider.get("displayName")
                            or item.get("publisher")
                            or "Yahoo Finance"
                        },
                        "url": url,
                        "publishedAt": content.get("pubDate")
                        or item.get("providerPublishTime"),
                        "ticker": ticker,
                    }
                )
                if len(articles) >= page_size:
                    return articles
        return articles
