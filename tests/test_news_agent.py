from stock_agent.agents.news_agent import NewsAgent
from stock_agent.config import Settings


class FakeNewsTool:
    def search(self, query: str, page_size: int = 10):
        assert "AAPL" in query
        return [
            {
                "title": "Apple releases results",
                "source": {"name": "Example News"},
                "url": "https://example.com/apple",
            }
        ]


def test_news_agent_formats_articles_and_sources() -> None:
    result = NewsAgent(FakeNewsTool(), Settings(news_api_key="test-key")).run(("AAPL",))
    assert "Apple releases results" in result.answer
    assert result.sources == ["https://example.com/apple"]


def test_news_agent_uses_yahoo_when_news_api_key_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "stock_agent.agents.news_agent.YahooNewsTool.search",
        lambda self, query, page_size: [
            {
                "title": "Fallback article",
                "source": {"name": "Yahoo Finance"},
                "url": "https://example.com/fallback",
            }
        ],
    )
    result = NewsAgent(settings=Settings(news_api_key="")).run(("AAPL",))
    assert "Fallback article" in result.answer
    assert result.data["provider"] == "Yahoo Finance"
