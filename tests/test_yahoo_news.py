from stock_agent.config import Settings
from stock_agent.tools.yahoo_news import YahooNewsTool


class FakeTicker:
    news = [
        {
            "id": "story-1",
            "content": {
                "title": "Apple announces results",
                "provider": {"displayName": "Example Publisher"},
                "canonicalUrl": {"url": "https://example.com/apple"},
                "pubDate": "2026-06-22T10:00:00Z",
            },
        }
    ]


def test_yahoo_news_maps_nested_payload(monkeypatch) -> None:
    monkeypatch.setattr("stock_agent.tools.yahoo_news.yf.Ticker", lambda ticker: FakeTicker())
    tool = YahooNewsTool(Settings(stock_universe="AAPL,MSFT"))
    articles = tool.search("AAPL", page_size=5)
    assert articles == [
        {
            "title": "Apple announces results",
            "source": {"name": "Example Publisher"},
            "url": "https://example.com/apple",
            "publishedAt": "2026-06-22T10:00:00Z",
            "ticker": "AAPL",
        }
    ]
