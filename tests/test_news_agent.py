from stock_agent.agents.news_agent import NewsAgent


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
    result = NewsAgent(FakeNewsTool()).run(("AAPL",))
    assert "Apple releases results" in result.answer
    assert result.sources == ["https://example.com/apple"]
