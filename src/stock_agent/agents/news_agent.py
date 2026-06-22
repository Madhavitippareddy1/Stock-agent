from stock_agent.models import AgentResult
from stock_agent.config import Settings, get_settings
from stock_agent.tools.news_api import NewsApiTool
from stock_agent.tools.yahoo_news import YahooNewsTool


class NewsAgent:
    def __init__(
        self,
        tool: NewsApiTool | YahooNewsTool | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.tool = tool or (
            NewsApiTool(self.settings)
            if self.settings.news_api_key
            else YahooNewsTool(self.settings)
        )

    def run(self, tickers: tuple[str, ...]) -> AgentResult:
        articles = self.tool.search(" OR ".join(tickers), page_size=10)
        lines = []
        sources = []
        for article in articles:
            title = article.get("title") or "Untitled"
            publisher = (article.get("source") or {}).get("name") or "Unknown"
            url = article.get("url") or ""
            lines.append(f"- **{title}** — {publisher}")
            if url:
                sources.append(url)
        return AgentResult(
            agent="News Agent",
            answer="\n".join(lines) if lines else "No recent articles were returned.",
            sources=sources,
            data={
                "articles": articles,
                "provider": "NewsAPI" if self.settings.news_api_key else "Yahoo Finance",
            },
        )
