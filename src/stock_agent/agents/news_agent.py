from stock_agent.models import AgentResult
from stock_agent.tools.news_api import NewsApiTool


class NewsAgent:
    def __init__(self, tool: NewsApiTool | None = None) -> None:
        self.tool = tool or NewsApiTool()

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
            data={"articles": articles},
        )
