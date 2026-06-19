from stock_agent.agents.news_agent import research_news
from stock_agent.agents.portfolio_agent import analyze_portfolio
from stock_agent.agents.rag_agent import answer_from_documents
from stock_agent.agents.sentiment_agent import analyze_sentiment
from stock_agent.models import AgentResult


def run_stock_research(question: str, pdf_bytes: bytes | None = None) -> AgentResult:
    """Route a request to specialist agents and combine their outputs."""
    results = [
        answer_from_documents(question, pdf_bytes),
        research_news(question),
        analyze_portfolio(question),
        analyze_sentiment(question),
    ]
    sections = [result.answer for result in results if result.answer]
    sources = list(dict.fromkeys(source for result in results for source in result.sources))
    return AgentResult(answer="\n\n".join(sections), sources=sources)
