from stock_agent.models import AgentResult


def research_news(question: str) -> AgentResult:
    return AgentResult("### News\nNews provider integration is ready to be implemented.")
