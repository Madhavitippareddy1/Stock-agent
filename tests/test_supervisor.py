from stock_agent.models import AgentResult
from stock_agent.supervisor import SupervisorAgent, extract_tickers


class FakeAgent:
    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, *args, **kwargs) -> AgentResult:
        return AgentResult(agent=self.name, answer=f"{self.name} result")


def test_extract_tickers_returns_selected_symbols() -> None:
    assert extract_tickers("Compare AAPL with NVDA", ("AAPL", "MSFT", "NVDA")) == (
        "AAPL",
        "NVDA",
    )


def test_supervisor_routes_price_request_to_stock_agent() -> None:
    supervisor = SupervisorAgent(
        rag_agent=FakeAgent("RAG"),
        news_agent=FakeAgent("News"),
        stock_agent=FakeAgent("Stock"),
    )
    result = supervisor.run("What is the AAPL stock price?", selected_tickers=("AAPL",))
    assert [section.agent for section in result.sections] == ["Stock"]
