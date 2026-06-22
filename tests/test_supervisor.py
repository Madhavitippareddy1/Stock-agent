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
        stock_agent=FakeAgent("Stock"),
    )
    result = supervisor.run("What is the AAPL stock price?", selected_tickers=("AAPL",))
    assert [section.agent for section in result.sections] == ["Stock"]


def test_uploaded_report_analysis_does_not_start_stock_data_agent() -> None:
    supervisor = SupervisorAgent(
        rag_agent=FakeAgent("RAG"),
        stock_agent=FakeAgent("Stock"),
    )
    result = supervisor.run(
        "analyse above stock",
        uploaded_content=b"COST annual report",
        uploaded_content_type="text/plain",
        uploaded_filename="COST-annual-financial-report.pdf",
        selected_tickers=("NVDA", "GOOGL", "AAPL"),
    )
    assert [section.agent for section in result.sections] == ["RAG"]


class TickerCaptureAgent:
    def __init__(self) -> None:
        self.tickers = ()

    def run(self, tickers) -> AgentResult:
        self.tickers = tickers
        return AgentResult(agent="Stock", answer="Stock result")


def test_uploaded_filename_scopes_explicit_price_request() -> None:
    stock_agent = TickerCaptureAgent()
    supervisor = SupervisorAgent(
        rag_agent=FakeAgent("RAG"),
        stock_agent=stock_agent,
    )
    result = supervisor.run(
        "What is the current price for the uploaded company?",
        uploaded_content=b"COST annual report",
        uploaded_content_type="text/plain",
        uploaded_filename="COST-annual-financial-report.pdf",
        selected_tickers=("NVDA", "GOOGL", "AAPL"),
    )
    assert [section.agent for section in result.sections] == ["RAG", "Stock"]
    assert stock_agent.tickers == ("COST",)


def test_question_ticker_overrides_sidebar_scope() -> None:
    stock_agent = TickerCaptureAgent()
    supervisor = SupervisorAgent(
        rag_agent=FakeAgent("RAG"),
        stock_agent=stock_agent,
    )
    supervisor.run(
        "What is the COST stock price?",
        selected_tickers=("NVDA", "GOOGL", "AAPL"),
    )
    assert stock_agent.tickers == ("COST",)
