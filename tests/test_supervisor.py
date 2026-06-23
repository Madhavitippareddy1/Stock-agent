from stock_agent.models import AgentResult
from stock_agent.supervisor import SupervisorAgent, extract_tickers


class FakeAgent:
    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, *args, **kwargs) -> AgentResult:
        return AgentResult(agent=self.name, answer=f"{self.name} result")

    def resolve_tickers(self, question: str) -> tuple[str, ...]:
        return ()


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


def test_supervisor_routes_recent_company_updates_to_stock_agent() -> None:
    supervisor = SupervisorAgent(
        rag_agent=FakeAgent("RAG"),
        stock_agent=FakeAgent("Stock"),
    )
    result = supervisor.run("Recent updates and performance for AAPL")
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

    def resolve_tickers(self, question: str) -> tuple[str, ...]:
        if "Cisco Systems" in question:
            return ("CSCO",)
        if "CSX Corporation" in question:
            return ("CSX",)
        if "PepsiCo" in question:
            return ("PEP",)
        if "nvdia" in question and "amzon" in question:
            return ("NVDA", "AMZN")
        return ()


class RagTickerCaptureAgent:
    def __init__(self) -> None:
        self.tickers = ()

    def run(self, question, tickers, uploaded_content=None, uploaded_content_type=None):
        self.tickers = tickers
        return AgentResult(agent="RAG", answer="RAG result")


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


def test_company_name_outside_top_ten_resolves_to_single_symbol() -> None:
    stock_agent = TickerCaptureAgent()
    supervisor = SupervisorAgent(
        rag_agent=FakeAgent("RAG"),
        stock_agent=stock_agent,
    )
    result = supervisor.run(
        "Cisco Systems share price",
        uploaded_content=b"Unrelated COST annual report",
        uploaded_content_type="text/plain",
        uploaded_filename="COST-annual-financial-report.pdf",
        selected_tickers=("NVDA", "GOOGL", "AAPL"),
    )
    assert [section.agent for section in result.sections] == ["Stock"]
    assert stock_agent.tickers == ("CSCO",)


def test_misspelled_comparison_scopes_both_agents_to_two_stocks() -> None:
    stock_agent = TickerCaptureAgent()
    rag_agent = RagTickerCaptureAgent()
    supervisor = SupervisorAgent(
        rag_agent=rag_agent,
        stock_agent=stock_agent,
    )

    result = supervisor.run(
        "compare nvdia with amzon",
        selected_tickers=("NVDA", "GOOGL", "AAPL", "MSFT", "AMZN"),
    )

    assert [section.agent for section in result.sections] == ["RAG", "Stock"]
    assert rag_agent.tickers == ("NVDA", "AMZN")
    assert stock_agent.tickers == ("NVDA", "AMZN")


def test_full_company_name_price_request_uses_one_resolved_symbol() -> None:
    stock_agent = TickerCaptureAgent()
    supervisor = SupervisorAgent(
        rag_agent=FakeAgent("RAG"),
        stock_agent=stock_agent,
    )

    result = supervisor.run(
        "CSX Corporation stock price",
        selected_tickers=("NVDA", "GOOGL", "AAPL"),
    )

    assert [section.agent for section in result.sections] == ["Stock"]
    assert stock_agent.tickers == ("CSX",)


def test_pepsico_buying_question_uses_only_pep() -> None:
    stock_agent = TickerCaptureAgent()
    supervisor = SupervisorAgent(
        rag_agent=FakeAgent("RAG"),
        stock_agent=stock_agent,
    )

    result = supervisor.run(
        "is buying PepsiCo shares right time or not",
        selected_tickers=("NVDA", "GOOGL", "AAPL"),
    )

    assert [section.agent for section in result.sections] == ["Stock"]
    assert stock_agent.tickers == ("PEP",)


def test_unresolved_stock_question_does_not_fall_back_to_top_ten() -> None:
    stock_agent = TickerCaptureAgent()
    supervisor = SupervisorAgent(
        rag_agent=FakeAgent("RAG"),
        stock_agent=stock_agent,
    )

    supervisor.run(
        "Should I buy an unknown company stock?",
        selected_tickers=("NVDA", "GOOGL", "AAPL"),
    )

    assert stock_agent.tickers == ()
