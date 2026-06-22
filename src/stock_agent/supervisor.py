import re

from stock_agent.agents.rag_agent import RagAgent
from stock_agent.agents.stock_agent import StockDataAgent
from stock_agent.config import Settings, get_settings
from stock_agent.models import AgentResult, ResearchResult


def find_tickers(value: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    tokens = set(re.findall(r"\b[A-Za-z]{2,5}\b", value.upper()))
    return tuple(ticker for ticker in allowed if ticker in tokens)


def extract_tickers(question: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    return find_tickers(question, allowed) or allowed


def _requests_live_stock_data(question: str, has_document: bool) -> bool:
    lowered = question.lower()
    if has_document:
        return any(
            phrase in lowered
            for phrase in (
                "price",
                "quote",
                "market price",
                "current market",
                "current stock",
                "performance",
                "trading",
            )
        )
    return any(
        word in lowered for word in ("price", "stock", "market", "quote", "performance")
    )


class SupervisorAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        rag_agent: RagAgent | None = None,
        stock_agent: StockDataAgent | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.rag_agent = rag_agent or RagAgent()
        self.stock_agent = stock_agent or StockDataAgent()

    def route(self, question: str, has_document: bool) -> tuple[str, ...]:
        lowered = question.lower()
        routes = []
        if has_document or any(word in lowered for word in ("report", "revenue", "filing", "10-k", "10-q")):
            routes.append("rag")
        if _requests_live_stock_data(question, has_document):
            routes.append("stock")
        return tuple(routes or ("rag", "stock"))

    def run(
        self,
        question: str,
        uploaded_content: bytes | None = None,
        uploaded_content_type: str = "application/pdf",
        uploaded_filename: str = "",
        selected_tickers: tuple[str, ...] | None = None,
    ) -> ResearchResult:
        tickers = (
            find_tickers(question, self.settings.tickers)
            or find_tickers(uploaded_filename, self.settings.tickers)
            or selected_tickers
            or self.settings.tickers
        )
        sections: list[AgentResult] = []
        for route in self.route(question, uploaded_content is not None):
            try:
                if route == "rag":
                    sections.append(
                        self.rag_agent.run(
                            question, tickers, uploaded_content, uploaded_content_type
                        )
                    )
                elif route == "stock":
                    sections.append(self.stock_agent.run(tickers))
            except Exception as exc:
                sections.append(
                    AgentResult(
                        agent=f"{route.title()} Agent",
                        answer=f"Service unavailable: {exc}",
                    )
                )
        answer = "\n\n".join(f"### {item.agent}\n{item.answer}" for item in sections)
        sources = list(dict.fromkeys(source for item in sections for source in item.sources))
        return ResearchResult(answer=answer, sections=sections, sources=sources)


def build_supervisor(settings: Settings | None = None) -> SupervisorAgent:
    settings = settings or get_settings()
    from stock_agent.services.bedrock_service import BedrockService
    from stock_agent.services.s3_reports import S3ReportRepository
    from stock_agent.services.vector_store import OpenSearchVectorStore

    reports = S3ReportRepository(settings) if settings.reports_bucket else None
    bedrock = BedrockService(settings) if settings.reports_bucket else None
    vector_store = (
        OpenSearchVectorStore(settings) if settings.opensearch_endpoint else None
    )
    return SupervisorAgent(
        settings=settings,
        rag_agent=RagAgent(
            reports=reports,
            bedrock=bedrock,
            vector_store=vector_store,
        ),
    )


def run_stock_research(
    question: str,
    pdf_bytes: bytes | None = None,
    uploaded_filename: str = "",
    selected_tickers: tuple[str, ...] | None = None,
) -> ResearchResult:
    return build_supervisor().run(
        question=question,
        uploaded_content=pdf_bytes,
        uploaded_filename=uploaded_filename,
        selected_tickers=selected_tickers,
    )
