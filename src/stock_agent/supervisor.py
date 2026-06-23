import re

from stock_agent.agents.rag_agent import RagAgent
from stock_agent.agents.stock_agent import StockDataAgent
from stock_agent.config import Settings, get_settings
from stock_agent.models import AgentResult, ResearchResult
from stock_agent.observability import Observability, get_observability


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


def _references_uploaded_document(question: str) -> bool:
    lowered = question.lower()
    return any(
        phrase in lowered
        for phrase in (
            "above",
            "uploaded",
            "this report",
            "the report",
            "document",
            "pdf",
            "analyse",
            "analyze",
            "summarize",
            "summary",
        )
    )


class SupervisorAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        rag_agent: RagAgent | None = None,
        stock_agent: StockDataAgent | None = None,
        observability: Observability | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.rag_agent = rag_agent or RagAgent()
        self.stock_agent = stock_agent or StockDataAgent()
        self.observability = observability or get_observability()

    def route(self, question: str, has_document: bool) -> tuple[str, ...]:
        lowered = question.lower()
        routes = []
        if (has_document and _references_uploaded_document(question)) or any(
            word in lowered for word in ("report", "revenue", "filing", "10-k", "10-q")
        ):
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
        routes = self.route(question, uploaded_content is not None)
        question_tickers = find_tickers(question, self.settings.tickers)
        tickers = question_tickers
        if "stock" in routes and not question_tickers:
            try:
                tickers = self.stock_agent.resolve_tickers(question)
            except (AttributeError, ValueError):
                tickers = ()
        tickers = (
            tickers
            or find_tickers(uploaded_filename, self.settings.tickers)
            or selected_tickers
            or self.settings.tickers
        )
        trace_input = self.observability.content(
            {"question": question},
            {"question_length": len(question)},
        )
        with self.observability.observe(
            "stock-agent-research",
            as_type="agent",
            input=trace_input,
            metadata={
                "routes": routes,
                "tickers": tickers,
                "has_document": uploaded_content is not None,
                "uploaded_content_type": uploaded_content_type if uploaded_content else None,
            },
        ) as trace:
            sections: list[AgentResult] = []
            for route in routes:
                try:
                    with self.observability.observe(
                        f"{route}-agent",
                        as_type="agent",
                        metadata={"route": route, "tickers": tickers},
                    ) as agent_span:
                        if route == "rag":
                            result = self.rag_agent.run(
                                question, tickers, uploaded_content, uploaded_content_type
                            )
                        else:
                            result = self.stock_agent.run(tickers)
                        sections.append(result)
                        if agent_span:
                            agent_span.update(
                                output={
                                    "agent": result.agent,
                                    "source_count": len(result.sources),
                                }
                            )
                except Exception as exc:
                    sections.append(
                        AgentResult(
                            agent=f"{route.title()} Agent",
                            answer=f"Service unavailable: {exc}",
                        )
                    )
            answer = "\n\n".join(f"### {item.agent}\n{item.answer}" for item in sections)
            sources = list(dict.fromkeys(source for item in sections for source in item.sources))
            if trace:
                trace.update(
                    output={
                        "agents": [item.agent for item in sections],
                        "source_count": len(sources),
                    }
                )
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
