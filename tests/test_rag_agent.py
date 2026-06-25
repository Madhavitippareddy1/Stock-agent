from contextlib import contextmanager

from stock_agent.agents.rag_agent import RagAgent, _retrieve
from stock_agent.config import Settings
from stock_agent.services.s3_reports import ReportObject


class FakeObservation:
    def __init__(self, name: str) -> None:
        self.name = name
        self.updates = []

    def update(self, **kwargs) -> None:
        self.updates.append(kwargs)


class FakeObservability:
    def __init__(self) -> None:
        self.observations = []
        self.trace_scores = []
        self.span_scores = []

    def content(self, value, fallback):
        return fallback

    def safe_text(self, value: str, *, label: str = "text"):
        return {f"{label}_length": len(value), "content_captured": False}

    @contextmanager
    def observe(self, name: str, **kwargs):
        observation = FakeObservation(name)
        self.observations.append({"name": name, "kwargs": kwargs, "observation": observation})
        yield observation

    def score_current_trace(self, name: str, value: float, **kwargs) -> None:
        self.trace_scores.append({"name": name, "value": value, **kwargs})

    def score_current_span(self, name: str, value: float, **kwargs) -> None:
        self.span_scores.append({"name": name, "value": value, **kwargs})


def test_retrieve_ranks_relevant_chunk_first() -> None:
    chunks = [
        ("a", "The company discussed office locations."),
        ("b", "Revenue increased while gross margin declined."),
    ]
    assert _retrieve("What happened to revenue and margin?", chunks)[0][0] == "b"


def test_rag_agent_uses_uploaded_text_without_aws() -> None:
    result = RagAgent().run(
        "What happened to revenue?",
        ("AAPL",),
        b"Annual report: revenue increased by ten percent.",
        "text/plain",
    )
    assert result.agent == "RAG Agent"
    assert "Uploaded document" in result.answer


def test_uploaded_rag_records_extraction_chunking_and_retrieval_spans() -> None:
    observability = FakeObservability()
    result = RagAgent(observability=observability).run(
        "Please summarize this",
        ("AAPL",),
        b"Company AAPL generated strong operating cash flow.",
        "text/plain",
    )

    observation_names = [item["name"] for item in observability.observations]
    assert result.sources == ["Uploaded document"]
    assert "extract-uploaded-financial-report" in observation_names
    assert "chunk-financial-report-documents" in observation_names
    assert "keyword-retrieve-financial-report-chunks" in observation_names
    assert any(
        score["name"] == "rag_retrieval_match_count" and score["value"] == 1.0
        for score in observability.trace_scores
    )


class FakeReports:
    def list_reports(self, tickers):
        raise AssertionError("S3 must not be searched when a document is uploaded")


class FakeS3Reports:
    settings = Settings(reports_bucket="reports-bucket", reports_prefix="financial-reports")

    def list_reports(self, tickers):
        assert tickers == ("AAPL",)
        return [
            ReportObject(
                key="financial-reports/AAPL/annual/2026/AAPL-annual.txt",
                ticker="AAPL",
                period="annual",
                year="2026",
            )
        ]

    def download(self, key):
        assert key == "financial-reports/AAPL/annual/2026/AAPL-annual.txt"
        return b"AAPL annual report revenue increased and margin improved.", "text/plain"


class FakeBedrock:
    def embed(self, text):
        return [0.1] * 1024

    def answer(self, question, context, system_prompt):
        return context


class FakeVectorStore:
    def search(self, embedding, tickers):
        from stock_agent.services.vector_store import VectorMatch

        assert tickers == ("AAPL",)
        return [
            VectorMatch(
                text="Apple annual revenue increased.",
                source="s3://reports/AAPL/annual.json",
                ticker="AAPL",
                period="annual",
                year="2025",
                score=0.9,
            )
        ]


def test_uploaded_document_is_isolated_from_s3_reports() -> None:
    result = RagAgent(reports=FakeReports()).run(
        "Analyse this above report and give summary",
        ("AAPL", "TSLA"),
        b"Company: AAPL. Total revenue was 416 billion dollars.",
        "text/plain",
    )
    assert result.sources == ["Uploaded document"]
    assert "TSLA" not in result.answer


def test_generic_summary_request_uses_uploaded_chunks_without_keyword_overlap() -> None:
    result = RagAgent().run(
        "Please summarize this",
        ("AAPL",),
        b"Company AAPL generated strong operating cash flow.",
        "text/plain",
    )
    assert "Uploaded document" in result.answer


def test_above_document_analysis_uses_uploaded_chunks_without_keyword_overlap() -> None:
    result = RagAgent().run(
        "analyse the above document",
        ("MRVL",),
        b"Marvell Technology annual report showed data center revenue growth.",
        "text/plain",
    )
    assert result.sources == ["Uploaded document"]
    assert "Marvell Technology" in result.answer


def test_s3_rag_records_financial_report_tool_spans() -> None:
    observability = FakeObservability()
    result = RagAgent(
        reports=FakeS3Reports(),
        observability=observability,
    ).run("What happened to revenue and margin?", ("AAPL",))

    observation_names = [item["name"] for item in observability.observations]
    assert result.sources == [
        "s3://reports-bucket/financial-reports/AAPL/annual/2026/AAPL-annual.txt"
    ]
    assert "s3-list-financial-reports" in observation_names
    assert "s3-download-financial-report" in observation_names
    assert "chunk-financial-report-documents" in observation_names
    assert "keyword-retrieve-financial-report-chunks" in observation_names


def test_rag_agent_uses_ticker_filtered_vector_search() -> None:
    result = RagAgent(
        bedrock=FakeBedrock(),
        vector_store=FakeVectorStore(),
    ).run("Summarize annual revenue", ("AAPL",))
    assert result.sources == ["s3://reports/AAPL/annual.json"]
    assert "Apple annual revenue" in result.answer
