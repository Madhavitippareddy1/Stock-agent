from stock_agent.agents.rag_agent import RagAgent, _retrieve


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


class FakeReports:
    def list_reports(self, tickers):
        raise AssertionError("S3 must not be searched when a document is uploaded")


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
