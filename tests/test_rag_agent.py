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
