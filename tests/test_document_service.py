from stock_agent.services.document_service import chunk_text, extract_text


def test_extract_plain_text() -> None:
    assert extract_text(b"quarterly revenue increased", "text/plain") == (
        "quarterly revenue increased"
    )


def test_chunk_text_overlaps_and_preserves_content() -> None:
    chunks = chunk_text("A" * 2500, chunk_size=1000, overlap=100)
    assert len(chunks) == 3
    assert all(len(chunk) <= 1000 for chunk in chunks)
