from stock_agent.supervisor import run_stock_research


def test_supervisor_returns_all_sections() -> None:
    result = run_stock_research("Review this stock")
    assert "Document analysis" in result.answer
    assert "News" in result.answer
    assert "Portfolio" in result.answer
    assert "Sentiment" in result.answer
