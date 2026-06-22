import pandas as pd

from stock_agent.agents.stock_agent import StockDataAgent
from stock_agent.config import Settings
from stock_agent.tools.stock_data import YahooStockTool, normalize_snapshot_frame


class FakeStockTool:
    def quote(self, ticker: str):
        return {"ticker": ticker, "price": 210.5, "currency": "USD"}


def test_stock_agent_formats_price() -> None:
    result = StockDataAgent(FakeStockTool()).run(("AAPL",))
    assert "$210.50" in result.answer


def test_yahoo_tool_rejects_ticker_outside_universe() -> None:
    tool = YahooStockTool(Settings(stock_universe="AAPL,MSFT"))
    try:
        tool.validate_ticker("IBM")
        raise AssertionError("Expected validation failure")
    except ValueError as exc:
        assert "outside" in str(exc)


def test_snapshot_normalization_handles_error_and_missing_rows() -> None:
    frame = pd.DataFrame(
        [
            {"ticker": "AAPL", "price": 210.5, "market_cap": 3_000_000},
            {"ticker": "NVDA", "price": None, "error": "vendor unavailable"},
            {"ticker": "MSFT", "price": "invalid", "market_cap": "unknown"},
        ]
    )
    normalized = normalize_snapshot_frame(frame)
    assert normalized.loc[0, "price"] == 210.5
    assert pd.isna(normalized.loc[1, "price"])
    assert pd.isna(normalized.loc[2, "price"])
    assert pd.isna(normalized.loc[2, "market_cap"])
