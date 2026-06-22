from stock_agent.agents.stock_agent import StockDataAgent
from stock_agent.config import Settings
from stock_agent.tools.stock_data import YahooStockTool


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
