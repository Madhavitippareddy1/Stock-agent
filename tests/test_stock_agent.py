import pandas as pd

from stock_agent.agents.stock_agent import StockDataAgent
from stock_agent.config import Settings
from stock_agent.tools.stock_data import (
    YahooStockTool,
    first_value,
    normalize_snapshot_frame,
    percentage_change,
    resolve_alias_symbols,
)


class FakeStockTool:
    def quote(self, ticker: str):
        return {"ticker": ticker, "price": 210.5, "currency": "USD"}


class DetailedFakeStockTool:
    def quote(self, ticker: str):
        return {
            "ticker": ticker,
            "price": 101.0,
            "previous_close": 100.0,
            "change": 1.0,
            "change_percent": 1.0,
            "day_low": 99.0,
            "day_high": 102.0,
            "year_low": 75.0,
            "year_high": 110.0,
            "market_cap": 120_000_000_000,
            "volume": 10_000_000,
            "average_volume": 8_000_000,
            "fifty_day_average": 98.0,
            "two_hundred_day_average": 90.0,
            "currency": "USD",
        }

    def performance(self, ticker: str):
        return {
            "one_month_percent": 2.5,
            "three_month_percent": 6.0,
            "six_month_percent": 10.0,
            "one_year_percent": 20.0,
        }

    def recent_updates(self, ticker: str, limit: int = 3):
        return [
            {
                "title": f"{ticker} posts an operating update",
                "provider": "Example Wire",
                "published_at": "2026-06-23T08:30:00Z",
                "url": "https://example.com/update",
                "summary": "The company reported an improvement in operations.",
            }
        ][:limit]


class FakeSearch:
    quotes = []

class InvalidThenCsxSearch:
    def __init__(self, *args, **kwargs) -> None:
        self.quotes = [
            {
                "symbol": "0HRJ.L",
                "quoteType": "EQUITY",
                "exchange": "LSE",
                "score": 30000,
            },
            {
                "symbol": "CSX",
                "quoteType": "EQUITY",
                "exchange": "NMS",
                "score": 20000,
            },
        ]


def test_stock_agent_formats_price() -> None:
    result = StockDataAgent(FakeStockTool()).run(("AAPL",))
    assert "$210.50" in result.answer


def test_stock_agent_adds_performance_trends_and_recent_updates() -> None:
    result = StockDataAgent(DetailedFakeStockTool()).run(("CSX",))
    assert "Daily change" in result.answer
    assert "52-week range" in result.answer
    assert "1 month: +2.50%" in result.answer
    assert "above 50-day average" in result.answer
    assert "CSX posts an operating update" in result.answer


def test_stock_agent_formats_two_requested_stocks_only() -> None:
    result = StockDataAgent(DetailedFakeStockTool()).run(("NVDA", "AMZN"))
    assert result.answer.count("#### ") == 2
    assert "#### NVDA" in result.answer
    assert "#### AMZN" in result.answer
    assert "GOOGL" not in result.answer


def test_percentage_change_handles_numeric_and_missing_values() -> None:
    assert percentage_change(110, 100) == 10
    assert percentage_change(None, 100) is None
    assert percentage_change(100, 0) is None


def test_yahoo_tool_accepts_valid_ticker_outside_universe() -> None:
    tool = YahooStockTool(Settings(stock_universe="AAPL,MSFT"))
    assert tool.validate_ticker("CSCO") == "CSCO"


def test_yahoo_tool_rejects_invalid_ticker_syntax() -> None:
    tool = YahooStockTool()
    try:
        tool.validate_ticker("bad ticker!")
        raise AssertionError("Expected validation failure")
    except ValueError as exc:
        assert "valid stock symbol" in str(exc)


def test_company_alias_resolves_cisco_without_vendor_search(monkeypatch) -> None:
    monkeypatch.setattr("stock_agent.tools.stock_data.yf.Search", FakeSearch)
    assert YahooStockTool().search_symbols("Cisco Systems share price", limit=1) == (
        "CSCO",
    )


def test_investment_style_pepsico_question_resolves_one_symbol(monkeypatch) -> None:
    monkeypatch.setattr("stock_agent.tools.stock_data.yf.Search", FakeSearch)
    assert YahooStockTool().search_symbols(
        "is buying PepsiCo shares right time or not",
        limit=5,
    ) == ("PEP",)


def test_empty_stock_scope_does_not_render_default_stocks() -> None:
    result = StockDataAgent(FakeStockTool()).run(())
    assert "could not identify" in result.answer
    assert "AAPL" not in result.answer


def test_multiple_misspelled_company_names_resolve_in_question_order() -> None:
    assert resolve_alias_symbols("compare nvdia with amzon") == ("NVDA", "AMZN")


def test_full_company_name_returns_one_primary_us_symbol(monkeypatch) -> None:
    monkeypatch.setattr(
        "stock_agent.tools.stock_data.yf.Search",
        InvalidThenCsxSearch,
    )

    assert YahooStockTool().search_symbols("CSX Corporation stock price", limit=5) == (
        "CSX",
    )


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


def test_first_value_supports_yfinance_camel_case_fields() -> None:
    fast_info = {
        "lastPrice": 298.01,
        "previousClose": 297.20,
        "marketCap": 4_376_979_104_991,
    }
    assert first_value(fast_info, "lastPrice", "last_price") == 298.01
    assert first_value(fast_info, "previousClose", "previous_close") == 297.20
    assert first_value(fast_info, "marketCap", "market_cap") == 4_376_979_104_991
