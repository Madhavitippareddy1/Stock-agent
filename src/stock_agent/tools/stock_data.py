from typing import Any

import pandas as pd
import yfinance as yf

from stock_agent.config import Settings, get_settings


def normalize_snapshot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce vendor values used by the UI to stable numeric columns."""
    normalized = frame.copy()
    for column in ("price", "previous_close", "market_cap"):
        if column in normalized:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


class YahooStockTool:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def validate_ticker(self, ticker: str) -> str:
        normalized = ticker.upper().strip()
        if normalized not in self.settings.tickers:
            raise ValueError(
                f"{normalized} is outside the configured NASDAQ-10 universe: "
                f"{', '.join(self.settings.tickers)}"
            )
        return normalized

    def quote(self, ticker: str) -> dict[str, Any]:
        symbol = self.validate_ticker(ticker)
        stock = yf.Ticker(symbol)
        fast = dict(stock.fast_info)
        return {
            "ticker": symbol,
            "price": fast.get("last_price"),
            "previous_close": fast.get("previous_close"),
            "market_cap": fast.get("market_cap"),
            "currency": fast.get("currency", "USD"),
        }

    def history(self, ticker: str, period: str = "6mo") -> pd.DataFrame:
        symbol = self.validate_ticker(ticker)
        return yf.download(symbol, period=period, auto_adjust=True, progress=False)

    def financials(self, ticker: str) -> dict[str, pd.DataFrame]:
        symbol = self.validate_ticker(ticker)
        stock = yf.Ticker(symbol)
        return {
            "annual_income": stock.financials,
            "quarterly_income": stock.quarterly_financials,
            "annual_balance_sheet": stock.balance_sheet,
            "quarterly_balance_sheet": stock.quarterly_balance_sheet,
            "annual_cashflow": stock.cashflow,
            "quarterly_cashflow": stock.quarterly_cashflow,
        }

    def universe_snapshot(self) -> pd.DataFrame:
        rows = []
        for ticker in self.settings.tickers:
            try:
                rows.append(self.quote(ticker))
            except Exception as exc:  # one vendor failure must not break the dashboard
                rows.append({"ticker": ticker, "error": str(exc)})
        return normalize_snapshot_frame(pd.DataFrame(rows))
