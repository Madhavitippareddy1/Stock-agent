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


def first_value(values: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = values.get(key)
        if value is not None:
            return value
    return None


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
        price = first_value(fast, "lastPrice", "last_price")
        previous_close = first_value(
            fast,
            "previousClose",
            "previous_close",
            "regularMarketPreviousClose",
        )
        market_cap = first_value(fast, "marketCap", "market_cap")

        if price is None or previous_close is None:
            history = stock.history(period="5d", auto_adjust=False)
            closes = history["Close"].dropna() if "Close" in history else pd.Series(dtype=float)
            if price is None and not closes.empty:
                price = float(closes.iloc[-1])
            if previous_close is None and len(closes) > 1:
                previous_close = float(closes.iloc[-2])

        return {
            "ticker": symbol,
            "price": price,
            "previous_close": previous_close,
            "market_cap": market_cap,
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
