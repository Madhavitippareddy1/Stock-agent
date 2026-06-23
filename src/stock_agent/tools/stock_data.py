import re
from difflib import get_close_matches
from typing import Any

import pandas as pd
import yfinance as yf

from stock_agent.config import Settings, get_settings


COMPANY_SYMBOL_ALIASES = {
    "nvidia": "NVDA",
    "amazon": "AMZN",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "broadcom": "AVGO",
    "tesla": "TSLA",
    "costco": "COST",
    "netflix": "NFLX",
    "cisco": "CSCO",
    "cisco systems": "CSCO",
}

SYMBOL_SEARCH_STOP_WORDS = {
    "and",
    "compare",
    "comparison",
    "current",
    "for",
    "give",
    "is",
    "latest",
    "live",
    "market",
    "of",
    "performance",
    "please",
    "price",
    "quote",
    "share",
    "show",
    "stock",
    "the",
    "trading",
    "versus",
    "what",
    "with",
}

PRIMARY_US_EXCHANGES = {"NMS", "NYQ", "NGM", "NCM", "ASE", "PCX", "BTS"}


def resolve_alias_symbols(query: str, limit: int = 5) -> tuple[str, ...]:
    """Resolve multiple company names, including close single-word misspellings."""
    normalized = query.lower()
    matches: list[tuple[int, str]] = []

    for company_name, symbol in COMPANY_SYMBOL_ALIASES.items():
        phrase_match = re.search(rf"\b{re.escape(company_name)}\b", normalized)
        if phrase_match:
            matches.append((phrase_match.start(), symbol))

    single_word_aliases = {
        name: symbol for name, symbol in COMPANY_SYMBOL_ALIASES.items() if " " not in name
    }
    for token_match in re.finditer(r"\b[a-z]{4,}\b", normalized):
        token = token_match.group()
        if token in SYMBOL_SEARCH_STOP_WORDS or token in single_word_aliases:
            continue
        close = get_close_matches(token, single_word_aliases, n=1, cutoff=0.78)
        if close:
            matches.append((token_match.start(), single_word_aliases[close[0]]))

    ordered = [symbol for _, symbol in sorted(matches)]
    return tuple(dict.fromkeys(ordered))[:limit]


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
        if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", normalized):
            raise ValueError(f"{normalized} is not a valid stock symbol")
        return normalized

    def search_symbols(self, query: str, limit: int = 3) -> tuple[str, ...]:
        alias_symbols = resolve_alias_symbols(query, limit=limit)
        if alias_symbols:
            return alias_symbols

        comparison_parts = self._comparison_parts(query)
        if len(comparison_parts) > 1:
            symbols = [
                symbol
                for part in comparison_parts
                if (symbol := self._search_best_symbol(part)) is not None
            ]
            return tuple(dict.fromkeys(symbols))[:limit]

        symbol = self._search_best_symbol(query)
        return (symbol,) if symbol else ()

    def _comparison_parts(self, query: str) -> tuple[str, ...]:
        if not re.search(r"\b(compare|comparison|versus|vs)\b", query, re.IGNORECASE):
            return ()
        cleaned = re.sub(
            r"\b(compare|comparison|between)\b",
            " ",
            query,
            flags=re.IGNORECASE,
        )
        parts = re.split(r"\b(?:with|versus|vs|and)\b", cleaned, flags=re.IGNORECASE)
        return tuple(part.strip(" ,") for part in parts if part.strip(" ,"))

    def _search_best_symbol(self, query: str) -> str | None:
        search_query = re.sub(
            r"\b(current|latest|live|stock|share|price|quote|market|performance|trading|please|show|give|what|is|the|for|of)\b",
            " ",
            query,
            flags=re.IGNORECASE,
        )
        search_query = " ".join(search_query.split()) or query
        search = yf.Search(search_query, max_results=10, news_count=0)
        equity_quotes = [
            quote
            for quote in search.quotes
            if str(quote.get("quoteType", "")).upper() in {"EQUITY", "ETF"}
        ]
        equity_quotes.sort(
            key=lambda quote: (
                str(quote.get("exchange", "")).upper() not in PRIMARY_US_EXCHANGES,
                -float(quote.get("score") or 0),
            )
        )
        for quote in equity_quotes:
            symbol = quote.get("symbol")
            if not symbol:
                continue
            try:
                return self.validate_ticker(symbol)
            except ValueError:
                continue
        return None

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
