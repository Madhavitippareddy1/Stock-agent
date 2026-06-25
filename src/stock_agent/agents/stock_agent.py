from numbers import Real

from stock_agent.models import AgentResult
from stock_agent.observability import Observability, get_observability
from stock_agent.tools.stock_data import YahooStockTool


class StockDataAgent:
    def __init__(
        self,
        tool: YahooStockTool | None = None,
        observability: Observability | None = None,
    ) -> None:
        self.tool = tool or YahooStockTool()
        self.observability = observability or get_observability()

    def run(self, tickers: tuple[str, ...]) -> AgentResult:
        with self.observability.observe(
            "stock-data-agent-run",
            as_type="agent",
            input={"tickers": tickers},
            metadata={"agent": "Stock Data Agent", "provider": "Yahoo Finance"},
        ) as agent_span:
            if not tickers:
                result = AgentResult(
                    agent="Stock Data Agent",
                    answer=(
                        "I could not identify the requested company or ticker. "
                        "Please enter a company name such as PepsiCo or a symbol such as PEP."
                    ),
                    sources=["Yahoo Finance"],
                    data={"quotes": []},
                )
                if agent_span:
                    agent_span.update(output={"quote_count": 0, "tickers": []})
                return result
            quotes = []
            for ticker in tickers:
                try:
                    with self.observability.observe(
                        "yahoo-finance-quote",
                        as_type="tool",
                        input={"ticker": ticker},
                        metadata={"provider": "Yahoo Finance"},
                    ) as tool_span:
                        quote = self.tool.quote(ticker)
                        quote["performance"] = self._load_performance(ticker)
                        quote["recent_updates"] = self._load_recent_updates(ticker)
                        quotes.append(quote)
                        if tool_span:
                            tool_span.update(
                                output={
                                    "ticker": ticker,
                                    "price_available": isinstance(
                                        quote.get("price"), (int, float)
                                    ),
                                    "currency": quote.get("currency"),
                                    "market_cap_available": isinstance(
                                        quote.get("market_cap"), (int, float)
                                    ),
                                }
                            )
                except Exception as exc:
                    quotes.append({"ticker": ticker, "error": str(exc)})
            rows = []
            for quote in quotes:
                if quote.get("error"):
                    rows.append(f"- **{quote['ticker']}**: unavailable")
                else:
                    rows.append(self._format_stock_details(quote))
            result = AgentResult(
                agent="Stock Data Agent",
                answer="\n".join(rows),
                sources=["Yahoo Finance"],
                data={"quotes": quotes},
            )
            price_summary = self._price_summary(quotes)
            with self.observability.observe(
                "stock-price-summary",
                as_type="span",
                input={
                    "tickers": [quote.get("ticker") for quote in quotes],
                    "quote_count": len(quotes),
                },
                metadata={
                    "metric": "average_current_price",
                    "provider": "Yahoo Finance",
                    "currency": price_summary.get("currency"),
                },
            ) as price_span:
                if price_span:
                    price_span.update(output=price_summary)
                if price_summary["available_price_count"]:
                    self.observability.score_current_span(
                        "stock_average_price",
                        float(price_summary["average_price"]),
                        comment="Average current price across requested stocks.",
                        metadata={
                            "tickers": price_summary["tickers"],
                            "currency": price_summary["currency"],
                            "mixed_currencies": price_summary["mixed_currencies"],
                        },
                    )
                if quotes:
                    availability_ratio = price_summary["available_price_count"] / len(quotes)
                    self.observability.score_current_span(
                        "stock_price_available_ratio",
                        availability_ratio,
                        comment="Share of requested stock quotes with current price available.",
                        metadata={
                            "available_price_count": price_summary["available_price_count"],
                            "quote_count": len(quotes),
                            "missing_price_count": price_summary["missing_price_count"],
                        },
                    )
            if agent_span:
                agent_span.update(
                    output={
                        "quote_count": len(quotes),
                        "tickers": [quote.get("ticker") for quote in quotes],
                        "error_count": sum(1 for quote in quotes if quote.get("error")),
                        "price_summary": price_summary,
                    }
                )
            return result

    def _price_summary(self, quotes: list[dict]) -> dict:
        valid_quotes = [
            quote
            for quote in quotes
            if self._is_number(quote.get("price")) and not quote.get("error")
        ]
        prices = [float(quote["price"]) for quote in valid_quotes]
        currencies = sorted(
            {
                str(quote.get("currency") or "USD")
                for quote in valid_quotes
                if quote.get("currency") or quote.get("price") is not None
            }
        )
        if not prices:
            return {
                "available_price_count": 0,
                "missing_price_count": len(quotes),
                "average_price": None,
                "min_price": None,
                "max_price": None,
                "currency": currencies[0] if len(currencies) == 1 else None,
                "mixed_currencies": len(currencies) > 1,
                "tickers": [quote.get("ticker") for quote in quotes],
            }
        average_price = sum(prices) / len(prices)
        return {
            "available_price_count": len(prices),
            "missing_price_count": len(quotes) - len(prices),
            "average_price": round(average_price, 4),
            "min_price": round(min(prices), 4),
            "max_price": round(max(prices), 4),
            "currency": currencies[0] if len(currencies) == 1 else None,
            "mixed_currencies": len(currencies) > 1,
            "tickers": [quote.get("ticker") for quote in valid_quotes],
        }

    def _load_performance(self, ticker: str) -> dict:
        performance_tool = getattr(self.tool, "performance", None)
        if not callable(performance_tool):
            return {}
        try:
            with self.observability.observe(
                "yahoo-finance-performance",
                as_type="tool",
                input={"ticker": ticker, "period": "1y"},
                metadata={"provider": "Yahoo Finance"},
            ) as tool_span:
                performance = performance_tool(ticker)
                if tool_span:
                    tool_span.update(output={"ticker": ticker, **performance})
                return performance
        except Exception:
            return {}

    def _load_recent_updates(self, ticker: str) -> list[dict]:
        updates_tool = getattr(self.tool, "recent_updates", None)
        if not callable(updates_tool):
            return []
        try:
            with self.observability.observe(
                "yahoo-finance-recent-updates",
                as_type="tool",
                input={"ticker": ticker, "limit": 3},
                metadata={"provider": "Yahoo Finance"},
            ) as tool_span:
                updates = updates_tool(ticker, limit=3)
                if tool_span:
                    tool_span.update(
                        output={"ticker": ticker, "update_count": len(updates)}
                    )
                return updates
        except Exception:
            return []

    def _format_stock_details(self, quote: dict) -> str:
        ticker = quote["ticker"]
        currency = quote.get("currency") or "USD"
        price = self._money(quote.get("price"))
        lines = [f"#### {ticker}", f"- **Current price:** {price} {currency}"]

        change = quote.get("change")
        change_percent = quote.get("change_percent")
        if self._is_number(change) or self._is_number(change_percent):
            lines.append(
                f"- **Daily change:** {self._signed_money(change)} "
                f"({self._percent(change_percent)})"
            )

        day_low, day_high = quote.get("day_low"), quote.get("day_high")
        if self._is_number(day_low) and self._is_number(day_high):
            lines.append(
                f"- **Today's range:** {self._money(day_low)} – {self._money(day_high)}"
            )

        year_low, year_high = quote.get("year_low"), quote.get("year_high")
        if self._is_number(year_low) and self._is_number(year_high):
            lines.append(
                f"- **52-week range:** {self._money(year_low)} – {self._money(year_high)}"
            )

        if self._is_number(quote.get("market_cap")):
            lines.append(f"- **Market capitalization:** {self._compact(quote['market_cap'])}")

        if self._is_number(quote.get("volume")):
            volume = self._compact(quote["volume"])
            average = self._compact(quote.get("average_volume"))
            suffix = f" (3-month average: {average})" if average != "N/A" else ""
            lines.append(f"- **Trading volume:** {volume}{suffix}")

        performance = quote.get("performance") or {}
        performance_items = [
            ("1 month", performance.get("one_month_percent")),
            ("3 months", performance.get("three_month_percent")),
            ("6 months", performance.get("six_month_percent")),
            (
                "1 year",
                performance.get("one_year_percent")
                if performance.get("one_year_percent") is not None
                else quote.get("year_change_percent"),
            ),
        ]
        available_performance = [
            f"{label}: {self._percent(value)}"
            for label, value in performance_items
            if self._is_number(value)
        ]
        if available_performance:
            lines.append(f"- **Price performance:** {' | '.join(available_performance)}")

        trend_parts = []
        current_price = quote.get("price")
        for label, average_key in (
            ("50-day average", "fifty_day_average"),
            ("200-day average", "two_hundred_day_average"),
        ):
            average = quote.get(average_key)
            if self._is_number(current_price) and self._is_number(average):
                direction = "above" if current_price >= average else "below"
                trend_parts.append(f"{direction} {label} ({self._money(average)})")
        if trend_parts:
            lines.append(f"- **Price trend:** {'; '.join(trend_parts)}")

        updates = quote.get("recent_updates") or []
        if updates:
            lines.append("\n**Recent updates from Yahoo Finance**")
            for index, update in enumerate(updates, start=1):
                title = update.get("title") or "Company update"
                url = update.get("url")
                linked_title = f"[{title}]({url})" if url else title
                details = " · ".join(
                    value
                    for value in (
                        update.get("provider"),
                        self._date_only(update.get("published_at")),
                    )
                    if value
                )
                lines.append(f"{index}. {linked_title}" + (f" — {details}" if details else ""))
                summary = update.get("summary")
                if summary:
                    lines.append(f"   {summary}")
        else:
            lines.append("\n_Recent company updates are temporarily unavailable._")

        return "\n".join(lines)

    @staticmethod
    def _is_number(value) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool)

    @classmethod
    def _money(cls, value) -> str:
        return f"${float(value):,.2f}" if cls._is_number(value) else "N/A"

    @classmethod
    def _signed_money(cls, value) -> str:
        return f"{float(value):+,.2f}" if cls._is_number(value) else "N/A"

    @classmethod
    def _percent(cls, value) -> str:
        return f"{float(value):+.2f}%" if cls._is_number(value) else "N/A"

    @classmethod
    def _compact(cls, value) -> str:
        if not cls._is_number(value):
            return "N/A"
        number = float(value)
        for divisor, suffix in (
            (1_000_000_000_000, "T"),
            (1_000_000_000, "B"),
            (1_000_000, "M"),
            (1_000, "K"),
        ):
            if abs(number) >= divisor:
                return f"{number / divisor:,.2f}{suffix}"
        return f"{number:,.0f}"

    @staticmethod
    def _date_only(value) -> str:
        return str(value)[:10] if value else ""

    def resolve_tickers(self, question: str) -> tuple[str, ...]:
        return self.tool.search_symbols(question, limit=5)
