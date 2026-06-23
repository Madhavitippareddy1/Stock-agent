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
        quotes = []
        for ticker in tickers:
            try:
                with self.observability.observe(
                    "yahoo-finance-quote",
                    as_type="tool",
                    input={"ticker": ticker},
                ) as tool_span:
                    quote = self.tool.quote(ticker)
                    quotes.append(quote)
                    if tool_span:
                        tool_span.update(
                            output={
                                "ticker": ticker,
                                "price_available": isinstance(
                                    quote.get("price"), (int, float)
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
                price = quote.get("price")
                formatted = f"${price:,.2f}" if isinstance(price, (int, float)) else "N/A"
                rows.append(f"- **{quote['ticker']}**: {formatted}")
        return AgentResult(
            agent="Stock Data Agent",
            answer="\n".join(rows),
            sources=["Yahoo Finance"],
            data={"quotes": quotes},
        )

    def resolve_tickers(self, question: str) -> tuple[str, ...]:
        return self.tool.search_symbols(question, limit=1)
