"""Small command-line smoke test for the supervisor."""

from stock_agent.supervisor import run_stock_research


def main() -> None:
    result = run_stock_research("Show AAPL stock price and recent news", selected_tickers=("AAPL",))
    print(result.answer)


if __name__ == "__main__":
    main()
