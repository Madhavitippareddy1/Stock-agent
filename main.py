"""Command-line development entry point."""

from stock_agent.supervisor import run_stock_research


def main() -> None:
    result = run_stock_research("Summarize the latest portfolio risks.")
    print(result.answer)


if __name__ == "__main__":
    main()
