"""Generate annual, quarterly and half-year reports from Yahoo Finance into S3."""

import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from stock_agent.config import get_settings
from stock_agent.services.s3_reports import S3ReportRepository
from stock_agent.tools.stock_data import YahooStockTool


def frame_payload(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    normalized = frame.copy()
    normalized.columns = [str(column) for column in normalized.columns]
    return json.loads(normalized.to_json(date_format="iso"))


def half_year_payload(statements: dict[str, pd.DataFrame]) -> dict[str, Any]:
    income = statements["quarterly_income"].iloc[:, :2].sum(axis=1).to_frame("latest_6_months")
    cashflow = (
        statements["quarterly_cashflow"].iloc[:, :2].sum(axis=1).to_frame("latest_6_months")
    )
    balance = statements["quarterly_balance_sheet"].iloc[:, :1]
    return {
        "income_statement": frame_payload(income),
        "cash_flow": frame_payload(cashflow),
        "balance_sheet_snapshot": frame_payload(balance),
    }


def reports_for(statements: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    return {
        "annual": {
            "income_statement": frame_payload(statements["annual_income"]),
            "balance_sheet": frame_payload(statements["annual_balance_sheet"]),
            "cash_flow": frame_payload(statements["annual_cashflow"]),
        },
        "quarterly": {
            "income_statement": frame_payload(statements["quarterly_income"]),
            "balance_sheet": frame_payload(statements["quarterly_balance_sheet"]),
            "cash_flow": frame_payload(statements["quarterly_cashflow"]),
        },
        "half-yearly": half_year_payload(statements),
    }


def main() -> None:
    settings = get_settings()
    repository = S3ReportRepository(settings)
    stock_tool = YahooStockTool(settings)
    year = str(datetime.now(UTC).year)
    uploaded = []

    for ticker in settings.tickers:
        statements = stock_tool.financials(ticker)
        for period, report in reports_for(statements).items():
            payload = {
                "ticker": ticker,
                "period": period,
                "generated_at": datetime.now(UTC).isoformat(),
                "source": "Yahoo Finance via yfinance",
                "statements": report,
            }
            uploaded.append(
                repository.upload(
                    content=json.dumps(payload, allow_nan=False).encode("utf-8"),
                    ticker=ticker,
                    period=period,
                    year=year,
                    filename=f"{ticker}-{period}-financial-report.json",
                    content_type="application/json",
                )
            )
            print(f"Uploaded {ticker} {period}")

    print(f"Uploaded {len(uploaded)} Yahoo financial reports")


if __name__ == "__main__":
    main()
