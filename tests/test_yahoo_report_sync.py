import pandas as pd

from scripts.sync_yahoo_reports import frame_payload, reports_for


def test_frame_payload_handles_empty_frame() -> None:
    assert frame_payload(pd.DataFrame()) == {}


def test_reports_for_generates_all_periods() -> None:
    frame = pd.DataFrame(
        {
            pd.Timestamp("2026-03-31"): [10.0, 5.0],
            pd.Timestamp("2025-12-31"): [8.0, 4.0],
        },
        index=["Revenue", "Profit"],
    )
    reports = reports_for(
        {
            "annual_income": frame,
            "annual_balance_sheet": frame,
            "annual_cashflow": frame,
            "quarterly_income": frame,
            "quarterly_balance_sheet": frame,
            "quarterly_cashflow": frame,
        }
    )
    assert set(reports) == {"annual", "quarterly", "half-yearly"}
    assert reports["half-yearly"]["income_statement"]["latest_6_months"]["Revenue"] == 18.0
