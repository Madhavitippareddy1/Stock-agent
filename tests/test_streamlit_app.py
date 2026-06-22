import pandas as pd
from streamlit.testing.v1 import AppTest

from stock_agent.tools.stock_data import YahooStockTool


def test_dashboard_renders_mixed_yahoo_rows(monkeypatch) -> None:
    monkeypatch.setattr(
        YahooStockTool,
        "universe_snapshot",
        lambda self: pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "price": 210.5,
                    "previous_close": 209.0,
                    "market_cap": 3_000_000,
                },
                {"ticker": "NVDA", "price": None, "error": "vendor unavailable"},
                {"ticker": "MSFT", "price": float("nan"), "market_cap": None},
            ]
        ),
    )

    app = AppTest.from_file("app.py")
    app.run(timeout=30)
    app.button[1].click().run(timeout=30)

    assert not app.exception
    assert len(app.dataframe) == 1
