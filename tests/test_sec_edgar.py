from stock_agent.config import Settings
from stock_agent.tools.sec_edgar import Filing, SecEdgarTool


class FakeResponse:
    content = b"<html>filing</html>"

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {
            "filings": {
                "recent": {
                    "form": ["8-K", "10-K", "10-Q"],
                    "filingDate": ["2026-01-01", "2026-02-01", "2026-05-01"],
                    "accessionNumber": ["0", "1", "2"],
                    "primaryDocument": ["8k.htm", "10k.htm", "10q.htm"],
                }
            }
        }


class FakeSession:
    def get(self, *args, **kwargs):
        return FakeResponse()


def test_sec_tool_selects_annual_and_quarterly_filings() -> None:
    tool = SecEdgarTool(Settings(sec_user_agent="test test@example.com"), FakeSession())
    filings = tool.recent_filings("AAPL")
    assert [item.form for item in filings] == ["10-K", "10-Q"]
    assert filings[1].period == "half-yearly"


def test_filing_period_for_annual_report() -> None:
    filing = Filing("AAPL", "10-K", "2026-02-01", "1", "report.htm")
    assert filing.period == "annual"
