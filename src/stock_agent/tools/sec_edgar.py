from dataclasses import dataclass
from typing import Any

import requests

from stock_agent.config import Settings, get_settings
from stock_agent.services.s3_reports import S3ReportRepository


CIKS = {
    "NVDA": "0001045810",
    "GOOGL": "0001652044",
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "AMZN": "0001018724",
    "META": "0001326801",
    "AVGO": "0001730168",
    "TSLA": "0001318605",
    "COST": "0000909832",
    "NFLX": "0001065280",
}


@dataclass(frozen=True)
class Filing:
    ticker: str
    form: str
    filing_date: str
    accession: str
    primary_document: str

    @property
    def period(self) -> str:
        if self.form == "10-K":
            return "annual"
        month = int(self.filing_date[5:7])
        return "half-yearly" if 5 <= month <= 8 else "quarterly"


class SecEdgarTool:
    def __init__(
        self,
        settings: Settings | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.session = session or requests.Session()

    @property
    def headers(self) -> dict[str, str]:
        return {"User-Agent": self.settings.sec_user_agent, "Accept-Encoding": "gzip, deflate"}

    def recent_filings(self, ticker: str, limit_per_form: int = 3) -> list[Filing]:
        cik = CIKS[ticker.upper()]
        response = self.session.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()
        recent: dict[str, list[Any]] = response.json()["filings"]["recent"]
        counts = {"10-K": 0, "10-Q": 0}
        filings: list[Filing] = []
        for index, form in enumerate(recent["form"]):
            if form not in counts or counts[form] >= limit_per_form:
                continue
            filings.append(
                Filing(
                    ticker=ticker.upper(),
                    form=form,
                    filing_date=recent["filingDate"][index],
                    accession=recent["accessionNumber"][index],
                    primary_document=recent["primaryDocument"][index],
                )
            )
            counts[form] += 1
        return filings

    def download(self, filing: Filing) -> bytes:
        cik = str(int(CIKS[filing.ticker]))
        accession = filing.accession.replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"
            f"{filing.primary_document}"
        )
        response = self.session.get(url, headers=self.headers, timeout=45)
        response.raise_for_status()
        return response.content

    def sync_to_s3(
        self,
        repository: S3ReportRepository,
        tickers: tuple[str, ...] | None = None,
    ) -> list[str]:
        uploaded: list[str] = []
        for ticker in tickers or self.settings.tickers:
            for filing in self.recent_filings(ticker):
                content = self.download(filing)
                year = filing.filing_date[:4]
                filename = f"{filing.form}-{filing.filing_date}-{filing.primary_document}"
                uploaded.append(
                    repository.upload(
                        content=content,
                        ticker=ticker,
                        period=filing.period,
                        year=year,
                        filename=filename,
                        content_type="text/html",
                    )
                )
        return uploaded
