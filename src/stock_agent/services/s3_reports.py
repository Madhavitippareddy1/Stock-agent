from dataclasses import dataclass
from typing import Any, Iterable

import boto3

from stock_agent.config import Settings, get_settings


@dataclass(frozen=True)
class ReportObject:
    key: str
    ticker: str
    period: str
    year: str


class S3ReportRepository:
    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or boto3.client("s3", region_name=self.settings.aws_region)

    def _require_bucket(self) -> str:
        if not self.settings.reports_bucket:
            raise ValueError("REPORTS_BUCKET is not configured")
        return self.settings.reports_bucket

    def key(self, ticker: str, period: str, year: str, filename: str) -> str:
        return (
            f"{self.settings.reports_prefix.strip('/')}/"
            f"{ticker.upper()}/{period}/{year}/{filename}"
        )

    def upload(
        self,
        content: bytes,
        ticker: str,
        period: str,
        year: str,
        filename: str,
        content_type: str,
    ) -> str:
        key = self.key(ticker, period, year, filename)
        self.client.put_object(
            Bucket=self._require_bucket(),
            Key=key,
            Body=content,
            ContentType=content_type,
            Metadata={"ticker": ticker.upper(), "period": period, "year": year},
        )
        return key

    def list_reports(self, tickers: Iterable[str] | None = None) -> list[ReportObject]:
        allowed = {item.upper() for item in tickers or []}
        prefix = f"{self.settings.reports_prefix.strip('/')}/"
        paginator = self.client.get_paginator("list_objects_v2")
        reports: list[ReportObject] = []
        for page in paginator.paginate(Bucket=self._require_bucket(), Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                parts = key[len(prefix) :].split("/")
                if len(parts) < 4 or (allowed and parts[0].upper() not in allowed):
                    continue
                reports.append(
                    ReportObject(key=key, ticker=parts[0], period=parts[1], year=parts[2])
                )
        return reports

    def download(self, key: str) -> tuple[bytes, str]:
        response = self.client.get_object(Bucket=self._require_bucket(), Key=key)
        return response["Body"].read(), response.get("ContentType", "")
