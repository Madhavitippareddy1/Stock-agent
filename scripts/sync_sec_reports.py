"""Download recent NASDAQ-10 SEC filings and store them in S3."""

from stock_agent.config import get_settings
from stock_agent.services.s3_reports import S3ReportRepository
from stock_agent.tools.sec_edgar import SecEdgarTool


def main() -> None:
    settings = get_settings()
    repository = S3ReportRepository(settings)
    uploaded = SecEdgarTool(settings).sync_to_s3(repository)
    print(f"Uploaded {len(uploaded)} filings to s3://{settings.reports_bucket}/")


if __name__ == "__main__":
    main()
