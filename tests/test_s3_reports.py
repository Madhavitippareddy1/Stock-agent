from io import BytesIO

from stock_agent.config import Settings
from stock_agent.services.s3_reports import S3ReportRepository


class FakePaginator:
    def paginate(self, **kwargs):
        return [
            {
                "Contents": [
                    {
                        "Key": "financial-reports/AAPL/annual/2025/10-K.html",
                    }
                ]
            }
        ]


class FakeS3:
    def __init__(self) -> None:
        self.put = None

    def put_object(self, **kwargs):
        self.put = kwargs

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return FakePaginator()

    def get_object(self, **kwargs):
        return {"Body": BytesIO(b"report"), "ContentType": "text/html"}


def test_s3_repository_upload_and_list() -> None:
    client = FakeS3()
    repository = S3ReportRepository(
        Settings(reports_bucket="reports"), client=client
    )
    key = repository.upload(b"data", "AAPL", "annual", "2025", "10-K.html", "text/html")
    assert key == "financial-reports/AAPL/annual/2025/10-K.html"
    assert client.put["Bucket"] == "reports"
    assert repository.list_reports(("AAPL",))[0].ticker == "AAPL"
