"""Embed S3 financial reports with Amazon Titan and index them in OpenSearch Serverless."""

import argparse

from stock_agent.config import get_settings
from stock_agent.services.bedrock_service import BedrockService
from stock_agent.services.document_service import chunk_text, extract_text
from stock_agent.services.s3_reports import S3ReportRepository
from stock_agent.services.vector_store import OpenSearchVectorStore


def index_reports(recreate: bool = False) -> int:
    settings = get_settings()
    reports = S3ReportRepository(settings)
    bedrock = BedrockService(settings)
    vector_store = OpenSearchVectorStore(settings)
    vector_store.ensure_index(recreate=recreate)

    documents = []
    for report in reports.list_reports(settings.tickers):
        content, content_type = reports.download(report.key)
        source = f"s3://{settings.reports_bucket}/{report.key}"
        for chunk_index, text in enumerate(chunk_text(extract_text(content, content_type))):
            documents.append(
                {
                    "embedding": bedrock.embed(text),
                    "text": text,
                    "source": source,
                    "ticker": report.ticker.upper(),
                    "period": report.period,
                    "year": report.year,
                    "chunk_index": chunk_index,
                }
            )

    indexed, errors = vector_store.index_documents(documents)
    if errors:
        raise RuntimeError(f"OpenSearch rejected {len(errors)} documents: {errors[:2]}")
    return indexed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the index before loading reports.",
    )
    args = parser.parse_args()
    print(f"Indexed {index_reports(recreate=args.recreate)} report chunks.")
