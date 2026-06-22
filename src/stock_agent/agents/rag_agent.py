import re
from collections import Counter

from stock_agent.models import AgentResult
from stock_agent.services.bedrock_service import BedrockService
from stock_agent.services.document_service import chunk_text, extract_text
from stock_agent.services.s3_reports import S3ReportRepository


def _terms(value: str) -> Counter[str]:
    return Counter(re.findall(r"[a-zA-Z]{3,}", value.lower()))


def _retrieve(question: str, chunks: list[tuple[str, str]], limit: int = 6) -> list[tuple[str, str]]:
    query_terms = _terms(question)
    scored = []
    for source, chunk in chunks:
        score = sum((_terms(chunk) & query_terms).values())
        scored.append((score, source, chunk))
    return [(source, chunk) for score, source, chunk in sorted(scored, reverse=True)[:limit] if score]


class RagAgent:
    def __init__(
        self,
        reports: S3ReportRepository | None = None,
        bedrock: BedrockService | None = None,
    ) -> None:
        self.reports = reports
        self.bedrock = bedrock

    def run(
        self,
        question: str,
        tickers: tuple[str, ...],
        uploaded_content: bytes | None = None,
        uploaded_content_type: str = "application/pdf",
    ) -> AgentResult:
        documents: list[tuple[str, str]] = []
        if uploaded_content:
            documents.append(
                ("Uploaded document", extract_text(uploaded_content, uploaded_content_type))
            )
        if self.reports:
            for report in self.reports.list_reports(tickers):
                content, content_type = self.reports.download(report.key)
                documents.append((f"s3://{self.reports.settings.reports_bucket}/{report.key}", extract_text(content, content_type)))

        chunks = [
            (source, chunk)
            for source, text in documents
            for chunk in chunk_text(text)
        ]
        matches = _retrieve(question, chunks)
        if not matches:
            return AgentResult(
                agent="RAG Agent",
                answer="No relevant uploaded or S3 financial-report content was found.",
            )

        context = "\n\n".join(f"[{source}]\n{chunk}" for source, chunk in matches)
        if self.bedrock:
            answer = self.bedrock.answer(
                question,
                context,
                (
                    "You are a financial-report research assistant. Answer only from the supplied "
                    "context, distinguish facts from interpretation, cite source labels, and never "
                    "present the response as investment advice."
                ),
            )
        else:
            answer = "Relevant report excerpts:\n\n" + "\n\n".join(
                f"- **{source}**: {chunk[:500]}…" for source, chunk in matches
            )
        return AgentResult(
            agent="RAG Agent",
            answer=answer,
            sources=list(dict.fromkeys(source for source, _ in matches)),
        )
