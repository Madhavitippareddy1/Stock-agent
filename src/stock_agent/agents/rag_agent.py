import re
from collections import Counter

from stock_agent.models import AgentResult
from stock_agent.observability import Observability, get_observability
from stock_agent.services.bedrock_service import BedrockService
from stock_agent.services.document_service import chunk_text, extract_text
from stock_agent.services.s3_reports import S3ReportRepository
from stock_agent.services.vector_store import OpenSearchVectorStore


def _terms(value: str) -> Counter[str]:
    return Counter(re.findall(r"[a-zA-Z]{3,}", value.lower()))


def _retrieve(question: str, chunks: list[tuple[str, str]], limit: int = 6) -> list[tuple[str, str]]:
    query_terms = _terms(question)
    scored = []
    for source, chunk in chunks:
        score = sum((_terms(chunk) & query_terms).values())
        scored.append((score, source, chunk))
    return [(source, chunk) for score, source, chunk in sorted(scored, reverse=True)[:limit] if score]


def _is_summary_request(question: str) -> bool:
    lowered = question.lower()
    return any(
        phrase in lowered
        for phrase in ("summarize", "summary", "analyse this", "analyze this", "above report")
    )


class RagAgent:
    def __init__(
        self,
        reports: S3ReportRepository | None = None,
        bedrock: BedrockService | None = None,
        vector_store: OpenSearchVectorStore | None = None,
        observability: Observability | None = None,
    ) -> None:
        self.reports = reports
        self.bedrock = bedrock
        self.vector_store = vector_store
        self.observability = observability or get_observability()

    def run(
        self,
        question: str,
        tickers: tuple[str, ...],
        uploaded_content: bytes | None = None,
        uploaded_content_type: str = "application/pdf",
    ) -> AgentResult:
        documents: list[tuple[str, str]] = []
        if uploaded_content:
            with self.observability.observe(
                "extract-uploaded-financial-report",
                as_type="tool",
                input={
                    "content_type": uploaded_content_type,
                    "uploaded_bytes": len(uploaded_content),
                },
                metadata={"tool": "document-text-extraction"},
            ) as extraction_span:
                uploaded_text = extract_text(uploaded_content, uploaded_content_type)
                documents.append(("Uploaded document", uploaded_text))
                if extraction_span:
                    extraction_span.update(
                        output={
                            "source": "Uploaded document",
                            "text_length": len(uploaded_text),
                        }
                    )
        elif self.vector_store and self.bedrock:
            bedrock_settings = getattr(self.bedrock, "settings", None)
            embedding_model = getattr(
                bedrock_settings,
                "bedrock_embedding_model_id",
                "amazon.titan-embed-text-v2:0",
            )
            embedding_dimension = getattr(bedrock_settings, "embedding_dimension", 1024)
            with self.observability.observe(
                "embed-rag-question",
                as_type="embedding",
                input=self.observability.content(
                    question,
                    {"text_length": len(question)},
                ),
                model=embedding_model,
                metadata={"dimensions": embedding_dimension},
            ) as embedding_span:
                embedding = self.bedrock.embed(question)
                if embedding_span:
                    embedding_span.update(
                        output={"dimensions": len(embedding)},
                        usage_details={"input": len(question)},
                    )
            with self.observability.observe(
                "retrieve-financial-report-chunks",
                as_type="retriever",
                input={
                    "tickers": tickers,
                    "limit": 6,
                    **self.observability.safe_text(question, label="question"),
                },
            ) as retrieval_span:
                matches = self.vector_store.search(embedding, tickers)
                if retrieval_span:
                    retrieval_span.update(
                        output={
                            "match_count": len(matches),
                            "sources": list(dict.fromkeys(match.source for match in matches)),
                            "scores": [match.score for match in matches],
                            "matches": [
                                {
                                    "ticker": match.ticker,
                                    "period": match.period,
                                    "year": match.year,
                                    "source": match.source,
                                    "score": match.score,
                                    "text_length": len(match.text),
                                }
                                for match in matches
                            ],
                        }
                    )
            if not matches:
                return AgentResult(
                    agent="RAG Agent",
                    answer="No relevant indexed financial-report content was found.",
                )
            context = "\n\n".join(f"[{match.source}]\n{match.text}" for match in matches)
            answer = self.bedrock.answer(
                question,
                context,
                (
                    "You are a financial-report research assistant. Answer only from the supplied "
                    "context. Never introduce another company or external report that is not present "
                    "in the context. Distinguish facts from interpretation, cite source labels, and "
                    "never present the response as investment advice."
                ),
            )
            return AgentResult(
                agent="RAG Agent",
                answer=answer,
                sources=list(dict.fromkeys(match.source for match in matches)),
            )
        elif self.reports:
            with self.observability.observe(
                "s3-list-financial-reports",
                as_type="tool",
                input={
                    "bucket": self.reports.settings.reports_bucket,
                    "prefix": self.reports.settings.reports_prefix,
                    "tickers": tickers,
                },
                metadata={"provider": "Amazon S3"},
            ) as s3_list_span:
                reports = self.reports.list_reports(tickers)
                if s3_list_span:
                    s3_list_span.update(
                        output={
                            "report_count": len(reports),
                            "reports": [
                                {
                                    "ticker": report.ticker,
                                    "period": report.period,
                                    "year": report.year,
                                    "key": report.key,
                                }
                                for report in reports
                            ],
                        }
                    )
            for report in reports:
                source = f"s3://{self.reports.settings.reports_bucket}/{report.key}"
                with self.observability.observe(
                    "s3-download-financial-report",
                    as_type="tool",
                    input={
                        "bucket": self.reports.settings.reports_bucket,
                        "key": report.key,
                        "ticker": report.ticker,
                        "period": report.period,
                        "year": report.year,
                    },
                    metadata={"provider": "Amazon S3"},
                ) as s3_download_span:
                    content, content_type = self.reports.download(report.key)
                    text = extract_text(content, content_type)
                    documents.append((source, text))
                    if s3_download_span:
                        s3_download_span.update(
                            output={
                                "source": source,
                                "content_type": content_type,
                                "content_bytes": len(content),
                                "text_length": len(text),
                            }
                        )

        with self.observability.observe(
            "chunk-financial-report-documents",
            as_type="span",
            input={
                "document_count": len(documents),
                "sources": [source for source, _ in documents],
            },
            metadata={"chunk_size": 1400, "chunk_overlap": 200},
        ) as chunk_span:
            chunks = [
                (source, chunk)
                for source, text in documents
                for chunk in chunk_text(text)
            ]
            if chunk_span:
                chunk_span.update(
                    output={
                        "chunk_count": len(chunks),
                        "sources": list(dict.fromkeys(source for source, _ in chunks)),
                    }
                )
        with self.observability.observe(
            "keyword-retrieve-financial-report-chunks",
            as_type="retriever",
            input={
                "chunk_count": len(chunks),
                "limit": 6,
                **self.observability.safe_text(question, label="question"),
            },
        ) as keyword_retrieval_span:
            matches = _retrieve(question, chunks)
        if uploaded_content and _is_summary_request(question):
            matches = chunks[:6]
        if keyword_retrieval_span:
            keyword_retrieval_span.update(
                output={
                    "match_count": len(matches),
                    "sources": list(dict.fromkeys(source for source, _ in matches)),
                    "matches": [
                        {
                            "source": source,
                            "text_length": len(chunk),
                        }
                        for source, chunk in matches
                    ],
                }
            )
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
                    "context. Never introduce another company or external report that is not present "
                    "in the context. Distinguish facts from interpretation, cite source labels, and "
                    "never present the response as investment advice."
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
