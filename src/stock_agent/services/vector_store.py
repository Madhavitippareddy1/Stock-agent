from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urlparse

import boto3
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection, helpers

from stock_agent.config import Settings, get_settings


@dataclass(frozen=True)
class VectorMatch:
    text: str
    source: str
    ticker: str
    period: str
    year: str
    score: float


class OpenSearchVectorStore:
    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or self._build_client()

    def _build_client(self) -> OpenSearch:
        if not self.settings.opensearch_endpoint:
            raise ValueError("OPENSEARCH_ENDPOINT is not configured")
        endpoint = self.settings.opensearch_endpoint
        parsed = urlparse(endpoint if "://" in endpoint else f"https://{endpoint}")
        credentials = boto3.Session().get_credentials()
        auth = AWSV4SignerAuth(credentials, self.settings.aws_region, "aoss")
        return OpenSearch(
            hosts=[{"host": parsed.hostname, "port": parsed.port or 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            pool_maxsize=20,
        )

    def ensure_index(self, recreate: bool = False) -> None:
        index = self.settings.opensearch_index
        if recreate and self.client.indices.exists(index=index):
            self.client.indices.delete(index=index)
        if self.client.indices.exists(index=index):
            return
        self.client.indices.create(
            index=index,
            body={
                "settings": {"index.knn": True},
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": self.settings.embedding_dimension,
                            "method": {
                                "engine": "faiss",
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                            },
                        },
                        "text": {"type": "text"},
                        "source": {"type": "keyword"},
                        "ticker": {"type": "keyword"},
                        "period": {"type": "keyword"},
                        "year": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                    }
                },
            },
        )

    def index_documents(self, documents: Iterable[dict[str, Any]]) -> tuple[int, list[Any]]:
        actions = (
            {
                "_op_type": "index",
                "_index": self.settings.opensearch_index,
                "_source": document,
            }
            for document in documents
        )
        return helpers.bulk(self.client, actions, raise_on_error=False)

    def search(
        self,
        embedding: list[float],
        tickers: tuple[str, ...],
        limit: int = 6,
    ) -> list[VectorMatch]:
        vector_query: dict[str, Any] = {"vector": embedding, "k": limit}
        if tickers:
            vector_query["filter"] = {"terms": {"ticker": list(tickers)}}
        response = self.client.search(
            index=self.settings.opensearch_index,
            body={
                "size": limit,
                "_source": ["text", "source", "ticker", "period", "year"],
                "query": {"knn": {"embedding": vector_query}},
            },
        )
        return [
            VectorMatch(
                text=hit["_source"]["text"],
                source=hit["_source"]["source"],
                ticker=hit["_source"]["ticker"],
                period=hit["_source"]["period"],
                year=hit["_source"]["year"],
                score=float(hit["_score"]),
            )
            for hit in response["hits"]["hits"]
        ]
