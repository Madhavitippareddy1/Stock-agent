"""Vector database adapter.

Implement this interface with Amazon OpenSearch Serverless or PostgreSQL pgvector.
"""


def search_similar_chunks(query: str, limit: int = 5) -> list[dict[str, object]]:
    return []
