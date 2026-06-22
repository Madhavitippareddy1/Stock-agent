from stock_agent.config import Settings
from stock_agent.services.vector_store import OpenSearchVectorStore


class FakeIndices:
    def __init__(self) -> None:
        self.created = None

    def exists(self, index):
        return False

    def create(self, index, body):
        self.created = (index, body)


class FakeClient:
    def __init__(self) -> None:
        self.indices = FakeIndices()
        self.search_body = None

    def search(self, index, body):
        self.search_body = body
        return {
            "hits": {
                "hits": [
                    {
                        "_score": 0.91,
                        "_source": {
                            "text": "Apple revenue increased.",
                            "source": "s3://bucket/AAPL.json",
                            "ticker": "AAPL",
                            "period": "annual",
                            "year": "2025",
                        },
                    }
                ]
            }
        }


def test_vector_index_uses_titan_dimension_and_faiss() -> None:
    client = FakeClient()
    store = OpenSearchVectorStore(Settings(), client=client)
    store.ensure_index()
    mapping = client.indices.created[1]
    vector = mapping["mappings"]["properties"]["embedding"]
    assert vector["dimension"] == 1024
    assert vector["method"]["engine"] == "faiss"


def test_vector_search_filters_by_ticker() -> None:
    client = FakeClient()
    store = OpenSearchVectorStore(Settings(), client=client)
    matches = store.search([0.1] * 1024, ("AAPL",))
    query = client.search_body["query"]["knn"]["embedding"]
    assert query["filter"] == {"terms": {"ticker": ["AAPL"]}}
    assert matches[0].ticker == "AAPL"
