import json
from contextlib import contextmanager

from stock_agent.config import Settings
from stock_agent.services.bedrock_service import BedrockService


class FakeBody:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


class FakeBedrockClient:
    def converse(self, **kwargs):
        assert kwargs["modelId"] == "amazon.nova-lite-v1:0"
        return {
            "output": {"message": {"content": [{"text": "Revenue improved."}]}},
            "usage": {"inputTokens": 1000, "outputTokens": 250, "totalTokens": 1250},
        }

    def invoke_model(self, **kwargs):
        assert kwargs["modelId"] == "amazon.titan-embed-text-v2:0"
        return {
            "body": FakeBody(
                {
                    "embedding": [0.1] * 1024,
                    "inputTextTokenCount": 500,
                }
            )
        }


class FakeGeneration:
    def __init__(self) -> None:
        self.updates = []

    def update(self, **kwargs) -> None:
        self.updates.append(kwargs)


class FakeObservability:
    def __init__(self) -> None:
        self.generations = []

    def content(self, value, fallback):
        return fallback

    @contextmanager
    def observe(self, *args, **kwargs):
        generation = FakeGeneration()
        self.generations.append({"args": args, "kwargs": kwargs, "generation": generation})
        yield generation


def test_bedrock_answer_sends_usage_and_cost_details_to_langfuse() -> None:
    observability = FakeObservability()
    service = BedrockService(
        settings=Settings(),
        client=FakeBedrockClient(),
        observability=observability,
    )

    answer = service.answer("What changed?", "Revenue context", "Be concise")

    assert answer == "Revenue improved."
    update = observability.generations[0]["generation"].updates[0]
    assert update["usage_details"] == {"input": 1000, "output": 250, "total": 1250}
    assert update["cost_details"] == {
        "input": 0.00006,
        "output": 0.00006,
        "total": 0.00012,
    }


def test_bedrock_embedding_sends_usage_and_cost_details_to_langfuse() -> None:
    observability = FakeObservability()
    service = BedrockService(
        settings=Settings(),
        client=FakeBedrockClient(),
        observability=observability,
    )

    embedding = service.embed("financial report text")

    assert len(embedding) == 1024
    update = observability.generations[0]["generation"].updates[0]
    assert update["usage_details"] == {"input": 500, "total": 500}
    assert update["cost_details"] == {"input": 0.00001, "total": 0.00001}
