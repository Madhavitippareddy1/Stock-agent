import json
from typing import Any

import boto3

from stock_agent.config import Settings, get_settings


class BedrockService:
    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or boto3.client(
            "bedrock-runtime", region_name=self.settings.aws_region
        )

    def answer(self, question: str, context: str, system_prompt: str) -> str:
        response = self.client.converse(
            modelId=self.settings.bedrock_model_id,
            system=[{"text": system_prompt}],
            messages=[
                {
                    "role": "user",
                    "content": [{"text": f"Context:\n{context}\n\nQuestion:\n{question}"}],
                }
            ],
            inferenceConfig={"maxTokens": 1200, "temperature": 0.1},
        )
        return response["output"]["message"]["content"][0]["text"]

    def embed(self, text: str) -> list[float]:
        response = self.client.invoke_model(
            modelId=self.settings.bedrock_embedding_model_id,
            body=json.dumps(
                {
                    "inputText": text,
                    "dimensions": self.settings.embedding_dimension,
                    "normalize": True,
                }
            ),
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(response["body"].read())
        return payload["embedding"]
