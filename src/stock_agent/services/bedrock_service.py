import json
from typing import Any

import boto3

from stock_agent.config import Settings, get_settings
from stock_agent.observability import Observability, get_observability


class BedrockService:
    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
        observability: Observability | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.observability = observability or get_observability()
        self.client = client or boto3.client(
            "bedrock-runtime", region_name=self.settings.aws_region
        )

    def _answer_cost_details(self, usage: dict[str, Any]) -> dict[str, float]:
        input_cost = (
            usage.get("inputTokens", 0)
            * self.settings.bedrock_input_cost_per_1m_tokens
            / 1_000_000
        )
        output_cost = (
            usage.get("outputTokens", 0)
            * self.settings.bedrock_output_cost_per_1m_tokens
            / 1_000_000
        )
        return {
            "input": input_cost,
            "output": output_cost,
            "total": input_cost + output_cost,
        }

    def _embedding_cost_details(self, input_tokens: int) -> dict[str, float]:
        total_cost = (
            input_tokens * self.settings.bedrock_embedding_cost_per_1m_tokens / 1_000_000
        )
        return {"input": total_cost, "total": total_cost}

    def answer(self, question: str, context: str, system_prompt: str) -> str:
        generation_input = self.observability.content(
            {
                "system": system_prompt,
                "question": question,
                "context": context,
            },
            {
                "question_length": len(question),
                "context_length": len(context),
            },
        )
        with self.observability.observe(
            "bedrock-nova-answer",
            as_type="generation",
            input=generation_input,
            model=self.settings.bedrock_model_id,
            model_parameters={"maxTokens": 1200, "temperature": 0.1},
            metadata={
                "provider": "Amazon Bedrock",
                "region": self.settings.aws_region,
                "context_length": len(context),
                "question_length": len(question),
                "content_capture_enabled": self.settings.langfuse_capture_content,
            },
        ) as generation:
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
            answer = response["output"]["message"]["content"][0]["text"]
            usage = response.get("usage", {})
            if generation:
                generation.update(
                    output=self.observability.content(
                        answer,
                        {"answer_length": len(answer)},
                    ),
                    usage_details={
                        "input": usage.get("inputTokens", 0),
                        "output": usage.get("outputTokens", 0),
                        "total": usage.get("totalTokens", 0),
                    },
                    cost_details=self._answer_cost_details(usage),
                )
            return answer

    def embed(self, text: str) -> list[float]:
        generation_input = self.observability.content(
            text,
            {"input_length": len(text)},
        )
        with self.observability.observe(
            "bedrock-titan-embedding",
            as_type="generation",
            input=generation_input,
            model=self.settings.bedrock_embedding_model_id,
            model_parameters={
                "dimensions": self.settings.embedding_dimension,
                "normalize": True,
            },
            metadata={
                "provider": "Amazon Bedrock",
                "region": self.settings.aws_region,
                "content_capture_enabled": self.settings.langfuse_capture_content,
            },
        ) as generation:
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
            embedding = payload["embedding"]
            input_tokens = int(payload.get("inputTextTokenCount") or max(len(text) // 4, 1))
            if generation:
                generation.update(
                    output={
                        "embedding_dimension": len(embedding),
                        "content_captured": False,
                    },
                    usage_details={
                        "input": input_tokens,
                        "total": input_tokens,
                    },
                    cost_details=self._embedding_cost_details(input_tokens),
                )
            return embedding
