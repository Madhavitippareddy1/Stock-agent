# Langfuse observability

The Stock Agent includes optional Langfuse tracing. The application continues to
work when Langfuse is disabled or credentials are missing.

## What is traced

- Supervisor request, selected routes, ticker scope, and uploaded-file presence.
- RAG Agent and Stock Data Agent execution.
- Amazon Titan question embedding and vector dimensions.
- OpenSearch retrieval count, similarity scores, and source identifiers.
- Amazon Nova Lite model name, inference settings, token usage, and failures.
- Yahoo Finance quote-tool availability by ticker.

Financial-report and question content is not sent to Langfuse by default.
`LANGFUSE_CAPTURE_CONTENT=false` stores lengths and operational metadata instead.

## 1. Create a Langfuse project

Use Langfuse Cloud or a self-hosted Langfuse deployment. Create a project and
copy its public key, secret key, and base URL.

## 2. Local configuration

Add the following values to `.env`:

```dotenv
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_ENVIRONMENT=development
LANGFUSE_RELEASE=local
LANGFUSE_CAPTURE_CONTENT=false
LANGFUSE_SAMPLE_RATE=1.0
```

Run the application and submit a research question:

```bash
uv sync
uv run streamlit run app.py
```

The trace appears in the Langfuse project after the SDK flushes its event batch.

## 3. AWS Secrets Manager

Store the two Langfuse keys in one JSON secret:

```bash
aws secretsmanager create-secret \
  --name dev/dsmay/stock-agent/langfuse \
  --secret-string '{"public_key":"pk-lf-...","secret_key":"sk-lf-..."}' \
  --region eu-west-2
```

Add these entries to the ECS container definition after replacing
`ACCOUNT_ID` with the AWS account ID:

```json
"secrets": [
  {
    "name": "LANGFUSE_PUBLIC_KEY",
    "valueFrom": "arn:aws:secretsmanager:eu-west-2:ACCOUNT_ID:secret:dev/dsmay/stock-agent/langfuse:public_key::"
  },
  {
    "name": "LANGFUSE_SECRET_KEY",
    "valueFrom": "arn:aws:secretsmanager:eu-west-2:ACCOUNT_ID:secret:dev/dsmay/stock-agent/langfuse:secret_key::"
  }
]
```

Grant the ECS execution role permission to read that secret:

```json
{
  "Effect": "Allow",
  "Action": "secretsmanager:GetSecretValue",
  "Resource": "arn:aws:secretsmanager:eu-west-2:ACCOUNT_ID:secret:dev/dsmay/stock-agent/langfuse-*"
}
```

Set `LANGFUSE_ENABLED=true` in the task definition and deploy a new task
revision. Do not place the secret key in GitHub, the Docker image, source code,
or the plain `environment` section of the ECS task definition.

## 4. Privacy controls

- Keep `LANGFUSE_CAPTURE_CONTENT=false` for production unless sending financial
  report excerpts to the chosen Langfuse deployment is explicitly approved.
- Lower `LANGFUSE_SAMPLE_RATE` if tracing every request is unnecessary.
- Use separate `LANGFUSE_ENVIRONMENT` values for development and production.
- Rotate Langfuse keys through Secrets Manager and restart the ECS service.

## 5. Trace hierarchy

```text
stock-agent-research
|-- rag-agent
|   |-- embed-rag-question
|   |-- retrieve-financial-report-chunks
|   `-- bedrock-nova-answer
`-- stock-agent
    `-- yahoo-finance-quote
```
