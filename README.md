# NASDAQ-10 Stock AI Agent

A production-oriented multi-agent stock research application built with
Streamlit, Yahoo Finance, SEC EDGAR, Amazon S3, Amazon Bedrock,
Amazon OpenSearch Serverless,
Docker, Amazon ECR, and Amazon ECS Fargate.

## Main flow

`End user → Streamlit → Supervisor → RAG / Stock agents → Output`

Specialist agents:

- RAG Agent searches uploaded PDF/text files and uses Titan embeddings plus
  ticker-filtered OpenSearch Serverless vector search for financial reports in S3.
- Stock Data Agent queries Yahoo Finance for prices and financial statements.
- Supervisor Agent selects the appropriate tools and combines responses.

## Local setup

```bash
copy .env.example .env
uv sync
uv run streamlit run app.py
```

## Docker

```bash
docker build -t stock-agent .
docker run --env-file .env -p 8501:8501 stock-agent
```

## Deployment

GitHub Actions tests and builds the Docker image, pushes it to Amazon ECR,
then deploys the image to Amazon ECS.

Current development AWS resources:

- ECS cluster: `dev-dsmay-stock-agent-cluster`
- ECS service: `dev-dsmay-stock-agent-ecs`
- ECR repository: `dev-dsmay-stock-agent-repo`
- S3 reports bucket: `dev-dsmay-stock-agent-bucket`

The test job runs on every push and pull request. AWS deployment is disabled
until the following GitHub Actions configuration is added:

- Repository variable `AWS_DEPLOY_ENABLED=true`
- Repository or `production` environment secret `AWS_ROLE_ARN`

The AWS region and renamed ECR/ECS resources are defined in the workflow
`env` section. The IAM role must trust GitHub's OIDC provider and allow ECR
push, ECS deployment, and `iam:PassRole`. Keep `AWS_DEPLOY_ENABLED` unset or
`false` until the AWS infrastructure is ready.

See [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) for the full
phase-by-phase implementation and deployment guide.

## Langfuse observability

Optional Langfuse tracing is available for supervisor routing, RAG retrieval,
Amazon Bedrock model calls, token usage, Yahoo Finance tools, latency, and
errors. It is disabled by default and does not capture report content unless
explicitly enabled.

See [docs/LANGFUSE_SETUP.md](docs/LANGFUSE_SETUP.md) for local configuration,
AWS Secrets Manager, ECS, and privacy instructions.
