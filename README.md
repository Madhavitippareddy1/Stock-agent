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

The test job runs on every push and pull request. AWS deployment is disabled
until the following GitHub Actions configuration is added:

- Repository variable `AWS_DEPLOY_ENABLED=true`
- Repository variable `AWS_REGION`
- Repository variable `ECR_REPOSITORY`
- Repository variable `ECS_CLUSTER`
- Repository variable `ECS_SERVICE`
- Repository variable `ECS_TASK_DEFINITION`
- Repository or `production` environment secret `AWS_ROLE_ARN`

The IAM role must trust GitHub's OIDC provider and allow ECR push, ECS
deployment, and `iam:PassRole`. Keep `AWS_DEPLOY_ENABLED` unset or `false`
until the AWS infrastructure is ready.

See [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) for the full
phase-by-phase implementation and deployment guide.
