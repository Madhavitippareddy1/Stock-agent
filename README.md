# Stock AI Agent

A clean multi-agent stock research application built with Streamlit, Docker,
Amazon Bedrock, RAG, PostgreSQL, and AWS deployment services.

## Main flow

`End user → Streamlit → Super Agent → Specialist Agents → Bedrock → Output`

Specialist agents:

- RAG Agent: searches uploaded PDF and text content.
- News Agent: collects relevant financial news.
- Portfolio Agent: analyzes holdings, allocation, and risk.
- Sentiment Agent: classifies news and market sentiment.

## Local setup

```bash
cp .env.example .env
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
then deploys the image to Amazon EC2.
