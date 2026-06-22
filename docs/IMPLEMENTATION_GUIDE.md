# NASDAQ-10 Stock AI Agent — Implementation Guide

## 1. Scope and architecture

The application supports a configurable universe of ten large NASDAQ-listed
stocks: NVDA, GOOGL, AAPL, MSFT, AMZN, META, AVGO, TSLA, COST, and NFLX.

The request path is:

1. The user chooses companies, enters text, and optionally uploads a PDF or
   text report in Streamlit.
2. The Supervisor Agent detects whether the request needs report retrieval,
   news, stock data, or a combination.
3. The RAG Agent retrieves relevant passages from the upload and S3 reports.
4. The News Agent calls NewsAPI.
5. The Stock Data Agent calls Yahoo Finance through `yfinance`.
6. Results and source links are returned to Streamlit.

## 2. Project structure

```text
app.py                              Streamlit UI
src/stock_agent/
  agents/                           Supervisor specialist agents
  services/                         Bedrock, S3, document and DB adapters
  tools/                            Yahoo, NewsAPI and SEC EDGAR tools
  config.py                         Environment configuration
infra/ecs-task-definition.json      ECS Fargate task template
scripts/sync_sec_reports.py         SEC-to-S3 report synchronization
tests/                              Unit tests for each component
.github/workflows/ci-cd.yml         Test, Docker, ECR and ECS pipeline
Dockerfile                          Production Streamlit image
```

## 3. Phase 1 — Local configuration

Install Python 3.12 and `uv`, copy `.env.example` to `.env`, then configure:

- `NEWS_API_KEY`: key from NewsAPI.
- `AWS_REGION`: region containing Bedrock, S3, ECR and ECS.
- `REPORTS_BUCKET`: private S3 bucket for filings.
- `SEC_USER_AGENT`: application name and monitored email address.
- Bedrock model IDs available in the selected region.

Run:

```bash
uv sync --dev
uv run streamlit run app.py
```

## 4. Phase 2 — Financial reports in S3

Create a private, encrypted S3 bucket with block-public-access enabled. Grant
the ECS task role `s3:ListBucket` and `s3:GetObject`; grant the synchronization
identity `s3:PutObject`.

Synchronize recent official 10-K and 10-Q filings:

```bash
uv run python scripts/sync_sec_reports.py
```

If SEC endpoints are unavailable from the execution network, generate
structured annual, quarterly, and half-yearly financial reports from Yahoo:

```bash
uv run python scripts/sync_yahoo_reports.py
```

Objects use this structure:

```text
financial-reports/{ticker}/{annual|half-yearly|quarterly}/{year}/{filename}
```

U.S. companies normally publish 10-Q quarterly filings rather than a distinct
half-year report. The second-quarter 10-Q is classified as `half-yearly`.

Schedule this script with EventBridge plus an ECS scheduled task or Lambda.
Respect SEC fair-access requirements and use a real contact email.

## 5. Phase 3 — RAG and Bedrock

The RAG Agent:

1. Reads PDF/text uploads and S3 filing objects.
2. Converts PDF or HTML into normalized text.
3. Splits text into overlapping chunks.
4. Scores chunks against the question.
5. Sends top passages to Amazon Bedrock Converse with a grounded prompt.

For larger production collections, replace lexical retrieval with OpenSearch
Serverless or Aurora PostgreSQL/pgvector and persist Bedrock embeddings during
the SEC synchronization phase.

## 6. Phase 4 — Data APIs

Yahoo Finance is accessed through the community `yfinance` library. It supplies
quotes, history, annual statements and quarterly statements. This is suitable
for prototypes; confirm licensing and use a contracted market-data vendor for
commercial or latency-sensitive production use.

NewsAPI uses `/v2/everything`, a 14-day lookback, English results and newest
articles first. Store the key in AWS Secrets Manager, never in Git.

When `NEWS_API_KEY` is not configured, the News Agent automatically falls back
to Yahoo Finance company news so research requests remain functional.

## 7. Phase 5 — Tests

Run:

```bash
uv run ruff check .
uv run pytest --cov=stock_agent --cov-report=term-missing
docker build -t stock-agent:test .
```

Tests mock network and AWS boundaries. They cover supervisor routing, document
extraction/chunking, RAG retrieval, News Agent formatting, Yahoo ticker
validation, S3 keys, SEC filing classification, and Stock Agent output.

## 8. Phase 6 — Docker, ECR and ECS

1. Create ECR repository `stock-agent`.
2. Create ECS cluster and Fargate service behind an Application Load Balancer.
3. Create `/ecs/stock-agent` CloudWatch log group.
4. Store NewsAPI key in Secrets Manager.
5. Create ECS execution and task IAM roles.
6. Register a task definition based on `infra/ecs-task-definition.json`.
7. Set health-check path to `/_stcore/health` on port 8501.

The image runs as a non-root user and exposes port 8501.

## 9. Phase 7 — GitHub Actions deployment

Configure GitHub environment `production`:

Secret:

- `AWS_ROLE_ARN`

Variables:

- `AWS_DEPLOY_ENABLED=true`
- `AWS_REGION`
- `ECR_REPOSITORY`
- `ECS_CLUSTER`
- `ECS_SERVICE`
- `ECS_TASK_DEFINITION` (path to a checked-in deployable task definition)

The workflow authenticates through GitHub OIDC, tests the code, builds the
Docker image, pushes the commit-tagged image to ECR, renders the task
definition, and updates ECS while waiting for service stability.

## 10. Security and operational checklist

- Keep S3 private, encrypted and versioned.
- Give ECS and GitHub roles least privilege.
- Place ECS tasks in private subnets with controlled outbound access.
- Protect the public UI with an ALB authentication layer or application login.
- Add CloudWatch alarms for ECS health, errors and response latency.
- Cache vendor responses and enforce timeouts/rate limits.
- Display an informational-use disclaimer.
- Do not log uploaded report content, API keys, or personal data.
