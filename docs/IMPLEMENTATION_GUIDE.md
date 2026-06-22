# NASDAQ-10 Stock AI Agent — Implementation Guide

## 1. Scope and architecture

The application supports a configurable universe of ten large NASDAQ-listed
stocks: NVDA, GOOGL, AAPL, MSFT, AMZN, META, AVGO, TSLA, COST, and NFLX.

The request path is:

1. The user chooses companies, enters text, and optionally uploads a PDF or
   text report in Streamlit.
2. The Supervisor Agent detects whether the request needs report retrieval,
   stock data, or a combination.
3. The RAG Agent retrieves relevant passages from the upload or the
   OpenSearch Serverless vector index backed by S3 reports.
4. The Stock Data Agent calls Yahoo Finance through `yfinance`.
5. Results and source links are returned to Streamlit.

## 2. Project structure

```text
app.py                              Streamlit UI
src/stock_agent/
  agents/                           Supervisor specialist agents
  services/                         Bedrock, S3, OpenSearch and document adapters
  tools/                            Yahoo stock-data and SEC EDGAR tools
  config.py                         Environment configuration
infra/ecs-task-definition.json      ECS Fargate task template
scripts/sync_sec_reports.py         SEC-to-S3 report synchronization
tests/                              Unit tests for each component
.github/workflows/ci-cd.yml         Test, Docker, ECR and ECS pipeline
Dockerfile                          Production Streamlit image
```

## 3. Phase 1 — Local configuration

Install Python 3.12 and `uv`, copy `.env.example` to `.env`, then configure:

- `AWS_REGION`: region containing Bedrock, S3, ECR and ECS.
- `REPORTS_BUCKET`: private S3 bucket for filings.
- `OPENSEARCH_ENDPOINT`: OpenSearch Serverless collection endpoint.
- `OPENSEARCH_INDEX`: vector index name, normally `financial-report-chunks`.
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

1. Reads PDF/text uploads directly; persisted reports remain in S3.
2. Converts PDF or HTML into normalized text.
3. Splits reports into 1,400-character chunks with 200-character overlap.
4. Creates normalized 1,024-dimension embeddings with Amazon Titan Text
   Embeddings V2.
5. Stores vectors and ticker/period/year/source metadata in an Amazon
   OpenSearch Serverless vector-search collection.
6. Embeds each question and runs a ticker-filtered k-NN search.
7. Sends the top six passages to Amazon Nova Lite through Bedrock Converse
   with a grounded prompt.

Uploaded documents are isolated from the S3 corpus. When a user uploads a
report, the RAG Agent analyzes only that file; S3 reports are searched only
when no upload is present. This prevents unrelated company data from being
mixed into document summaries.

Create or refresh the vector index after S3 synchronization:

```bash
uv run python scripts/index_reports_opensearch.py --recreate
```

The OpenSearch data-access policy authorizes the indexing identity and ECS
task role. The task role also needs `aoss:APIAccessAll` for the collection ARN.
The current network policy permits the public collection endpoint because ECS
uses public networking. For production, use private ECS subnets and an
OpenSearch Serverless VPC endpoint.

## 6. Phase 4 — Stock data API

Yahoo Finance is accessed through the community `yfinance` library. It supplies
quotes, history, annual statements and quarterly statements. This is suitable
for prototypes; confirm licensing and use a contracted market-data vendor for
commercial or latency-sensitive production use.

## 7. Phase 5 — Tests

Run:

```bash
uv run ruff check .
uv run pytest --cov=stock_agent --cov-report=term-missing
docker build -t stock-agent:test .
```

Tests mock network and AWS boundaries. They cover supervisor routing, document
extraction/chunking, RAG retrieval, Yahoo ticker validation, S3 keys, SEC
filing classification, and Stock Agent output.

## 8. Phase 6 — Docker, ECR and ECS

1. Create ECR repository `stock-agent`.
2. Create ECS cluster and Fargate service behind an Application Load Balancer.
3. Create `/ecs/stock-agent` CloudWatch log group.
4. Create ECS execution and task IAM roles.
5. Register a task definition based on `infra/ecs-task-definition.json`.
6. Set health-check path to `/_stcore/health` on port 8501.

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
