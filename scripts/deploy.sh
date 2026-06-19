#!/usr/bin/env bash
set -euo pipefail

aws ecr get-login-password --region "$AWS_REGION" |
  docker login --username AWS --password-stdin "$ECR_REGISTRY"
docker pull "$ECR_REGISTRY/$ECR_REPOSITORY:latest"
docker stop stock-agent || true
docker rm stock-agent || true
docker run -d --restart unless-stopped \
  --name stock-agent \
  --env-file /opt/stock-agent/.env \
  -p 8501:8501 \
  "$ECR_REGISTRY/$ECR_REPOSITORY:latest"
