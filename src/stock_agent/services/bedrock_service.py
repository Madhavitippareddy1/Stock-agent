import boto3

from stock_agent.config import get_settings


def get_bedrock_client():
    settings = get_settings()
    return boto3.client("bedrock-runtime", region_name=settings.aws_region)
