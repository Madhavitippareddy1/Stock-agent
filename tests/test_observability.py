from stock_agent.config import Settings
from stock_agent.observability import Observability


def test_observability_is_disabled_without_credentials() -> None:
    observability = Observability(Settings(langfuse_enabled=True))

    assert observability.enabled is False
    with observability.observe("test-span") as span:
        assert span is None


def test_content_is_redacted_by_default() -> None:
    observability = Observability(Settings(langfuse_capture_content=False))

    assert observability.content("private report", {"length": 14}) == {"length": 14}
