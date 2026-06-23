import atexit
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator

from langfuse import Langfuse

from stock_agent.config import Settings, get_settings


class Observability:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client: Langfuse | None = None
        if (
            self.settings.langfuse_enabled
            and self.settings.langfuse_public_key
            and self.settings.langfuse_secret_key
        ):
            self.client = Langfuse(
                public_key=self.settings.langfuse_public_key,
                secret_key=self.settings.langfuse_secret_key,
                base_url=self.settings.langfuse_base_url,
                environment=self.settings.langfuse_environment,
                release=self.settings.langfuse_release or None,
                sample_rate=self.settings.langfuse_sample_rate,
            )
            atexit.register(self.client.shutdown)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def content(self, value: Any, fallback: Any) -> Any:
        return value if self.settings.langfuse_capture_content else fallback

    @contextmanager
    def observe(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any = None,
        metadata: Any = None,
        model: str | None = None,
        model_parameters: dict[str, Any] | None = None,
    ) -> Iterator[Any | None]:
        if not self.client:
            yield None
            return
        with self.client.start_as_current_observation(
            name=name,
            as_type=as_type,
            input=input,
            metadata=metadata,
            model=model,
            model_parameters=model_parameters,
        ) as observation:
            try:
                yield observation
            except Exception as exc:
                observation.update(level="ERROR", status_message=str(exc))
                raise

    def flush(self) -> None:
        if self.client:
            self.client.flush()


@lru_cache
def get_observability() -> Observability:
    return Observability()
