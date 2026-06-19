from dataclasses import dataclass, field


@dataclass
class AgentResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
