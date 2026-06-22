from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AgentName(StrEnum):
    RAG = "rag"
    NEWS = "news"
    STOCK = "stock"


@dataclass
class AgentResult:
    agent: str
    answer: str
    sources: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchResult:
    answer: str
    sections: list[AgentResult] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
