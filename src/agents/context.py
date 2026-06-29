from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.generation.generator import GenerationResult

if TYPE_CHECKING:
    from src.core.container import AppContainer


@dataclass
class RAGRunContext:
    raw_query: str
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    auth_token: str | None = None
    correlation_id: str = ""
    namespace: str = "default"
    processed_query: Any | None = None
    query_routing_intent: str = "retrieval"
    context_items: list[Any] = field(default_factory=list)
    generation_result: GenerationResult | None = None
    agent_trace: list[str] = field(default_factory=list)
    user_preferences: list[str] = field(default_factory=list)
    container: AppContainer | None = field(default=None, repr=False)
    _query_vector: Any | None = field(default=None, repr=False)
    _evaluator: Any | None = field(default=None, repr=False)

    def record(self, agent_name: str, event: str) -> None:
        self.agent_trace.append(f"[{agent_name}] {event}")
