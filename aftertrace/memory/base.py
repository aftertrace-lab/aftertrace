"""Shared state and item types for memory conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aftertrace.world.generator import Episode


@dataclass
class MemoryItem:
    """One persistent memory item and its source/provenance metadata."""

    key: str
    episode_id: int
    tokens: tuple[str, ...]
    source_episode: Episode | None
    source_entity: str | None = None
    created_at: int = 0
    last_access: int = 0
    use_count: int = 0
    fidelity: float = 1.0
    level: int = 4
    uniqueness: float = 0.0
    fact_classes: tuple[str, ...] = ()
    fact_class_score: float = 0.0
    reconstruction_passes: int = 0

    @property
    def token_count(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True)
class ItemView:
    """Query-specific rendering of an item returned by a condition."""

    item: MemoryItem
    tokens: tuple[str, ...]
    key_tokens: tuple[str, ...] | None = None


class MemoryCondition:
    """Base condition API: observe the stream, expose candidates, update use."""

    def __init__(
        self,
        name: str,
        *,
        params: dict[str, Any] | None = None,
        store_budget_tokens: int | None = None,
    ) -> None:
        self.name = name
        self.params = dict(params or {})
        self.store_budget_tokens = store_budget_tokens
        self.items: list[MemoryItem] = []
        self.maintenance_ops_last_step = 0
        self.reconstruction_passes_last_query = 0

    def observe(self, episode: Episode, step: int) -> None:
        del episode, step
        self.maintenance_ops_last_step = 0

    def views(self, query: Any, step: int) -> list[ItemView]:
        del query, step
        return [ItemView(item, item.tokens) for item in self.items]

    def after_retrieval(
        self,
        selected: list[ItemView],
        step: int,
        *,
        successful: bool,
    ) -> None:
        del successful
        for view in selected:
            view.item.last_access = step
            view.item.use_count += 1

    @property
    def store_tokens(self) -> int:
        return sum(item.token_count for item in self.items)

    @property
    def policy_params(self) -> int:
        return len(self.params)

    @property
    def store_metadata_fields(self) -> int:
        # DECISION: Charge eight fixed 32-bit fields per stored item: episode
        # id, creation step, last access, use count, fidelity/level, uniqueness,
        # fact-class encoding, and reconstruction-pass count.
        return 8
