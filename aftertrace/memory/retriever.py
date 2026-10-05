"""Deterministic shared lexical retrieval for all memory conditions."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Callable, Sequence

from aftertrace.memory.base import ItemView, MemoryItem


_NON_CONTENT_QUERY_TOKENS = {
    "Q", "A", "CURRENT", "PREV", "HOP2", "CTX", "TIME", "FROM",
}


@dataclass(frozen=True)
class RetrievalResult:
    selected: tuple[ItemView, ...]
    retrieval_ops: int
    context_tokens: int


def query_features(query_tokens: Sequence[str]) -> frozenset[str]:
    return frozenset(
        token for token in query_tokens if token not in _NON_CONTENT_QUERY_TOKENS
    )


def item_key(tokens: Sequence[str]) -> frozenset[str]:
    return frozenset(token for token in tokens if not token.startswith("F_"))


def lexical_score(query_tokens: Sequence[str], item_tokens: Sequence[str]) -> int:
    return len(query_features(query_tokens) & item_key(item_tokens))


def retrieve(
    candidates: Sequence[ItemView] | Sequence[MemoryItem],
    query_tokens: Sequence[str],
    budget: int,
    *,
    render: Callable[[MemoryItem], tuple[str, ...]] | None = None,
    max_items: int | None = None,
) -> RetrievalResult:
    """Rank by overlap then recency and greedily pack whole items within K.

    DECISION: Items are not token-truncated when they do not fit. Keeping item
    boundaries intact lets all representations be parsed losslessly; the item
    is skipped and later smaller items may still fill the remaining budget.
    """

    return SharedRetriever().retrieve(
        candidates, query_tokens, budget, render=render, max_items=max_items
    )


class SharedRetriever:
    """Incremental inverted-index retriever shared across a stream evaluation."""

    def __init__(self) -> None:
        self._keys: dict[int, frozenset[str]] = {}
        self._postings: dict[str, set[int]] = defaultdict(set)

    def retrieve(
        self,
        candidates: Sequence[ItemView] | Sequence[MemoryItem],
        query_tokens: Sequence[str],
        budget: int,
        *,
        render: Callable[[MemoryItem], tuple[str, ...]] | None = None,
        max_items: int | None = None,
    ) -> RetrievalResult:
        if budget < 0:
            raise ValueError("retrieval budget K must be non-negative")
        if max_items is not None and max_items < 1:
            raise ValueError("max_items must be positive when specified")
        views: list[ItemView] = []
        for candidate in candidates:
            if isinstance(candidate, ItemView):
                view = candidate
                if render is not None:
                    view = ItemView(view.item, render(view.item))
            else:
                view = ItemView(candidate, render(candidate) if render else candidate.tokens)
            views.append(view)

        active = {id(view.item): view for view in views}
        stale = self._keys.keys() - active.keys()
        for item_id in sorted(stale):
            for token in self._keys.pop(item_id):
                self._postings[token].discard(item_id)
        operations = 0
        for item_id, view in active.items():
            if isinstance(view.key_tokens, frozenset):
                key = view.key_tokens
            elif view.key_tokens is None:
                key = view.item.retrieval_key or item_key(view.tokens)
            else:
                key = item_key(view.key_tokens)
            operations += len(key)
            if item_id not in self._keys:
                self._keys[item_id] = key
                for token in sorted(key):
                    self._postings[token].add(item_id)

        scores: dict[int, int] = defaultdict(int)
        for token in sorted(query_features(query_tokens)):
            for item_id in sorted(self._postings.get(token, ())):
                if item_id in active:
                    scores[item_id] += 1
        ranked = sorted(views, key=lambda view: _retrieval_tie_key(view, scores.get(id(view.item), 0)))
        selected: list[ItemView] = []
        used = 0
        for view in ranked:
            item_length = len(view.tokens)
            if item_length and used + item_length <= budget:
                selected.append(view)
                used += item_length
                if max_items is not None and len(selected) >= max_items:
                    break
            if used == budget:
                break
        assert used <= budget
        return RetrievalResult(tuple(selected), operations, used)


def _retrieval_tie_key(view: ItemView, score: int) -> tuple[object, ...]:
    item = view.item
    entity = item.source_entity or next(
        iter(sorted(
            (token for token in view.tokens if token.startswith("E_")),
            key=lambda token: int(token[2:]),
        )),
        "",
    )
    try:
        entity_order: tuple[int, object] = (0, int(entity[2:])) if entity.startswith("E_") else (1, entity)
    except ValueError:
        entity_order = (1, entity)
    recency = item.episode_id if item.episode_id is not None else -1
    return (-score, -recency, str(item.key), entity_order, tuple(view.tokens))
