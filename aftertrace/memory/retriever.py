"""Deterministic shared lexical retrieval for all memory conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

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
) -> RetrievalResult:
    """Rank by overlap then recency and greedily pack whole items within K.

    DECISION: Items are not token-truncated when they do not fit. Keeping item
    boundaries intact lets all representations be parsed losslessly; the item
    is skipped and later smaller items may still fill the remaining budget.
    """

    if budget < 0:
        raise ValueError("retrieval budget K must be non-negative")
    views: list[ItemView] = []
    for candidate in candidates:
        if isinstance(candidate, ItemView):
            view = candidate
            if render is not None:
                view = ItemView(view.item, render(view.item))
        else:
            view = ItemView(candidate, render(candidate) if render else candidate.tokens)
        views.append(view)

    ranked: list[tuple[int, int, ItemView]] = []
    operations = 0
    for view in views:
        key = item_key(view.key_tokens if view.key_tokens is not None else view.tokens)
        operations += len(key)
        score = len(query_features(query_tokens) & key)
        ranked.append((score, view.item.episode_id, view))
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)

    selected: list[ItemView] = []
    used = 0
    for _score, _recency, view in ranked:
        item_length = len(view.tokens)
        if not item_length or used + item_length > budget:
            continue
        selected.append(view)
        used += item_length
    assert used <= budget
    return RetrievalResult(tuple(selected), operations, used)
