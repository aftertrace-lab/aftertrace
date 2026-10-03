"""Implementations of the frozen Stage-0 memory conditions."""

from __future__ import annotations

from collections import Counter
from typing import Any

from aftertrace.memory.base import ItemView, MemoryCondition, MemoryItem
from aftertrace.memory.render import render_level, render_l1, render_l2, render_l3, render_l4, render_residual, render_summaries
from aftertrace.world.generator import Episode, World


class EpisodeStore(MemoryCondition):
    def __init__(self, name: str, world: World, cfg: dict[str, Any], budget: int | None = None):
        super().__init__(name, params=cfg, store_budget_tokens=budget)
        self.world = world
        self.grammar = world.grammar

    def observe(self, episode: Episode, step: int) -> None:
        super().observe(episode, step)
        if self.name == "none":
            return
        if self.name == "summary":
            return
        if self.name == "full_oracle":
            return
        if self.name.startswith("progressive"):
            rate = float(self.params.get("decay_rate", 0.01))
            rule = self.params.get("decay_rule", "time")
            for old in self.items:
                delta = max(0, step - old.created_at)
                if rule == "use_count":
                    decay = rate * old.use_count
                elif rule == "recency_of_use":
                    decay = rate * max(0, step - old.last_access)
                elif rule == "expected_utility":
                    weights = self.params.get("utility_weights", {})
                    weight = sum(float(weights.get(c, 0.5)) for c in old.fact_classes) / max(1, len(old.fact_classes))
                    decay = rate * delta / max(0.1, weight)
                else:
                    decay = rate * delta
                old.fidelity = max(0.0, old.fidelity - decay)
        if self.name.startswith("evict_"):
            tokens, level = render_l3(episode), 3
        elif self.name == "residual_offload":
            tokens, level = render_residual(episode, self.grammar), 4
        else:
            tokens, level = render_l4(episode), 4
        classes = tuple(f.fact_class for f in episode.facts)
        uniqueness = len(set(episode.entities)) / max(1, len(self.world.entities))
        weights = self.params.get("utility_weights", {})
        class_score = sum(float(weights.get(label, 0.5)) for label in classes) / max(1, len(classes))
        self.items.append(MemoryItem(
            key=str(episode.id), episode_id=episode.id, tokens=tokens,
            source_episode=episode, created_at=step, last_access=step,
            fidelity=1.0, level=level, uniqueness=uniqueness,
            fact_classes=classes, fact_class_score=class_score,
        ))
        self._enforce_budget()

    def _enforce_budget(self) -> None:
        if not self.name.startswith("evict_") or self.store_budget_tokens is None:
            return
        while self.items and self.store_tokens > self.store_budget_tokens:
            policy = self.name.removeprefix("evict_")
            if policy == "fifo":
                victim = min(self.items, key=lambda x: x.created_at)
            elif policy == "lru":
                victim = min(self.items, key=lambda x: x.last_access)
            elif policy == "freq":
                victim = min(self.items, key=lambda x: (x.use_count, x.last_access))
            else:
                victim = min(self.items, key=lambda x: (x.fact_class_score, x.last_access))
            self.items.remove(victim)

    def views(self, query: Any, step: int) -> list[ItemView]:
        del step
        if self.name in {"none", "summary", "full_oracle"}:
            return []
        result = []
        candidate_count = sum(
            bool(set(query.tokens) & set(item.tokens)) for item in self.items
        )
        for item in self.items:
            tokens = item.tokens
            if self.name.startswith("progressive"):
                threshold = float(self.params.get("tau", 0.7))
                gated = self.name != "progressive_nogate"
                if item.fidelity < threshold and gated and candidate_count > int(self.params.get("c", 3)):
                    continue
                level = min(4, int(item.fidelity * 5))
                if item.source_episode is not None:
                    tokens = render_level(
                        item.source_episode, level,
                        utility_weights=self.params.get("utility_weights"),
                    )
                else:
                    continue
            result.append(ItemView(item, tokens, item.tokens))
        return result

    @staticmethod
    def render_retrieved(view: ItemView, query: Any) -> ItemView:
        return ItemView(view.item, _query_render(view.item, query), view.key_tokens)

    def after_retrieval(self, selected: list[ItemView], step: int, *, successful: bool) -> None:
        super().after_retrieval(selected, step, successful=successful)
        if successful and self.name.startswith("progressive"):
            for view in selected:
                view.item.fidelity = 1.0


def _query_render(item: MemoryItem, query: Any) -> tuple[str, ...]:
    """Keep only exact fact keys or cue-supporting facts in a retrieved item."""
    episode = item.source_episode
    if episode is None:
        return item.tokens
    q = set(query.tokens)
    cue_targets: set[str] = set()
    cue_interactions = []
    if "CTX" in q and len(query.tokens) >= 6:
        partner, place, time = query.tokens[2], query.tokens[4], query.tokens[5]
        for fact in episode.facts:
            if (
                fact.relation in {"TOLD", "MET"}
                and fact.subject == partner
                and fact.extra == ("AT", place)
                and episode.time == time
            ):
                cue_targets.add(fact.object)
                cue_interactions.append(fact)
    elif "HOP2" in q and len(query.tokens) >= 8:
        partner, place, time = query.tokens[2], query.tokens[5], query.tokens[7]
        for fact in episode.facts:
            if (
                fact.relation in {"TOLD", "MET"}
                and fact.subject == partner
                and fact.extra == ("AT", place)
                and episode.time == time
            ):
                cue_interactions.append(fact)
    kept = []
    for fact in episode.facts:
        exact = (
            query.query_type != "R"
            and fact.subject in q
            and fact.relation in q
        )
        cue = fact in cue_interactions or (
            fact.relation == "MOVED"
            and fact.subject in cue_targets
            and episode.time in q
        )
        if exact or cue:
            kept.append(fact)
    if not kept:
        return ("EP", episode.time, "EOS")
    out = ["EP", episode.time]
    for fact in kept:
        out.extend(fact.render())
        out.append("SEP")
    out.append("EOS")
    return tuple(out)


class SummaryStore(EpisodeStore):
    def __init__(self, world: World, cfg: dict[str, Any]):
        super().__init__("summary", world, cfg)

    def observe(self, episode: Episode, step: int) -> None:
        super().observe(episode, step)
        interval = int(self.params.get("summary_interval", 100))
        if (step + 1) % interval:
            return
        prefix = self.world.episodes[:step + 1]
        snapshots = render_summaries(prefix)
        self.items = []
        for tokens in snapshots:
            entity = tokens[1]
            self.items.append(MemoryItem(
                f"sum-{step}-{entity}", step, tuple(tokens), None,
                source_entity=entity, created_at=step, last_access=step,
            ))
        self.maintenance_ops_last_step = sum(len(item.facts) for item in prefix)


def make_condition(name: str, world: World, config: dict[str, Any], *, store_budget_tokens: int | None = None) -> MemoryCondition:
    """Create a condition from conditions.yaml parameters."""
    if name not in config:
        raise ValueError(f"unknown condition {name!r}")
    params = dict(config[name] or {})
    if name == "summary":
        return SummaryStore(world, params)
    return EpisodeStore(name, world, params, store_budget_tokens)
