"""Shared causal evaluation, parameter tuning, and condition accounting."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Iterable

from aftertrace.accounting import AccountingRecord, accounting_from_values
from aftertrace.memory.base import ItemView, MemoryItem
from aftertrace.memory.conditions import make_condition
from aftertrace.memory.retriever import retrieve
from aftertrace.reader.oracle import OracleReader
from aftertrace.world.generator import World
from aftertrace.world.seeds import assert_phase
from aftertrace.world import vocab


@dataclass(frozen=True)
class QueryResult:
    query_id: int
    query_type: str
    answer: str
    confidence: float
    target: str
    score: int
    evidence_episode_ids: tuple[int, ...]
    selected_episode_ids: tuple[int, ...]
    from_superseding: bool | None
    accounting: AccountingRecord


@dataclass
class Evaluation:
    condition: str
    K: int
    results: list[QueryResult]
    utility: float
    per_type: dict[str, dict[str, float]]
    mean_accounting: dict[str, float]


def _query_relevant_episode_ids(query: Any) -> tuple[int, ...]:
    return tuple(query.evidence_episode_ids)


def _full_oracle_views(world: World, query: Any) -> list[ItemView]:
    evidence = set(_query_relevant_episode_ids(query))
    return [
        ItemView(
            item=MemoryItem(
                key=str(ep.id), episode_id=ep.id, tokens=ep.tokens,
                source_episode=ep, created_at=ep.id,
            ),
            tokens=ep.tokens,
        )
        for ep in world.episodes if ep.id in evidence
    ]


def evaluate_condition(
    world: World,
    condition_name: str,
    K: int,
    *,
    condition_config: dict[str, Any],
    tuned: dict[str, Any] | None = None,
    store_budget_tokens: int | None = None,
    queries: Iterable[Any] | None = None,
    reader: OracleReader | None = None,
    expected_phase: str = "eval",
) -> Evaluation:
    """Run one paired world/condition/K, observing only causal prefixes."""
    assert_phase(world.seed, expected_phase)
    if world.phase != expected_phase:
        raise AssertionError(f"world phase {world.phase!r} != {expected_phase!r}")
    resolved_config = {key: dict(value or {}) for key, value in condition_config.items()}
    if tuned and condition_name.startswith("progressive"):
        tuned_values = {
            key: value for key, value in tuned.items()
            if key in {"tau", "c", "decay_rule", "decay_rate", "utility_weights"}
        }
        if condition_name == "progressive_uniform":
            tuned_values.pop("utility_weights", None)
            tuned_values["utility_weights"] = {}
        resolved_config.setdefault(condition_name, {}).update(tuned_values)
    if tuned and condition_name == "evict_learned":
        resolved_config.setdefault(condition_name, {}).update({
            "utility_weights": tuned.get("utility_weights", {})
        })
    condition = make_condition(
        condition_name, world, resolved_config,
        store_budget_tokens=store_budget_tokens,
    )
    oracle = reader or OracleReader()
    selected_queries = list(world.queries if queries is None else queries)
    by_step: dict[int, list[Any]] = defaultdict(list)
    for query in selected_queries:
        by_step[query.after_episode].append(query)
    results: list[QueryResult] = []
    sum_fields: dict[str, float] = defaultdict(float)
    for episode in world.episodes:
        condition.observe(episode, episode.id)
        for query in by_step.get(episode.id, ()):
            query_started = perf_counter()
            if condition_name == "full_oracle":
                views = _full_oracle_views(world, query)
                retrieval_ops = sum(len(v.tokens) for v in views)
                selected = views
                context_tokens = sum(len(v.tokens) for v in views)
            else:
                views = condition.views(query, episode.id)
                retrieved = retrieve(views, query.tokens, K)
                selected = list(retrieved.selected)
                if condition_name == "raw_offload_qrender":
                    selected = [condition.render_retrieved(view, query) for view in selected]
                retrieval_ops = retrieved.retrieval_ops
                context_tokens = sum(len(view.tokens) for view in selected)
            prediction = oracle.predict(
                selected, query, world.grammar, float(world.config["p_redundancy"])
            )
            elapsed_ms = (perf_counter() - query_started) * 1000.0
            if condition_name != "full_oracle":
                condition.after_retrieval(
                    selected, episode.id,
                    successful=prediction.answer == query.answer,
                )
            answer = prediction.answer
            score = 1 if answer == query.answer else (0 if answer == "ABSTAIN" else -1)
            record = accounting_from_values(
                context_tokens=context_tokens,
                retrieval_ops=retrieval_ops,
                reconstruction_passes=1 + sum(v.item.reconstruction_passes for v in selected),
                store_tokens=condition.store_tokens,
                store_bits=context_storage_bits(
                    condition.store_tokens, len(condition.items), len(vocab.TOKENS),
                    condition.store_metadata_fields,
                ),
                maintenance_ops_per_step=float(condition.maintenance_ops_last_step),
                policy_params=condition.policy_params,
                wallclock_ms=elapsed_ms,
            )
            results.append(QueryResult(
                query.id, query.query_type, answer, prediction.confidence,
                query.answer, score, query.evidence_episode_ids,
                tuple(v.item.episode_id for v in selected if v.item.episode_id is not None),
                query.from_superseding, record,
            ))
            for key, value in asdict(record).items():
                if isinstance(value, (int, float)):
                    sum_fields[key] += float(value)
            if condition_name != "full_oracle" and context_tokens > K:
                raise AssertionError(f"context {context_tokens} exceeds K={K}")
    if len(results) != len(selected_queries):
        raise AssertionError("some queries were not evaluated within their causal stream")
    per_type: dict[str, dict[str, float]] = {}
    for query_type in sorted({result.query_type for result in results}):
        group = [result for result in results if result.query_type == query_type]
        per_type[query_type] = {
            "n": float(len(group)),
            "utility": sum(r.score for r in group) / max(1, len(group)),
            "accuracy": sum(r.answer == r.target for r in group) / max(1, len(group)),
            "abstain": sum(r.answer == "ABSTAIN" for r in group) / max(1, len(group)),
        }
    count = max(1, len(results))
    return Evaluation(
        condition_name, K, results,
        sum(r.score for r in results) / count,
        per_type,
        {key: value / count for key, value in sum_fields.items()},
    )


def tune_parameters(
    policy_worlds: Iterable[World], condition_config: dict[str, Any],
    tuning_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tune policy-only quantities; eval worlds are rejected before inspection."""
    worlds = list(policy_worlds)
    if not worlds:
        raise ValueError("at least one policy_tune world is required")
    for world in worlds:
        assert_phase(world.seed, "policy_tune")
        if world.phase != "policy_tune":
            raise AssertionError("tuning is restricted to policy_tune worlds")
    tune_cfg = dict(tuning_config or {
        "tau_candidates": (0.5, 0.7, 0.9),
        "decay_rate_candidates": (0.005, 0.01, 0.02, 0.05),
        "c_candidates": (1, 3, 5),
        "decay_rules": ("time", "recency_of_use", "use_count", "expected_utility"),
        "minimum_class_weight": 0.1,
        "gated_reinstatement_credit": 0.5,
    })
    # Tune from policy-only query/evidence trajectories. The proxy rewards
    # retaining evidence items and penalizes storage fidelity spent on items
    # never cited; tie-breaks prefer slower decay.
    class_hits: dict[str, int] = defaultdict(int)
    class_totals: dict[str, int] = defaultdict(int)
    samples = []
    for world in worlds:
        evidence_ids = {q.id: set(q.evidence_episode_ids) for q in world.queries}
        queried = set().union(*evidence_ids.values()) if evidence_ids else set()
        for episode in world.episodes:
            cited = episode.id in queried
            for fact in episode.facts:
                class_totals[fact.fact_class] += 1
                class_hits[fact.fact_class] += int(cited)
        samples.extend((world, query) for query in world.queries)
    utility_weights = {
        cls: max(float(tune_cfg["minimum_class_weight"]), class_hits[cls] / max(1, class_totals[cls]))
        for cls in class_totals
    }
    # DECISION: This small deterministic grid tunes policy hyperparameters
    # against policy-world evidence retention; equal objectives prefer higher τ
    # and lower decay, preserving information conservatively.
    candidates = []
    for rule in tune_cfg["decay_rules"]:
        for rate in tune_cfg["decay_rate_candidates"]:
            for tau in tune_cfg["tau_candidates"]:
                for c in tune_cfg["c_candidates"]:
                    score = 0.0
                    count = 0
                    for world, query in samples:
                        age = max(0, query.after_episode - min(query.evidence_episode_ids, default=query.after_episode))
                        if rule == "use_count":
                            decay = rate * 1.0
                        elif rule == "expected_utility":
                            decay = rate * age / max(0.1, sum(utility_weights.values()) / max(1, len(utility_weights)))
                        else:
                            decay = rate * age
                        survives = 1.0 - decay >= tau
                        if survives:
                            score += 1.0
                        elif (
                            rule != "recency_of_use"
                            and query.query_type == "R"
                            and query.competitor_count is not None
                            and query.competitor_count <= c
                        ):
                            score += float(tune_cfg["gated_reinstatement_credit"])
                        count += 1
                    candidates.append((score / max(1, count), -rate, tau, rule, rate, c))
    _score, _neg_rate, tau, rule, rate, c = max(candidates)
    return {
        "tau": tau,
        "c": c,
        "decay_rule": rule,
        "decay_rate": rate,
        "utility_weights": utility_weights,
        "tuned_world_seeds": [world.seed for world in worlds],
    }


def context_storage_bits(tokens: int, items: int, vocab_size: int, metadata_fields: int) -> float:
    from aftertrace.accounting import storage_bits
    return storage_bits(tokens, items, vocab_size=vocab_size, metadata_fields=metadata_fields)
