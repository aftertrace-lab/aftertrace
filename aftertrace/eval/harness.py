"""Shared causal evaluation, parameter tuning, and condition accounting."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Iterable

from aftertrace.accounting import AccountingRecord, accounting_from_values
from aftertrace.memory.base import ItemView, MemoryItem
from aftertrace.memory.conditions import make_condition
from aftertrace.memory.retriever import SharedRetriever
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
    retriever = SharedRetriever()
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
                oracle_budget = sum(len(view.tokens) for view in views)
                selected, retrieval_ops = _retrieve_two_hop(
                    retriever, views, query.tokens, oracle_budget
                )
                context_tokens = sum(len(view.tokens) for view in selected)
            else:
                views = condition.views(query, episode.id)
                selected, retrieval_ops = _retrieve_two_hop(
                    retriever, views, query.tokens, K
                )
                if condition_name == "raw_offload_qrender":
                    selected = [condition.render_retrieved(view, query) for view in selected]
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


def _retrieve_two_hop(retriever: SharedRetriever, views, query_tokens, budget):
    """Expand the query once with entities found in its first retrieved view."""

    first = retriever.retrieve(views, query_tokens, budget)
    entities = {
        token
        for view in first.selected
        for token in view.tokens
        if token.startswith("E_")
    }
    expanded = tuple(query_tokens) + tuple(sorted(entities - set(query_tokens), key=lambda token: int(token[2:])))
    second = retriever.retrieve(views, expanded, budget)
    if second.context_tokens > budget:
        raise AssertionError(f"two-hop context {second.context_tokens} exceeds K={budget}")
    return list(second.selected), first.retrieval_ops + second.retrieval_ops


def tune_parameters(
    policy_worlds: Iterable[World], condition_config: dict[str, Any],
    tuning_config: dict[str, Any] | None = None,
    budgets: Iterable[int] = (384, 256, 128, 64, 32, 16),
) -> dict[str, Any]:
    """Maximize symbolic-oracle utility on policy worlds separately for each K."""
    worlds = list(policy_worlds)
    if not worlds:
        raise ValueError("at least one policy_tune world is required")
    for world in worlds:
        assert_phase(world.seed, "policy_tune")
        if world.phase != "policy_tune":
            raise AssertionError("tuning is restricted to policy_tune worlds")
    tune_cfg = dict(tuning_config or {})
    required = {
        "tau_candidates", "decay_rate_candidates", "c_candidates", "decay_rules",
        "queries_per_type", "utility_weight_profiles",
    }
    if required - tune_cfg.keys():
        raise ValueError(f"tuning config missing {sorted(required - tune_cfg.keys())}")
    budget_values = tuple(int(k) for k in budgets)
    by_k: dict[str, dict[str, Any]] = {}
    for K in budget_values:
        policy_samples = [(world, _policy_query_sample(world, int(tune_cfg["queries_per_type"]))) for world in worlds]
        candidates = []
        for rule in tune_cfg["decay_rules"]:
            for rate in tune_cfg["decay_rate_candidates"]:
                for tau in tune_cfg["tau_candidates"]:
                    for c in tune_cfg["c_candidates"]:
                        for weights in tune_cfg["utility_weight_profiles"]:
                            params = {
                                "tau": float(tau), "c": int(c), "decay_rule": str(rule),
                                "decay_rate": float(rate), "utility_weights": dict(weights),
                            }
                            utilities = []
                            stores = []
                            for world, queries in policy_samples:
                                result = evaluate_condition(
                                    world, "progressive", K,
                                    condition_config=condition_config, tuned=params,
                                    queries=queries, expected_phase="policy_tune",
                                )
                                utilities.append(result.utility)
                                stores.append(result.mean_accounting["store_tokens"])
                            candidates.append((
                                sum(utilities) / len(utilities), params,
                                sum(stores) / len(stores),
                            ))
        # DECISION: maximize measured oracle utility; exact ties prefer higher τ
        # and lower decay, then lower c as a deterministic conservative tie-break.
        best_utility = max(row[0] for row in candidates)
        tied = [row for row in candidates if abs(row[0] - best_utility) <= 1e-12]
        best_utility, progressive_params, matched_budget = max(
            tied,
            key=lambda row: (
                row[1]["tau"], -row[1]["decay_rate"], -row[1]["c"],
                tuple(sorted(row[1]["utility_weights"].items())),
            ),
        )
        # Tune learned eviction's fact-class weights against the same oracle
        # utility and per-world progressive-matched persistent-token budget.
        learned_candidates = []
        for weights in tune_cfg["utility_weight_profiles"]:
            utilities = []
            for world, queries in policy_samples:
                # Re-run the chosen progressive policy to obtain this world's
                # matched budget at this K; no eval world enters this calculation.
                progressive_result = evaluate_condition(
                    world, "progressive", K, condition_config=condition_config,
                    tuned=progressive_params, queries=queries,
                    expected_phase="policy_tune",
                )
                learned_result = evaluate_condition(
                    world, "evict_learned", K, condition_config=condition_config,
                    tuned={"utility_weights": dict(weights)},
                    store_budget_tokens=int(progressive_result.mean_accounting["store_tokens"]),
                    queries=queries, expected_phase="policy_tune",
                )
                utilities.append(learned_result.utility)
            learned_candidates.append((sum(utilities) / len(utilities), dict(weights)))
        learned_best = max(
            learned_candidates,
            key=lambda row: (row[0], tuple(sorted(row[1].items()))),
        )[1]
        by_k[str(K)] = {
            "progressive": progressive_params,
            "evict_learned": {"utility_weights": learned_best},
            "matched_store_budget_tokens": matched_budget,
            "policy_utility": best_utility,
        }
        print(f"[oracle-tune] K={K} best policy utility={best_utility:.4f}", flush=True)
    return {"by_K": by_k, "tuned_world_seeds": [world.seed for world in worlds]}


def _policy_query_sample(world: World, per_type: int) -> list[Any]:
    """Deterministic no-replacement, type-stratified policy-tuning sample."""
    grouped: dict[str, list[Any]] = defaultdict(list)
    seen: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    for query in world.queries:
        signature = (query.tokens, query.answer)
        if signature not in seen[query.query_type]:
            grouped[query.query_type].append(query)
            seen[query.query_type].add(signature)
    return [query for kind in sorted(grouped) for query in grouped[kind][:per_type]]


def context_storage_bits(tokens: int, items: int, vocab_size: int, metadata_fields: int) -> float:
    from aftertrace.accounting import storage_bits
    return storage_bits(tokens, items, vocab_size=vocab_size, metadata_fields=metadata_fields)
