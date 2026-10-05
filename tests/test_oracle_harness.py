from __future__ import annotations

import random

import pytest

from aftertrace.eval.harness import _policy_query_sample, _retrieve_two_hop, evaluate_condition, tune_parameters
from aftertrace.memory.base import ItemView, MemoryItem
from aftertrace.memory.conditions import make_condition
from aftertrace.memory.retriever import SharedRetriever
from aftertrace.reader.oracle import OracleReader
from scripts.oracle_frontier import (
    _decision_from_pair_ratios, _frontier_k, _matched_k_bootstrap,
    _matched_k_fallback, _ratio_bound,
)
from aftertrace.world.generator import make_world


@pytest.fixture(scope="module")
def eval_world(world_config):
    return make_world(2_000_101, {**world_config, "phase": "eval"})


def test_oracle_full_context_answers_supported_queries_and_abstains_e(eval_world):
    by_type = {}
    for query in eval_world.queries:
        by_type.setdefault(query.query_type, []).append(query)
    sample = [q for kind in ("A", "B", "C", "E", "R") for q in by_type[kind]]
    result = evaluate_condition(
        eval_world, "full_oracle", 384,
        condition_config={"full_oracle": {}}, queries=sample,
    )
    assert result.per_type["E"]["accuracy"] == 1.0
    for kind in ("A", "B", "C", "R"):
        # Any oracle abstentions here indicate a parsing/provenance defect.
        group = [row for row in result.results if row.query_type == kind]
        assert group
        assert all(row.answer == row.target for row in group)
        assert all(row.confidence > 0.5 for row in group)


def test_context_never_exceeds_budget(eval_world):
    query = next(q for q in eval_world.queries if q.query_type in {"A", "E"})
    result = evaluate_condition(
        eval_world, "raw_offload", 384,
        condition_config={"raw_offload": {}}, queries=[query],
    )
    assert result.results[0].accounting.context_tokens <= 384


def test_two_hop_retrieval_expands_entities_and_charges_both_passes():
    first_item = ItemView(
        MemoryItem("first", 4, ("EP", "T_0", "E_11", "TOLD", "E_9", "EOS"), None),
        ("EP", "T_0", "E_11", "TOLD", "E_9", "EOS"),
    )
    second_item = ItemView(
        MemoryItem("second", 3, ("EP", "T_1", "E_9", "WORKS", "O_2", "EOS"), None),
        ("EP", "T_1", "E_9", "WORKS", "O_2", "EOS"),
    )

    class TrackingRetriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, views, query_tokens, budget, *, max_items=None):
            self.calls.append((tuple(query_tokens), budget, max_items))
            if len(self.calls) == 1:
                selected, context_tokens = (first_item,), 6
            else:
                selected, context_tokens = (second_item,), 2
            return type("Result", (), {
                "selected": selected, "retrieval_ops": 7,
                "context_tokens": context_tokens,
            })()

    retriever = TrackingRetriever()
    selected, retrieval_ops = _retrieve_two_hop(
        retriever, [first_item, second_item], ("Q", "E_4", "WORKS", "A"), 8
    )
    assert retriever.calls == [
        (("Q", "E_4", "WORKS", "A"), 8, 1),
        (("Q", "E_4", "WORKS", "A", "E_9", "E_11"), 2, None),
    ]
    assert selected == [first_item, second_item]
    assert retrieval_ops == 14


def test_two_hop_retrieval_keeps_the_first_hop_and_finds_b_work_episode(world_config):
    eval_world = make_world(
        2_000_004, {**world_config, "phase": "eval", "p_redundancy": 0.6}
    )
    seen = set()
    queries = []
    for query in eval_world.queries:
        signature = (query.tokens, query.answer)
        if query.query_type == "B" and signature not in seen:
            seen.add(signature)
            queries.append(query)
        if len(queries) == 20:
            break
    assert len(queries) == 20

    condition = make_condition(
        "raw_offload", eval_world, {"raw_offload": {}}
    )
    retriever = SharedRetriever()
    queries_by_step = {}
    for query in queries:
        queries_by_step.setdefault(query.after_episode, []).append(query)
    retrieved_work_episodes = 0
    for episode in eval_world.episodes:
        condition.observe(episode, episode.id)
        for query in queries_by_step.get(episode.id, ()):
            selected, _ = _retrieve_two_hop(
                retriever, condition.views(query, episode.id), query.tokens, 384
            )
            selected_ids = {view.item.episode_id for view in selected}
            work_episode = min(query.evidence_episode_ids)
            retrieved_work_episodes += work_episode in selected_ids
    assert retrieved_work_episodes == 20


def test_qrender_keeps_second_hop_facts_and_packs_on_rendered_tokens(eval_world):
    query = next(query for query in eval_world.queries if query.query_type == "B")
    config = {"raw_offload_qrender": {}}
    condition = make_condition("raw_offload_qrender", eval_world, config)
    for episode in eval_world.episodes[:query.after_episode + 1]:
        condition.observe(episode, episode.id)
    views = condition.views(query, query.after_episode)
    render = lambda view, entities: condition.render_retrieved(
        view, query, extra_entities=entities
    )
    selected, _ = _retrieve_two_hop(
        SharedRetriever(), views, query.tokens, 32, render=render
    )
    assert selected
    rendered_total = sum(len(view.tokens) for view in selected)
    assert rendered_total <= 32
    assert sum(view.item.token_count for view in selected) > 32
    assert all(len(view.tokens) < view.item.token_count for view in selected)

    told_episode = next(
        episode for episode in eval_world.episodes
        if episode.id == query.target_episode_id
    )
    target = next(
        fact.object for fact in told_episode.facts
        if fact.relation == "TOLD" and fact.subject == query.tokens[2]
    )
    work_episode_id = next(
        episode_id for episode_id in query.evidence_episode_ids
        if episode_id != query.target_episode_id
    )
    work_view = next(view for view in views if view.item.episode_id == work_episode_id)
    rendered_work = condition.render_retrieved(
        work_view, query, extra_entities={target}
    )
    assert target in rendered_work.tokens
    assert "WORKS" in rendered_work.tokens
    retrieved_work = next(
        view for view in selected if view.item.episode_id == work_episode_id
    )
    assert target in retrieved_work.tokens and "WORKS" in retrieved_work.tokens


def test_tuning_rejects_eval_world_before_using_it(eval_world):
    with pytest.raises((AssertionError, ValueError)):
        tune_parameters([eval_world], {"progressive": {}})


def test_policy_tuning_uses_balanced_eval_query_strata(world_config):
    policy_world = make_world(1_000_001, {**world_config, "phase": "policy_tune"})
    sample = _policy_query_sample(policy_world, 300)
    counts = {kind: sum(query.query_type == kind for query in sample) for kind in ("A", "B", "C", "E", "R")}
    assert len(set(counts.values())) == 1
    assert 0 < next(iter(counts.values())) <= 300
    assert len({(query.query_type, query.tokens, query.answer) for query in sample}) == len(sample)


def test_oracle_confidence_reports_prediction_probability(eval_world):
    query = next(q for q in eval_world.queries if q.query_type == "E")
    prediction = OracleReader().predict([], query, eval_world.grammar, 0.6)
    assert prediction.answer == "ABSTAIN"
    assert prediction.confidence == 1.0


def test_oracle_utility_is_monotone_for_1000_nested_context_samples(eval_world):
    rng = random.Random(144)
    queries = rng.sample(eval_world.queries, 1000)
    reader = OracleReader()
    score = lambda answer, target: 1 if answer == target else (0 if answer == "ABSTAIN" else -1)
    for query in queries:
        prefix = eval_world.episodes[:query.after_episode + 1]
        selected = rng.sample(prefix, min(len(prefix), rng.randint(0, 5)))
        remaining = [episode for episode in prefix if episode not in selected]
        extra = rng.choice(remaining) if remaining else None
        smaller = [MemoryItem(str(ep.id), ep.id, ep.tokens, ep) for ep in selected]
        larger_eps = selected + ([extra] if extra is not None else [])
        larger = [MemoryItem(str(ep.id), ep.id, ep.tokens, ep) for ep in larger_eps]
        before = reader.predict(smaller, query, eval_world.grammar, 0.6)
        after = reader.predict(larger, query, eval_world.grammar, 0.6)
        assert score(after.answer, query.answer) >= score(before.answer, query.answer), (
            query.query_type, query.tokens, query.answer,
            [ep.id for ep in selected], extra.id if extra else None,
            before.answer, after.answer,
        )


def test_oracle_does_not_guess_from_partial_r_cue(eval_world):
    from aftertrace.reader.oracle import FactEvidence

    query = next(query for query in eval_world.queries if query.query_type == "R")
    partner, place, time = query.tokens[2], query.tokens[4], query.tokens[5]
    facts = (
        FactEvidence(partner, "TOLD", "E_17", ("AT", place), 5, "T_3", "EP", 0),
        FactEvidence("E_17", "MOVED", "C_2", (), 5, "T_3", "EP", 1),
        FactEvidence(partner, "MET", "E_19", ("AT", place), 6, "T_3", "EP", 0),
        FactEvidence("E_19", "MOVED", "C_9", (), 6, "T_3", "EP", 1),
    )
    prediction = OracleReader._predict_r(facts, query.tokens)
    assert prediction.answer == "ABSTAIN"


def test_oracle_abstains_when_summary_values_conflict_or_are_insufficient(eval_world):
    query = next(
        query for query in eval_world.queries
        if query.query_type == "A" and query.tokens[2] == "MOVED"
    )
    entity, _relation = query.tokens[1:3]
    alternate = "C_59" if query.answer != "C_59" else "C_58"
    conflicting = [
        MemoryItem("sum-old", 99, ("SUM", entity, "MOVED", query.answer, "EOS"), None),
        MemoryItem("sum-new", 199, ("SUM", entity, "MOVED", alternate, "EOS"), None),
    ]
    absent = [
        MemoryItem("sum-work", 199, ("SUM", entity, "WORKS", "O_1", "EOS"), None),
    ]
    reader = OracleReader()
    assert reader.predict(conflicting, query, eval_world.grammar, 0.6).answer == "ABSTAIN"
    assert reader.predict(absent, query, eval_world.grammar, 0.6).answer == "ABSTAIN"


def test_summary_r_utility_is_nonnegative(eval_world):
    queries = [query for query in eval_world.queries if query.query_type == "R"][:40]
    result = evaluate_condition(
        eval_world, "summary", 384,
        condition_config={"summary": {"summary_interval": 100}},
        queries=queries,
    )
    assert result.per_type["R"]["utility"] >= 0.0
    assert all(row.answer == "ABSTAIN" for row in result.results)


def test_frontier_reports_censored_kstar_and_ratio_bounds():
    rows = [
        {"condition": "full_oracle", "p_redundancy": 0.6, "K": 32, "utility": 1.0},
        {"condition": "full_oracle", "p_redundancy": 0.6, "K": 64, "utility": 1.0},
        {"condition": "summary", "p_redundancy": 0.6, "K": 32, "utility": 0.1},
        {"condition": "summary", "p_redundancy": 0.6, "K": 64, "utility": 0.2},
    ]
    k_star, censored = _frontier_k(rows, "summary", 0.6, 0.8)
    assert k_star == 64 and censored
    assert _ratio_bound(384, True, 128, False, 384) == (3.0, ">")
    assert _ratio_bound(128, False, 384, True, 384) == (1 / 3, "<")


def test_frontier_decision_accounts_for_every_censoring_bound():
    exact_low = [{"ratio": 0.8, "bound": "="}, {"ratio": 1.19, "bound": "<"}]
    assert _decision_from_pair_ratios(exact_low) == "STOP"
    assert _decision_from_pair_ratios(exact_low + [{"ratio": 1.1, "bound": ">"}]) == "REDUCED"
    assert _decision_from_pair_ratios(exact_low + [{"ratio": float("nan"), "bound": "interval"}]) == "REDUCED"
    assert _decision_from_pair_ratios(exact_low + [{"ratio": 1.5, "bound": "="}]) == "PASS"


def test_matched_k_bootstrap_is_paired_deterministic_and_drives_fallback():
    rows = []
    values = {
        "progressive": (0.50, 0.50, 0.50, 0.50),
        "raw_offload": (0.48, 0.48, 0.48, 0.48),
        "raw_offload_qrender": (0.49, 0.49, 0.49, 0.49),
    }
    for condition, scores in values.items():
        for seed, utility in enumerate(scores, start=1):
            rows.append({
                "seed": seed, "p_redundancy": 0.2, "K": 32,
                "condition": condition, "utility": utility,
            })
    first = _matched_k_bootstrap(rows, [0.2], [32], replicates=1000, seed=17)
    second = _matched_k_bootstrap(rows, [0.2], [32], replicates=1000, seed=17)
    assert first == second
    assert first[0]["difference"] == pytest.approx(0.01)
    assert first[0]["ci95_upper"] == pytest.approx(0.01)
    censored = [
        {"p_redundancy": 0.2, "target": target, "condition": condition, "censored": True}
        for target in (0.8, 0.9, 0.95)
        for condition in ("progressive", "best_raw")
    ]
    assert _matched_k_fallback(censored, first).startswith("STOP")
    over_threshold = [{**first[0], "ci95_upper": 0.03}]
    assert _matched_k_fallback(censored, over_threshold).startswith("NO STOP")


def test_summary_exposes_snapshot_items_after_rewrite(eval_world):
    condition = make_condition("summary", eval_world, {"summary": {"summary_interval": 100}})
    for episode in eval_world.episodes[:100]:
        condition.observe(episode, episode.id)
    views = condition.views(eval_world.queries[0], 99)
    assert views
    assert all(view.tokens[0] == "SUM" for view in views)
