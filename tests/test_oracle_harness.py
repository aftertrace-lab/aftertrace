from __future__ import annotations

import random

import pytest

from aftertrace.eval.harness import _retrieve_two_hop, evaluate_condition, tune_parameters
from aftertrace.memory.base import ItemView, MemoryItem
from aftertrace.memory.conditions import make_condition
from aftertrace.reader.oracle import OracleReader
from scripts.oracle_frontier import _frontier_k, _ratio_bound
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

        def retrieve(self, views, query_tokens, budget):
            self.calls.append((tuple(query_tokens), budget))
            selected = (first_item,) if len(self.calls) == 1 else (second_item,)
            return type("Result", (), {"selected": selected, "retrieval_ops": 7, "context_tokens": 6})()

    retriever = TrackingRetriever()
    selected, retrieval_ops = _retrieve_two_hop(
        retriever, [first_item, second_item], ("Q", "E_4", "WORKS", "A"), 8
    )
    assert retriever.calls == [
        (("Q", "E_4", "WORKS", "A"), 8),
        (("Q", "E_4", "WORKS", "A", "E_9", "E_11"), 8),
    ]
    assert selected == [second_item]
    assert retrieval_ops == 14


def test_tuning_rejects_eval_world_before_using_it(eval_world):
    with pytest.raises((AssertionError, ValueError)):
        tune_parameters([eval_world], {"progressive": {}})


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


def test_summary_exposes_snapshot_items_after_rewrite(eval_world):
    condition = make_condition("summary", eval_world, {"summary": {"summary_interval": 100}})
    for episode in eval_world.episodes[:100]:
        condition.observe(episode, episode.id)
    views = condition.views(eval_world.queries[0], 99)
    assert views
    assert all(view.tokens[0] == "SUM" for view in views)
