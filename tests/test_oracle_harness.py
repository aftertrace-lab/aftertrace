from __future__ import annotations

import pytest

from aftertrace.eval.harness import evaluate_condition, tune_parameters
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


def test_tuning_rejects_eval_world_before_using_it(eval_world):
    with pytest.raises((AssertionError, ValueError)):
        tune_parameters([eval_world], {"progressive": {}})


def test_oracle_confidence_reports_prediction_probability(eval_world):
    query = next(q for q in eval_world.queries if q.query_type == "E")
    prediction = OracleReader().predict([], query, eval_world.grammar, 0.6)
    assert prediction.answer == "ABSTAIN"
    assert prediction.confidence == 1.0


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
