from __future__ import annotations

from collections import Counter

import pytest

from aftertrace.world.generator import make_world
from aftertrace.world.queries import solve_query


@pytest.mark.slow
def test_make_world_is_deterministic(world_config: dict) -> None:
    first = make_world(2_000_000, world_config)
    second = make_world(2_000_000, world_config)
    assert first == second


@pytest.mark.parametrize("p_redundancy", [0.2, 0.6, 0.9])
@pytest.mark.slow
def test_moved_prior_match_rate_over_three_worlds(
    world_config: dict, p_redundancy: float
) -> None:
    matches = 0
    total = 0
    for offset in range(3):
        cfg = {**world_config, "p_redundancy": p_redundancy}
        world = make_world(2_000_010 + offset, cfg)
        for episode in world.episodes:
            for index, fact in enumerate(episode.facts):
                if fact.relation == "MOVED":
                    assert index > 0
                    conditioning_work = episode.facts[index - 1]
                    assert conditioning_work.relation == "WORKS"
                    assert conditioning_work.subject == fact.subject
                    total += 1
                    matches += world.grammar.predicts_moved(
                        conditioning_work.object, fact.object
                    )
    assert total > 1000
    assert abs(matches / total - p_redundancy) <= 0.03


def test_conflicts_supersede_works_and_moved_together(world_config: dict) -> None:
    world = make_world(2_000_023, world_config)
    state: dict[str, tuple[str, str]] = {}
    conflicted: set[str] = set()
    for episode in world.episodes:
        for index, fact in enumerate(episode.facts):
            if fact.relation != "MOVED":
                continue
            work = episode.facts[index - 1]
            previous = state.get(fact.subject)
            current = (work.object, fact.object)
            if previous is not None and current != previous:
                assert work.relation == "WORKS"
                assert work.subject == fact.subject
                assert work.object != previous[0]
                assert fact.object != previous[1]
                conflicted.add(fact.subject)
            state[fact.subject] = current
    expected = round(world_config["conflict_fraction"] * world_config["n_entities"])
    assert len(conflicted) == expected


@pytest.mark.slow
def test_episode_fact_keys_are_unique_across_three_worlds(world_config: dict) -> None:
    for seed in range(2_000_040, 2_000_043):
        world = make_world(seed, world_config)
        for episode in world.episodes:
            keys = [(fact.subject, fact.relation) for fact in episode.facts]
            assert len(keys) == len(set(keys))


def test_place_tokens_only_follow_at(world_config: dict) -> None:
    world = make_world(2_000_043, world_config)
    for episode in world.episodes:
        for index, token in enumerate(episode.tokens):
            if token.startswith("P_"):
                assert episode.tokens[index - 1] == "AT"
        for fact in episode.facts:
            if fact.relation == "OWNS":
                assert fact.object.startswith("J_")
    for query in world.queries:
        for index, token in enumerate(query.tokens):
            if token.startswith("P_"):
                assert query.tokens[index - 1] == "AT"


def test_queries_are_symbolically_derivable_from_declared_evidence(
    world_config: dict,
) -> None:
    world = make_world(2_000_020, world_config)
    observed_types = Counter(query.query_type for query in world.queries)
    assert set(observed_types) == {"A", "B", "C", "E", "R"}
    conflict_relations = {
        query.tokens[2] for query in world.queries if query.query_type == "C"
    }
    assert conflict_relations == {"WORKS", "MOVED"}
    for query in world.queries:
        assert all(index <= query.after_episode for index in query.evidence_episode_ids)
        assert solve_query(query, world.episodes, world.grammar) == query.answer


def test_type_b_uses_distinct_earlier_work_and_nonconflict_target(
    world_config: dict,
) -> None:
    world = make_world(2_000_024, world_config)
    work_histories: dict[str, list[tuple[int, str]]] = {}
    for episode in world.episodes:
        for fact in episode.facts:
            if fact.relation == "WORKS":
                work_histories.setdefault(fact.subject, []).append((episode.id, fact.object))
    conflict_entities = {
        entity
        for entity, events in work_histories.items()
        if len({value for _, value in events}) >= 2
    }
    queries = [query for query in world.queries if query.query_type == "B"]
    assert queries
    for query in queries:
        assert len(query.evidence_episode_ids) == 2
        assert len(set(query.evidence_episode_ids)) == 2
        work_episode_id, told_episode_id = query.evidence_episode_ids
        assert work_episode_id < told_episode_id
        told_episode = world.episodes[told_episode_id]
        subject, place, time = query.tokens[2], query.tokens[5], query.tokens[7]
        told = next(
            fact
            for fact in told_episode.facts
            if fact.subject == subject
            and fact.relation == "TOLD"
            and fact.extra == ("AT", place)
            and told_episode.time == time
        )
        target = told.object
        assert target not in conflict_entities
        assert any(
            fact.subject == target and fact.relation == "WORKS"
            for fact in world.episodes[work_episode_id].facts
        )


def test_type_c_uses_superseding_entities_complete_history_and_later_bucket(
    world_config: dict,
) -> None:
    world = make_world(2_000_025, world_config)
    queries = [query for query in world.queries if query.query_type == "C"]
    assert queries
    for query in queries:
        entity, relation = query.tokens[1], query.tokens[2]
        events = [
            (episode.id, episode.time, fact.object)
            for episode in world.episodes[: query.after_episode + 1]
            for fact in episode.facts
            if fact.subject == entity and fact.relation == relation
        ]
        distinct = []
        for event in events:
            if not distinct or event[2] != distinct[-1][2]:
                distinct.append(event)
        assert len(distinct) >= 2
        assert int(distinct[1][1].split("_")[1]) > int(distinct[0][1].split("_")[1])
        assert query.evidence_episode_ids == tuple(dict.fromkeys(
            episode_id for episode_id, _, _ in events
        ))


def test_type_r_cues_are_target_free_unique_and_log_competitors(
    world_config: dict,
) -> None:
    world = make_world(2_000_026, world_config)
    queries = [query for query in world.queries if query.query_type == "R"]
    assert queries
    latest: dict[str, dict[str, str]] = {"WORKS": {}, "MOVED": {}}
    superseding_events: set[tuple[int, str]] = set()
    for episode in world.episodes:
        current = {
            relation: {
                fact.subject: fact.object
                for fact in episode.facts
                if fact.relation == relation
            }
            for relation in ("WORKS", "MOVED")
        }
        for entity in current["WORKS"].keys() & current["MOVED"].keys():
            if (
                entity in latest["WORKS"]
                and entity in latest["MOVED"]
                and current["WORKS"][entity] != latest["WORKS"][entity]
                and current["MOVED"][entity] != latest["MOVED"][entity]
            ):
                superseding_events.add((episode.id, entity))
        for relation in ("WORKS", "MOVED"):
            latest[relation].update(current[relation])
    for query in queries:
        assert isinstance(query.competitor_count, int)
        assert isinstance(query.from_superseding, bool)
        assert len(query.evidence_episode_ids) == 1
        evidence_id = query.evidence_episode_ids[0]
        evidence = world.episodes[evidence_id]
        partner, place, time = query.tokens[2], query.tokens[4], query.tokens[5]
        cue = (partner, place, time)
        matching_interactions = [
            fact
            for fact in evidence.facts
            if fact.relation in {"TOLD", "MET"}
            and fact.subject == partner
            and fact.extra == ("AT", place)
        ]
        assert evidence.time == time
        assert matching_interactions
        moved_entities = {
            fact.subject for fact in evidence.facts if fact.relation == "MOVED"
        }
        targets = {fact.object for fact in matching_interactions if fact.object in moved_entities}
        assert len(targets) == 1
        assert targets.isdisjoint(query.tokens)
        assert query.from_superseding == (
            (evidence_id, next(iter(targets))) in superseding_events
        )
        assert sum(
            all(feature in episode.tokens for feature in cue)
            for episode in world.episodes
        ) == 1
        competitors = {
            episode.id
            for episode in world.episodes[:evidence_id]
            if sum(feature in episode.tokens for feature in cue) == 2
        }
        assert query.competitor_count == len(competitors)


@pytest.mark.slow
def test_competitor_count_matches_brute_force_for_200_r_queries(
    world_config: dict,
) -> None:
    world = make_world(2_000_027, world_config)
    queries = [query for query in world.queries if query.query_type == "R"][:200]
    assert len(queries) == 200
    for query in queries:
        evidence_id = query.evidence_episode_ids[0]
        cue = (query.tokens[2], query.tokens[4], query.tokens[5])
        competitor_episode_ids: set[int] = set()
        for episode in world.episodes[:evidence_id]:
            # Count an episode once based only on its three feature-presence
            # booleans. Interaction type, direction, and multiple matching
            # facts cannot add duplicate counts.
            matching_features = sum(feature in episode.tokens for feature in cue)
            if matching_features == 2:
                competitor_episode_ids.add(episode.id)
        assert query.competitor_count == len(competitor_episode_ids)


def test_logged_restatement_rates_match_stream_and_cap(world_config: dict) -> None:
    world = make_world(2_000_028, world_config)
    repeated: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    seen: set[tuple[str, str, str]] = set()
    for episode in world.episodes:
        for fact in episode.facts:
            if fact.relation not in {"WORKS", "MOVED"}:
                continue
            key = (fact.subject, fact.relation, fact.object)
            totals[fact.relation] += 1
            repeated[fact.relation] += int(key in seen)
            seen.add(key)
    for relation in ("WORKS", "MOVED"):
        realized = repeated[relation] / totals[relation]
        assert world.restatement_rates[relation] == pytest.approx(realized)
        assert realized <= world_config["p_restate"]
    combined = sum(repeated.values()) / sum(totals.values())
    assert world.restatement_rates["combined"] == pytest.approx(combined)


def test_old_query_quota(world_config: dict) -> None:
    world = make_world(2_000_021, world_config)
    age_threshold = world_config["old_query_age_fraction"] * len(world.episodes)
    old = sum(query.lag is not None and query.lag > age_threshold for query in world.queries)
    assert old / len(world.queries) >= world_config["old_query_min_fraction"]


def test_type_e_queries_never_have_evidence(world_config: dict) -> None:
    world = make_world(2_000_022, world_config)
    unavailable = [query for query in world.queries if query.query_type == "E"]
    assert unavailable
    assert all(query.answer == "ABSTAIN" for query in unavailable)
    assert all(query.evidence_episode_ids == () for query in unavailable)
    assert not any(
        fact.relation == "STUDIED"
        for episode in world.episodes
        for fact in episode.facts
    )


def test_labels_never_name_a_value_absent_from_the_world(world_config: dict) -> None:
    world = make_world(2_000_029, world_config)
    world_values = {
        fact.object for episode in world.episodes for fact in episode.facts
    }
    missing_moved = 0
    for query in world.queries:
        if query.answer != "ABSTAIN":
            assert query.answer in world_values
        else:
            entity, relation = query.tokens[1], query.tokens[2]
            assert not any(
                fact.subject == entity and fact.relation == relation
                for episode in world.episodes
                for fact in episode.facts
            )
            if relation == "MOVED":
                missing_moved += 1
            assert query.evidence_episode_ids == ()
    assert missing_moved > 0


@pytest.mark.slow
@pytest.mark.parametrize(
    ("seed", "p_redundancy"),
    ((2_000_060, 0.2), (2_000_061, 0.6), (2_000_062, 0.9)),
)
def test_eval_worlds_meet_unique_query_floors_for_each_stage0_type(
    world_config: dict, seed: int, p_redundancy: float
) -> None:
    world = make_world(seed, {**world_config, "phase": "eval", "p_redundancy": p_redundancy})
    minimum_unique = {"A": 300, "B": 300, "C": 300, "E": 300, "R": 250}
    for query_type, minimum in minimum_unique.items():
        unique_queries = {
            query.tokens for query in world.queries if query.query_type == query_type
        }
        assert len(unique_queries) >= minimum, (query_type, len(unique_queries))
    for mixture in world.demand_mixtures:
        type_shares: Counter[str] = Counter()
        for key, weight in mixture.items():
            query_type = key.split(":", 1)[0]
            type_shares[query_type] += weight
        assert all(
            type_shares[query_type] >= 0.08 - 1e-10
            for query_type in ("A", "B", "C", "E", "R")
        )


@pytest.mark.parametrize("seed", range(2_000_030, 2_000_035))
@pytest.mark.slow
def test_eval_world_has_at_least_5000_poisson_queries(
    world_config: dict, seed: int
) -> None:
    world = make_world(seed, world_config)
    assert len(world.queries) >= world_config["query_min_eval"]
