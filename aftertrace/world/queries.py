"""Interleaved query generation, provenance, and symbolic solving."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    from .generator import Episode, Fact
    from .grammar import GrammarPrior


@dataclass(frozen=True)
class Query:
    id: int
    after_episode: int
    query_type: str
    fact_class: str
    tokens: tuple[str, ...]
    answer: str
    evidence_episode_ids: tuple[int, ...]
    target_episode_id: int | None
    lag: int | None
    demand_half: int
    competitor_count: int | None = None
    from_superseding: bool | None = None

    @property
    def rendered(self) -> tuple[str, ...]:
        return (*self.tokens, self.answer)


_DEMAND_KEYS = (
    ("A", "core_relational"),
    ("A", "causal_linked"),
    ("A", "temporal"),
    ("A", "social"),
    ("A", "incidental"),
    ("B", "social"),
    ("C", "core_relational"),
    ("E", "unavailable"),
    ("R", "reinstatement"),
)
_MIN_QUERY_TYPE_SHARE = 0.08


def _key_name(key: tuple[str, str]) -> str:
    return f"{key[0]}:{key[1]}"


def _floor_type_mixture(weights: np.ndarray) -> np.ndarray:
    """Floor aggregate query-type mass, preserving within-type proportions."""

    types = tuple(dict.fromkeys(query_type for query_type, _ in _DEMAND_KEYS))
    masses = {
        query_type: sum(
            float(weights[index])
            for index, (candidate_type, _) in enumerate(_DEMAND_KEYS)
            if candidate_type == query_type
        )
        for query_type in types
    }
    fixed: dict[str, float] = {}
    active = list(types)
    remaining = 1.0
    while active:
        active_total = sum(masses[query_type] for query_type in active)
        below_floor = [
            query_type
            for query_type in active
            if remaining * masses[query_type] / active_total < _MIN_QUERY_TYPE_SHARE
        ]
        if not below_floor:
            for query_type in active:
                fixed[query_type] = remaining * masses[query_type] / active_total
            break
        for query_type in below_floor:
            fixed[query_type] = _MIN_QUERY_TYPE_SHARE
            remaining -= _MIN_QUERY_TYPE_SHARE
            active.remove(query_type)
    result = np.zeros_like(weights, dtype=float)
    for query_type in types:
        indices = [
            index
            for index, (candidate_type, _) in enumerate(_DEMAND_KEYS)
            if candidate_type == query_type
        ]
        original = weights[indices]
        within = original / original.sum() if original.sum() else np.full(len(indices), 1 / len(indices))
        result[indices] = fixed[query_type] * within
    return result


def _histories(episodes: tuple[Episode, ...]) -> dict[str, dict[str, list[tuple[int, Fact]]]]:
    histories: dict[str, dict[str, list[tuple[int, Fact]]]] = {}
    for episode in episodes:
        for fact in episode.facts:
            histories.setdefault(fact.relation, {}).setdefault(fact.subject, []).append(
                (episode.id, fact)
            )
    return histories


def _at_or_before(events: list[tuple[int, Fact]], step: int) -> list[tuple[int, Fact]]:
    return [event for event in events if event[0] <= step]


def _latest_candidates(
    histories: Mapping[str, Mapping[str, list[tuple[int, Fact]]]],
    relation: str,
    step: int,
) -> list[tuple[int, Fact]]:
    result = []
    for events in histories.get(relation, {}).values():
        prior = _at_or_before(events, step)
        if prior:
            result.append(prior[-1])
    return result


def _sample_lag(rng: np.random.Generator, step: int, cfg: Mapping[str, Any]) -> int:
    low = int(cfg["lag_min"])
    configured_max = cfg["lag_max"]
    high = min(step + 1, int(configured_max) if configured_max is not None else len(range(int(cfg["n_episodes"]))))
    if high <= low:
        return max(0, high)
    # A continuous log-uniform draw rounded to an integer yields a heavy tail.
    return int(round(np.exp(rng.uniform(np.log(low), np.log(high)))))


def _choose_near(
    candidates: list[tuple[int, Fact]], target_step: int, rng: np.random.Generator
) -> tuple[int, Fact] | None:
    if not candidates:
        return None
    distances = np.asarray([abs(episode_id - target_step) for episode_id, _ in candidates])
    best = np.flatnonzero(distances == distances.min())
    return candidates[int(rng.choice(best))]


def _query_a(
    fact_class: str,
    step: int,
    target_step: int,
    histories: Mapping[str, Mapping[str, list[tuple[int, Fact]]]],
    rng: np.random.Generator,
) -> tuple[tuple[str, ...], str, tuple[int, ...], int | None, str] | None:
    relations_by_class = {
        "core_relational": ("WORKS",),
        "causal_linked": ("MOVED",),
        "temporal": ("JOINED_YEAR",),
        "social": ("TOLD", "MET"),
        "incidental": ("COLOR", "OWNS"),
    }
    relation = str(rng.choice(relations_by_class[fact_class]))
    chosen = _choose_near(_latest_candidates(histories, relation, step), target_step, rng)
    if chosen is None:
        return None
    episode_id, fact = chosen
    return (("Q", fact.subject, relation, "A"), fact.object, (episode_id,), episode_id, fact_class)


def _query_b(
    step: int,
    target_step: int,
    histories: Mapping[str, Mapping[str, list[tuple[int, Fact]]]],
    episodes: tuple[Episode, ...],
    rng: np.random.Generator,
) -> tuple[tuple[str, ...], str, tuple[int, ...], int, str] | None:
    candidates: list[tuple[int, Fact, int, Fact]] = []
    conflict_entities = {
        entity
        for entity, events in histories.get("WORKS", {}).items()
        if len(_distinct_events(events)) >= 2
    }
    for events in histories.get("TOLD", {}).values():
        for told_episode, told in _at_or_before(events, step):
            if told.object in conflict_entities:
                continue
            works = [
                event
                for event in histories.get("WORKS", {}).get(told.object, [])
                if event[0] < told_episode
            ]
            if works:
                work_episode, work = works[-1]
                told_episode_has_target_work = any(
                    event[0] == told_episode
                    for event in histories.get("WORKS", {}).get(told.object, [])
                )
                if told_episode_has_target_work:
                    continue
                candidates.append((told_episode, told, work_episode, work))
    if not candidates:
        return None
    distances = np.asarray([abs(item[0] - target_step) for item in candidates])
    selected = candidates[int(rng.choice(np.flatnonzero(distances == distances.min())))]
    told_episode, told, work_episode, work = selected
    place = told.extra[1]
    evidence = tuple(sorted({told_episode, work_episode}))
    tokens = (
        "Q", "HOP2", told.subject, "TOLD", "AT", place,
        "TIME", episodes[told_episode].time, "WORKS", "A",
    )
    return tokens, work.object, evidence, told_episode, "social"


def _unique_b_query_specs(
    histories: Mapping[str, Mapping[str, list[tuple[int, Fact]]]],
    episodes: tuple[Episode, ...],
) -> list[tuple[int, tuple[tuple[str, ...], str, tuple[int, ...], int, str]]]:
    """Enumerate unique B forms over valid earlier WORKS evidence episodes."""
    conflicts = {
        entity for entity, events in histories.get("WORKS", {}).items()
        if len(_distinct_events(events)) >= 2
    }
    specs: dict[tuple[str, ...], tuple[int, tuple[tuple[str, ...], str, tuple[int, ...], int, str]]] = {}
    for told_events in histories.get("TOLD", {}).values():
        for told_episode, told in told_events:
            if told.object in conflicts:
                continue
            for work_episode, work in histories.get("WORKS", {}).get(told.object, []):
                if work_episode >= told_episode:
                    continue
                tokens = (
                    "Q", "HOP2", told.subject, "TOLD", "AT", told.extra[1],
                    "TIME", episodes[told_episode].time, "WORKS", "A",
                )
                evidence = tuple(sorted({told_episode, work_episode}))
                specs.setdefault(
                    tokens,
                    (told_episode, (tokens, work.object, evidence, told_episode, "social")),
                )
    return list(specs.values())


def _distinct_events(events: list[tuple[int, Fact]]) -> list[tuple[int, Fact]]:
    distinct: list[tuple[int, Fact]] = []
    for event in events:
        if not distinct or event[1].object != distinct[-1][1].object:
            distinct.append(event)
    return distinct


def _unique_c_query_specs(
    histories: Mapping[str, Mapping[str, list[tuple[int, Fact]]]],
) -> list[tuple[int, tuple[tuple[str, ...], str, tuple[int, ...], int, str]]]:
    """Enumerate unique CURRENT/PREV queries supported by superseding facts."""

    specs: list[tuple[int, tuple[tuple[str, ...], str, tuple[int, ...], int, str]]] = []
    for relation in ("WORKS", "MOVED"):
        for entity, events in histories.get(relation, {}).items():
            distinct = _distinct_events(events)
            if len(distinct) < 2:
                continue
            evidence = tuple(dict.fromkeys(episode_id for episode_id, _ in events))
            latest_evidence = max(evidence)
            for operator, target_index in (("CURRENT", -1), ("PREV", -2)):
                target_episode, fact = distinct[target_index]
                tokens = ("Q", entity, relation, operator, "A")
                built = (tokens, fact.object, evidence, target_episode, "core_relational")
                specs.append((latest_evidence, built))
    return specs


def _query_c(
    step: int,
    histories: Mapping[str, Mapping[str, list[tuple[int, Fact]]]],
    rng: np.random.Generator,
    required_relation: str | None = None,
    required_operator: str | None = None,
) -> tuple[tuple[str, ...], str, tuple[int, ...], int, str] | None:
    candidates: list[
        tuple[str, str, list[tuple[int, Fact]], list[tuple[int, Fact]]]
    ] = []
    relations = (required_relation,) if required_relation is not None else ("WORKS", "MOVED")
    for relation in relations:
        for entity, events in histories.get(relation, {}).items():
            prior_events = _at_or_before(events, step)
            distinct = _distinct_events(prior_events)
            if len(distinct) >= 2:
                candidates.append((entity, relation, prior_events, distinct))
    if not candidates:
        return None
    entity, relation, events, distinct = candidates[int(rng.integers(len(candidates)))]
    operator = required_operator or str(rng.choice(("CURRENT", "PREV")))
    all_evidence = tuple(dict.fromkeys(episode_id for episode_id, _ in events))
    if operator == "CURRENT":
        target = distinct[-1]
    else:
        target = distinct[-2]
    tokens = ("Q", entity, relation, operator, "A")
    return tokens, target[1].object, all_evidence, target[0], "core_relational"


def _query_e(step: int, episodes: tuple[Episode, ...], rng: np.random.Generator):
    entities = sorted({entity for episode in episodes[: step + 1] for entity in episode.entities})
    if not entities:
        return None
    moved_entities = {
        fact.subject
        for episode in episodes
        for fact in episode.facts
        if fact.relation == "MOVED"
    }
    missing_moved = [entity for entity in entities if entity not in moved_entities]
    if missing_moved and rng.random() < 0.5:
        entity = str(rng.choice(missing_moved))
        relation = "MOVED"
    else:
        entity = str(rng.choice(entities))
        relation = "STUDIED"
    return (("Q", entity, relation, "A"), "ABSTAIN", (), None, "unavailable")


def _superseding_entity_events(episodes: tuple[Episode, ...]) -> set[tuple[int, str]]:
    """Identify same-episode WORKS+MOVED changes from earlier entity values."""
    latest: dict[str, dict[str, str]] = {"WORKS": {}, "MOVED": {}}
    result: set[tuple[int, str]] = set()
    for episode in episodes:
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
                result.add((episode.id, entity))
        for relation in ("WORKS", "MOVED"):
            latest[relation].update(current[relation])
    return result


def _build_r_candidates(
    episodes: tuple[Episode, ...],
) -> list[tuple[int, str, str, str, str, int, bool]]:
    """Return R targets with cue, answer, competitors, and superseding provenance."""

    superseding_events = _superseding_entity_events(episodes)
    raw_candidates: list[tuple[int, str, str, str, str, bool]] = []
    for episode in episodes:
        moved = {
            fact.subject: fact.object for fact in episode.facts if fact.relation == "MOVED"
        }
        for fact in episode.facts:
            if fact.relation not in {"TOLD", "MET"} or fact.extra[:1] != ("AT",):
                continue
            partner = fact.subject
            place = fact.extra[1]
            target = fact.object
            if target in moved and target != partner:
                raw_candidates.append((
                    episode.id, target, partner, place, moved[target],
                    (episode.id, target) in superseding_events,
                ))

    targets_by_cue: dict[tuple[str, str, str], set[tuple[str, str]]] = {}
    for episode_id, target, partner, place, answer, _from_superseding in raw_candidates:
        cue = (partner, place, episodes[episode_id].time)
        targets_by_cue.setdefault(cue, set()).add((target, answer))

    candidates: list[tuple[int, str, str, str, str, int, bool]] = []
    for episode_id, target, partner, place, answer, from_superseding in raw_candidates:
        cue = (partner, place, episodes[episode_id].time)
        full_matches = [
            episode.id
            for episode in episodes
            if all(feature in episode.tokens for feature in cue)
        ]
        if len(full_matches) != 1 or len(targets_by_cue[cue]) != 1:
            continue
        competitor_episodes = {
            earlier.id
            for earlier in episodes[:episode_id]
            if sum(feature in earlier.tokens for feature in cue) == 2
        }
        candidates.append(
            (
                episode_id, target, partner, place, answer,
                len(competitor_episodes), from_superseding,
            )
        )
    return candidates


def _query_r(
    step: int,
    target_step: int,
    episodes: tuple[Episode, ...],
    candidates: list[tuple[int, str, str, str, str, int, bool]],
    rng: np.random.Generator,
    maximum_target: int | None = None,
    required_candidate: tuple[int, str, str, str, str, int, bool] | None = None,
):
    eligible = ([required_candidate] if required_candidate is not None else [
        candidate for candidate in candidates
        if candidate[0] <= step
        and (maximum_target is None or candidate[0] <= maximum_target)
    ])
    if not eligible:
        return None
    distances = np.asarray([abs(candidate[0] - target_step) for candidate in eligible])
    chosen = eligible[int(rng.choice(np.flatnonzero(distances == distances.min())))]
    episode_id, target, partner, place, answer, competitor_count, from_superseding = chosen
    episode = episodes[episode_id]
    tokens = ("Q", "CTX", partner, "AT", place, episode.time, "MOVED", "A")
    assert target not in tokens
    return (
        tokens,
        answer,
        (episode_id,),
        episode_id,
        "reinstatement",
        competitor_count,
        from_superseding,
    )


def generate_queries(
    episodes: tuple[Episode, ...],
    grammar: GrammarPrior,
    cfg: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[tuple[Query, ...], tuple[dict[str, float], dict[str, float]]]:
    """Generate phase-shifted demand and queries interleaved after episodes."""

    del grammar  # Labels are determined only by facts instantiated in the world.

    n_episodes = len(episodes)
    alpha = np.full(len(_DEMAND_KEYS), float(cfg["query_dirichlet_alpha"]))
    weight_arrays = tuple(
        _floor_type_mixture(rng.dirichlet(alpha)) for _ in range(2)
    )
    mixtures = tuple(
        {_key_name(key): float(weight) for key, weight in zip(_DEMAND_KEYS, weights, strict=True)}
        for weights in weight_arrays
    )

    histories = _histories(episodes)
    r_candidates = _build_r_candidates(episodes)
    counts = rng.poisson(float(cfg["query_poisson_lambda"]), size=n_episodes)
    schedule = [episode_id for episode_id, count in enumerate(counts) for _ in range(int(count))]
    if (
        str(cfg["phase"]) == "eval"
        and len(schedule) < int(cfg["query_min_eval"])
    ):
        raise RuntimeError(
            "Poisson query schedule is below the configured eval minimum; "
            "use another eval seed rather than adding floor-fill emissions"
        )

    old_fraction = float(cfg["old_query_min_fraction"])
    old_age = int(np.ceil(float(cfg["old_query_age_fraction"]) * n_episodes))
    n_forced_old = ceil(old_fraction * len(schedule))
    oldest_moved = min(candidate[0] for candidate in r_candidates)
    # Use a strict age greater than 90% and ensure at least one eligible source
    # episode exists for every forced reinstatement query.
    eligible_late = np.arange(oldest_moved + old_age + 1, n_episodes)
    if not len(eligible_late):
        raise ValueError("stream is too short to satisfy the old-query quota")
    forced_steps = [int(x) for x in rng.choice(eligible_late, size=n_forced_old, replace=True)]
    rng.shuffle(schedule)
    schedule[:n_forced_old] = forced_steps
    rng.shuffle(schedule)
    forced_old_indices = set(int(x) for x in rng.choice(
        len(schedule), size=n_forced_old, replace=False
    ))
    schedule.sort()
    # Sorting invalidates random positions; mark the first required number among
    # sufficiently late slots. These remain distributed across the final 10%.
    forced_old_indices = set()
    for index, step in enumerate(schedule):
        if step >= old_age and len(forced_old_indices) < n_forced_old:
            forced_old_indices.add(index)
    if len(forced_old_indices) < n_forced_old:
        raise RuntimeError("could not allocate the required old-query quota")

    # Reserve existing Poisson emissions for conflict-query coverage.
    earliest_conflict = min(
        distinct[1][0]
        for events in histories["WORKS"].values()
        if len(distinct := _distinct_events(events)) >= 2
    )
    coverage_candidates = [
        index
        for index, step in enumerate(schedule)
        if step >= earliest_conflict and index not in forced_old_indices
    ]
    if str(cfg["phase"]) == "eval":
        unique_c_specs = sorted(
            _unique_c_query_specs(histories), key=lambda item: (item[0], item[1][0])
        )
        if len(unique_c_specs) < 300:
            raise RuntimeError(
                f"eval world supports only {len(unique_c_specs)} unique type-C queries; need 300"
            )
        unique_c_specs = unique_c_specs[:300]
        latest_required_episode = max(item[0] for item in unique_c_specs)
        forced_c_slots = [
            index
            for index, step in enumerate(schedule)
            if latest_required_episode <= step
            and index not in forced_old_indices
        ]
        if len(forced_c_slots) < len(unique_c_specs):
            raise RuntimeError(
                "not enough non-old Poisson emissions after conflict events to schedule "
                "300 unique type-C queries"
            )
        forced_c_queries = {
            index: spec
            for index, (_, spec) in zip(
                forced_c_slots[-len(unique_c_specs):], unique_c_specs, strict=True
            )
        }
    else:
        forced_c_specs = (
            (("WORKS", "CURRENT"), ("WORKS", "PREV"))
            + (("MOVED", "CURRENT"),) * 3
            + (("MOVED", "PREV"),) * 3
        )
        if len(coverage_candidates) < len(forced_c_specs):
            raise RuntimeError("not enough Poisson emissions to cover both conflict relations")
        forced_c_queries = {
            index: spec
            for index, spec in zip(
                coverage_candidates[-len(forced_c_specs):], forced_c_specs, strict=True
            )
        }

    forced_r_queries: dict[int, tuple[int, str, str, str, str, int, bool]] = {}
    forced_b_queries: dict[int, tuple[tuple[str, ...], str, tuple[int, ...], int, str]] = {}
    if str(cfg["phase"]) == "eval":
        unique_r: dict[tuple[str, str, str], tuple[int, str, str, str, str, int, bool]] = {}
        for candidate in r_candidates:
            episode_id, _target, partner, place, _answer, _competitors, _superseding = candidate
            cue = (partner, place, episodes[episode_id].time)
            unique_r.setdefault(cue, candidate)
        if len(unique_r) < 250:
            raise RuntimeError(
                f"eval world supports only {len(unique_r)} unique type-R queries; need 250"
            )
        reserved_c = set(forced_c_queries)
        available = [
            index for index, step in enumerate(schedule)
            if index not in forced_old_indices and index not in reserved_c
        ]
        target_r_count = 250
        for candidate in list(unique_r.values()):
            eligible_slot = next(
                (index for index in available if schedule[index] >= candidate[0]), None
            )
            if eligible_slot is None:
                continue
            forced_r_queries[eligible_slot] = candidate
            available.remove(eligible_slot)
            if len(forced_r_queries) == target_r_count:
                break
        if len(forced_r_queries) < target_r_count:
            raise RuntimeError(
                f"not enough Poisson emissions to schedule {target_r_count} unique type-R queries"
            )
        b_specs = sorted(_unique_b_query_specs(histories, episodes), key=lambda item: (item[0], item[1][0]))
        if len(b_specs) < 300:
            raise RuntimeError(f"eval world supports only {len(b_specs)} unique type-B queries; need 300")
        for told_episode, spec in b_specs:
            eligible_slot = next(
                (index for index in available if schedule[index] >= told_episode), None
            )
            if eligible_slot is None:
                continue
            forced_b_queries[eligible_slot] = spec
            available.remove(eligible_slot)
            if len(forced_b_queries) == 300:
                break
        if len(forced_b_queries) < 300:
            raise RuntimeError("not enough Poisson emissions to schedule 300 unique type-B queries")

    queries: list[Query] = []
    shift_at = int(float(cfg["demand_shift_fraction"]) * n_episodes)
    for schedule_index, step in enumerate(schedule):
        half = int(step >= shift_at)
        sampled_lag = _sample_lag(rng, step, cfg)
        target_step = max(0, step - sampled_lag)
        if schedule_index in forced_c_queries:
            query_type = "C"
            forced_spec = forced_c_queries[schedule_index]
            if str(cfg["phase"]) == "eval":
                built = forced_spec
            else:
                relation, operator = forced_spec
                built = _query_c(
                    step,
                    histories,
                    rng,
                    required_relation=relation,
                    required_operator=operator,
                )
        elif schedule_index in forced_r_queries:
            built = _query_r(
                step, target_step, episodes, r_candidates, rng,
                required_candidate=forced_r_queries[schedule_index],
            )
            query_type = "R"
        elif schedule_index in forced_b_queries:
            built = forced_b_queries[schedule_index]
            query_type = "B"
        elif schedule_index in forced_old_indices:
            maximum_target = step - old_age - 1
            built = _query_r(
                step,
                maximum_target,
                episodes,
                r_candidates,
                rng,
                maximum_target=maximum_target,
            )
            query_type = "R"
        else:
            key_index = int(rng.choice(len(_DEMAND_KEYS), p=weight_arrays[half]))
            query_type, fact_class = _DEMAND_KEYS[key_index]
            if query_type == "A":
                built = _query_a(
                    fact_class,
                    step,
                    target_step,
                    histories,
                    rng,
                )
            elif query_type == "B":
                built = _query_b(step, target_step, histories, episodes, rng)
            elif query_type == "C":
                built = _query_c(step, histories, rng)
            elif query_type == "E":
                built = _query_e(step, episodes, rng)
            else:
                built = _query_r(step, target_step, episodes, r_candidates, rng)
        if built is None:
            query_type = "E"
            built = _query_e(step, episodes, rng)
        if built is None:
            continue
        if len(built) == 7:
            (
                tokens, answer, evidence, target_episode_id, actual_class,
                competitor_count, from_superseding,
            ) = built
        elif len(built) == 6:
            tokens, answer, evidence, target_episode_id, actual_class, competitor_count = built
            from_superseding = None
        else:
            tokens, answer, evidence, target_episode_id, actual_class = built
            competitor_count = None
            from_superseding = None
        lag = step - target_episode_id if target_episode_id is not None else None
        queries.append(Query(
            id=len(queries),
            after_episode=step,
            query_type=query_type,
            fact_class=actual_class,
            tokens=tokens,
            answer=answer,
            evidence_episode_ids=evidence,
            target_episode_id=target_episode_id,
            lag=lag,
            demand_half=half,
            competitor_count=competitor_count,
            from_superseding=from_superseding,
        ))
    return tuple(queries), mixtures  # type: ignore[return-value]


def solve_query(
    query: Query,
    episodes: tuple[Episode, ...],
    grammar: GrammarPrior,
) -> str:
    """Symbolically solve a query using only its declared world evidence."""

    del grammar

    evidence = [episodes[index] for index in query.evidence_episode_ids]
    tokens = query.tokens
    if query.query_type == "E":
        return "ABSTAIN"
    if query.query_type == "A":
        entity, relation = tokens[1], tokens[2]
        matches = [
            fact
            for episode in evidence
            for fact in episode.facts
            if fact.subject == entity and fact.relation == relation
        ]
        if matches:
            return matches[-1].object
        return "ABSTAIN"
    if query.query_type == "B":
        subject, place, time = tokens[2], tokens[5], tokens[7]
        told = [
            fact
            for episode in evidence
            if episode.time == time
            for fact in episode.facts
            if fact.subject == subject and fact.relation == "TOLD" and fact.extra == ("AT", place)
        ]
        target = told[-1].object
        works = [
            fact
            for episode in evidence
            for fact in episode.facts
            if fact.subject == target and fact.relation == "WORKS"
        ]
        return works[-1].object
    if query.query_type == "C":
        entity, relation, operator = tokens[1], tokens[2], tokens[3]
        matches = [
            (episode.id, fact)
            for episode in sorted(evidence, key=lambda item: item.id)
            for fact in episode.facts
            if fact.subject == entity and fact.relation == relation
        ]
        distinct = _distinct_events(matches)
        return distinct[-1 if operator == "CURRENT" else -2][1].object
    if query.query_type == "R":
        partner, place, time = tokens[2], tokens[4], tokens[5]
        matches: list[str] = []
        for episode in evidence:
            if episode.time != time:
                continue
            moved = {
                fact.subject: fact.object
                for fact in episode.facts
                if fact.relation == "MOVED"
            }
            for fact in episode.facts:
                if (
                    fact.relation in {"TOLD", "MET"}
                    and fact.subject == partner
                    and fact.extra == ("AT", place)
                    and fact.object in moved
                ):
                    matches.append(moved[fact.object])
        if len(set(matches)) != 1:
            raise ValueError("R evidence does not resolve to one target city")
        return matches[0]
    raise ValueError(f"unsupported query type {query.query_type!r}")
