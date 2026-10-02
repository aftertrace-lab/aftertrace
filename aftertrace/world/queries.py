"""Interleaved query generation, provenance, and symbolic solving."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from . import vocab

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
    ("D", "core_relational"),
    ("E", "unavailable"),
    ("R", "reinstatement"),
)


def _key_name(key: tuple[str, str]) -> str:
    return f"{key[0]}:{key[1]}"


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
) -> tuple[tuple[str, ...], str, tuple[int, ...], int, str] | None:
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
    tokens = ("Q", "HOP2", told.subject, "TOLD", "AT", place, "WORKS", "A")
    return tokens, work.object, evidence, told_episode, "social"


def _distinct_events(events: list[tuple[int, Fact]]) -> list[tuple[int, Fact]]:
    distinct: list[tuple[int, Fact]] = []
    for event in events:
        if not distinct or event[1].object != distinct[-1][1].object:
            distinct.append(event)
    return distinct


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


def _query_d(
    step: int,
    histories: Mapping[str, Mapping[str, list[tuple[int, Fact]]]],
    rng: np.random.Generator,
) -> tuple[tuple[str, ...], str, tuple[int, ...], int | None, str] | None:
    current: dict[str, tuple[int, Fact]] = {}
    all_evidence: set[int] = set()
    for entity, events in histories.get("WORKS", {}).items():
        prior = _at_or_before(events, step)
        if prior:
            current[entity] = prior[-1]
            all_evidence.update(episode_id for episode_id, _ in prior)
    counts = {
        company: sum(fact.object == company for _, fact in current.values())
        for company in vocab.COMPANIES
    }
    eligible = tuple(company for company, count in counts.items() if count <= 15)
    if not eligible:
        return None
    company = str(rng.choice(eligible))
    answer = f"N_{counts[company]}"
    evidence = tuple(sorted(all_evidence))
    target = evidence[0] if evidence else None
    return (("Q", "COUNT", "WORKS", company, "A"), answer, evidence, target, "core_relational")


def _query_e(step: int, episodes: tuple[Episode, ...], rng: np.random.Generator):
    entities = sorted({entity for episode in episodes[: step + 1] for entity in episode.entities})
    if not entities:
        return None
    entity = str(rng.choice(entities))
    return (("Q", entity, "STUDIED", "A"), "ABSTAIN", (), None, "unavailable")


def _build_r_candidates(
    episodes: tuple[Episode, ...],
) -> list[tuple[int, str, str, str, str, int]]:
    """Return unique-cue R targets as (episode, target, partner, place, answer, competitors)."""

    raw_candidates: list[tuple[int, str, str, str, str]] = []
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
                raw_candidates.append((episode.id, target, partner, place, moved[target]))

    targets_by_cue: dict[tuple[str, str, str], set[tuple[str, str]]] = {}
    for episode_id, target, partner, place, answer in raw_candidates:
        cue = (partner, place, episodes[episode_id].time)
        targets_by_cue.setdefault(cue, set()).add((target, answer))

    candidates: list[tuple[int, str, str, str, str, int]] = []
    for episode_id, target, partner, place, answer in raw_candidates:
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
            (episode_id, target, partner, place, answer, len(competitor_episodes))
        )
    return candidates


def _query_r(
    step: int,
    target_step: int,
    episodes: tuple[Episode, ...],
    candidates: list[tuple[int, str, str, str, str, int]],
    rng: np.random.Generator,
    maximum_target: int | None = None,
):
    eligible = [
        candidate
        for candidate in candidates
        if candidate[0] <= step
        and (maximum_target is None or candidate[0] <= maximum_target)
    ]
    if not eligible:
        return None
    distances = np.asarray([abs(candidate[0] - target_step) for candidate in eligible])
    chosen = eligible[int(rng.choice(np.flatnonzero(distances == distances.min())))]
    episode_id, target, partner, place, answer, competitor_count = chosen
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
    )


def generate_queries(
    episodes: tuple[Episode, ...],
    grammar: GrammarPrior,
    cfg: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[tuple[Query, ...], tuple[dict[str, float], dict[str, float]]]:
    """Generate phase-shifted demand and queries interleaved after episodes."""

    del grammar  # Query answers are ground truth; grammar is used by context solvers.
    n_episodes = len(episodes)
    alpha = np.full(len(_DEMAND_KEYS), float(cfg["query_dirichlet_alpha"]))
    weight_arrays = (rng.dirichlet(alpha), rng.dirichlet(alpha))
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

    # Reserve existing Poisson emissions (without adding any) so conflict
    # queries cover both relations and both MOVED operators are inspectable.
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

    queries: list[Query] = []
    shift_at = int(float(cfg["demand_shift_fraction"]) * n_episodes)
    for schedule_index, step in enumerate(schedule):
        half = int(step >= shift_at)
        sampled_lag = _sample_lag(rng, step, cfg)
        target_step = max(0, step - sampled_lag)
        if schedule_index in forced_c_queries:
            query_type = "C"
            relation, operator = forced_c_queries[schedule_index]
            built = _query_c(
                step,
                histories,
                rng,
                required_relation=relation,
                required_operator=operator,
            )
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
                built = _query_a(fact_class, step, target_step, histories, rng)
            elif query_type == "B":
                built = _query_b(step, target_step, histories, rng)
            elif query_type == "C":
                built = _query_c(step, histories, rng)
            elif query_type == "D":
                built = _query_d(step, histories, rng)
            elif query_type == "E":
                built = _query_e(step, episodes, rng)
            else:
                built = _query_r(step, target_step, episodes, r_candidates, rng)
        if built is None:
            query_type = "E"
            built = _query_e(step, episodes, rng)
        if built is None:
            continue
        if len(built) == 6:
            tokens, answer, evidence, target_episode_id, actual_class, competitor_count = built
        else:
            tokens, answer, evidence, target_episode_id, actual_class = built
            competitor_count = None
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
        ))
    return tuple(queries), mixtures  # type: ignore[return-value]


def solve_query(
    query: Query,
    episodes: tuple[Episode, ...],
    grammar: GrammarPrior,
) -> str:
    """Symbolically solve a query using only its declared evidence and prior."""

    del grammar  # Current Day-1 queries are all direct-evidence or unavailable cases.
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
        return matches[-1].object
    if query.query_type == "B":
        subject, place = tokens[2], tokens[5]
        told = [
            fact
            for episode in evidence
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
    if query.query_type == "D":
        company = tokens[3]
        state: dict[str, str] = {}
        for episode in sorted(evidence, key=lambda item: item.id):
            for fact in episode.facts:
                if fact.relation == "WORKS":
                    state[fact.subject] = fact.object
        return f"N_{sum(value == company for value in state.values())}"
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
