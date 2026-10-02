"""Deterministic generation and rendering of symbolic episode streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from . import vocab
from .grammar import GrammarPrior, make_grammar, redundant_or_uniform
from .seeds import assert_phase


@dataclass(frozen=True)
class Fact:
    relation: str
    subject: str
    object: str
    extra: tuple[str, ...] = ()
    fact_class: str = ""

    def render(self) -> tuple[str, ...]:
        return (self.subject, self.relation, self.object, *self.extra)


@dataclass(frozen=True)
class Episode:
    id: int
    time: str
    facts: tuple[Fact, ...]
    filler: tuple[str, ...]
    tokens: tuple[str, ...]

    @property
    def entities(self) -> tuple[str, ...]:
        present = {
            token
            for fact in self.facts
            for token in fact.render()
            if token.startswith("E_")
        }
        return tuple(sorted(present, key=lambda token: int(token.split("_")[1])))


@dataclass(frozen=True)
class World:
    seed: int
    phase: str
    entities: tuple[str, ...]
    grammar: GrammarPrior
    episodes: tuple[Episode, ...]
    queries: tuple[Any, ...]
    demand_mixtures: tuple[dict[str, float], dict[str, float]]
    restatement_rates: dict[str, float]
    config: dict[str, Any]


_FACT_CLASSES = {
    "WORKS": "core_relational",
    "MOVED": "causal_linked",
    "JOINED_YEAR": "temporal",
    "TOLD": "social",
    "MET": "social",
    "COLOR": "incidental",
    "OWNS": "incidental",
    "SIBLING": "interference",
}

_REQUIRED_CONFIG = {
    "grammar_seed", "typical_places_per_city", "hiring_window_years", "phase",
    "p_redundancy", "p_restate", "n_entities", "entity_vocab_size", "n_episodes",
    "min_entities_per_episode", "max_entities_per_episode",
    "min_facts_per_episode", "max_facts_per_episode", "min_filler_tokens",
    "max_filler_tokens", "n_time_buckets", "conflict_fraction",
    "interference_fraction", "fact_relation_weights", "query_dirichlet_alpha",
    "query_poisson_lambda", "query_min_eval", "demand_shift_fraction",
    "lag_min", "lag_max", "old_query_min_fraction", "old_query_age_fraction",
}


def _render_episode(
    episode_id: int,
    time: str,
    facts: tuple[Fact, ...],
    filler: tuple[str, ...],
) -> Episode:
    tokens: list[str] = ["EP", time]
    for fact in facts:
        tokens.extend(fact.render())
        tokens.append("SEP")
    tokens.extend(filler)
    tokens.append("EOS")
    return Episode(episode_id, time, facts, filler, tuple(tokens))


def _different(rng: np.random.Generator, values: tuple[str, ...], old: str) -> str:
    choices = tuple(value for value in values if value != old)
    return str(rng.choice(choices))


def _profiles(
    entities: tuple[str, ...],
    grammar: GrammarPrior,
    cfg: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[dict[str, dict[str, str]], tuple[tuple[str, str], ...]]:
    profiles: dict[str, dict[str, str]] = {}
    p = float(cfg["p_redundancy"])
    # Sampling without replacement fixes the finite-world MOVED match rate close
    # to p while retaining random assignment of which entities follow the prior.
    n_matching = round(p * len(entities))
    matching = set(str(x) for x in rng.choice(entities, size=n_matching, replace=False))
    for entity in entities:
        company = str(rng.choice(vocab.COMPANIES))
        home = grammar.company_home_city[company]
        moved = home if entity in matching else _different(rng, vocab.CITIES, home)
        year = redundant_or_uniform(
            rng, grammar.company_year_window[company], vocab.YEARS, p
        )
        profiles[entity] = {
            "WORKS": company,
            "MOVED": moved,
            "JOINED_YEAR": year,
            "COLOR": str(rng.choice(vocab.COLORS)),
            "OWNS": str(rng.choice(vocab.OBJECTS)),
        }

    n_interference_entities = int(round(float(cfg["interference_fraction"]) * len(entities)))
    n_interference_entities -= n_interference_entities % 2
    selected = [str(x) for x in rng.choice(
        entities, size=n_interference_entities, replace=False
    )]
    pairs: list[tuple[str, str]] = []
    for left, right in zip(selected[::2], selected[1::2], strict=True):
        clone = dict(profiles[left])
        clone["COLOR"] = _different(rng, vocab.COLORS, clone["COLOR"])
        profiles[right] = clone
        pairs.append((left, right))
    return profiles, tuple(pairs)


def _profile_fact(entity: str, relation: str, profiles: Mapping[str, Mapping[str, str]]) -> Fact:
    return Fact(relation, entity, profiles[entity][relation], (), _FACT_CLASSES[relation])


def _social_fact(
    relation: str,
    subject: str,
    target: str,
    profiles: Mapping[str, Mapping[str, str]],
    grammar: GrammarPrior,
    p_redundancy: float,
    rng: np.random.Generator,
) -> Fact:
    city = profiles[subject]["MOVED"]
    place = redundant_or_uniform(
        rng, grammar.city_typical_places[city], vocab.PLACES, p_redundancy
    )
    return Fact(relation, subject, target, ("AT", place), _FACT_CLASSES[relation])


def _restatement_rates(episodes: tuple[Episode, ...]) -> dict[str, float]:
    counts = {relation: [0, 0] for relation in ("WORKS", "MOVED")}
    seen: set[tuple[str, str, str]] = set()
    for episode in episodes:
        for fact in episode.facts:
            if fact.relation not in counts:
                continue
            key = (fact.subject, fact.relation, fact.object)
            counts[fact.relation][1] += 1
            counts[fact.relation][0] += int(key in seen)
            seen.add(key)
    repeated = sum(value[0] for value in counts.values())
    total = sum(value[1] for value in counts.values())
    return {
        "WORKS": counts["WORKS"][0] / counts["WORKS"][1],
        "MOVED": counts["MOVED"][0] / counts["MOVED"][1],
        "combined": repeated / total,
    }


def make_world(seed: int, cfg: Mapping[str, Any]) -> World:
    """Return the world determined solely by ``(seed, cfg)``."""

    missing = sorted(_REQUIRED_CONFIG - cfg.keys())
    if missing:
        raise ValueError(f"world config is missing keys: {', '.join(missing)}")
    phase = str(cfg["phase"])
    assert_phase(seed, phase)
    if int(cfg["entity_vocab_size"]) > len(vocab.ENTITIES):
        raise ValueError("entity_vocab_size exceeds the fixed vocabulary")
    if int(cfg["n_entities"]) > int(cfg["entity_vocab_size"]):
        raise ValueError("n_entities exceeds entity_vocab_size")

    seed_sequence = np.random.SeedSequence(seed)
    entity_rng, episode_rng, query_rng = (
        np.random.default_rng(child) for child in seed_sequence.spawn(3)
    )
    entity_pool = vocab.ENTITIES[: int(cfg["entity_vocab_size"])]
    entities = tuple(str(x) for x in entity_rng.choice(
        entity_pool, size=int(cfg["n_entities"]), replace=False
    ))
    grammar = make_grammar(
        int(cfg["grammar_seed"]),
        int(cfg["typical_places_per_city"]),
        int(cfg["hiring_window_years"]),
    )
    profiles, sibling_pairs = _profiles(entities, grammar, cfg, entity_rng)

    n_episodes = int(cfg["n_episodes"])
    n_conflicts = int(round(float(cfg["conflict_fraction"]) * len(entities)))
    conflict_entities = [str(x) for x in episode_rng.choice(
        entities, size=n_conflicts, replace=False
    )]
    conflict_episode_ids = episode_rng.choice(
        np.arange(n_episodes // 2, n_episodes), size=n_conflicts, replace=False
    )
    conflict_schedule = {
        int(episode_id): entity
        for episode_id, entity in zip(conflict_episode_ids, conflict_entities, strict=True)
    }
    initial_conflict_schedule = {
        episode_id: entity for episode_id, entity in enumerate(conflict_entities)
    }

    weights_by_relation = dict(cfg["fact_relation_weights"])
    relations = tuple(weights_by_relation)
    relation_weights = np.asarray([float(weights_by_relation[r]) for r in relations])
    relation_weights /= relation_weights.sum()
    episodes: list[Episode] = []
    seen_relational_facts: set[tuple[str, str, str]] = set()
    restatement_counts = {relation: [0, 0] for relation in ("WORKS", "MOVED")}
    p_restate = float(cfg["p_restate"])
    if not 0.0 <= p_restate <= 1.0:
        raise ValueError("p_restate must be in [0, 1]")
    for episode_id in range(n_episodes):
        n_episode_entities = int(episode_rng.integers(
            int(cfg["min_entities_per_episode"]),
            int(cfg["max_entities_per_episode"]) + 1,
        ))
        forced_pair = sibling_pairs[episode_id] if episode_id < len(sibling_pairs) else None
        initial_conflict_entity = initial_conflict_schedule.get(episode_id)
        forced_conflict = conflict_schedule.get(episode_id)
        required = list(forced_pair or ())
        if initial_conflict_entity is not None and initial_conflict_entity not in required:
            required.append(initial_conflict_entity)
        if forced_conflict is not None and forced_conflict not in required:
            required.append(forced_conflict)
        if initial_conflict_entity is not None:
            n_episode_entities = max(n_episode_entities, 2)
        n_episode_entities = max(n_episode_entities, len(required))
        # Two entities provide enough non-relational fact keys to resample a
        # repeated WORKS/MOVED candidate without violating the 4-fact minimum.
        n_episode_entities = max(n_episode_entities, 2)
        remainder = tuple(entity for entity in entities if entity not in required)
        chosen = required + [str(x) for x in episode_rng.choice(
            remainder, size=n_episode_entities - len(required), replace=False
        )]

        maximum_facts = int(cfg["max_facts_per_episode"])
        if len(chosen) == 1:
            maximum_facts = min(maximum_facts, 4)
        n_facts = int(episode_rng.integers(
            int(cfg["min_facts_per_episode"]), maximum_facts + 1
        ))
        facts: list[Fact] = []
        used_keys: set[tuple[str, str]] = set()

        def add_fact(fact: Fact) -> None:
            key = (fact.subject, fact.relation)
            if key in used_keys:
                raise RuntimeError(f"duplicate episode fact key {key}")
            used_keys.add(key)
            facts.append(fact)
            if fact.relation in {"WORKS", "MOVED"}:
                value_key = (fact.subject, fact.relation, fact.object)
                restatement_counts[fact.relation][1] += 1
                restatement_counts[fact.relation][0] += int(
                    value_key in seen_relational_facts
                )
                seen_relational_facts.add(value_key)

        def admit_restatement(*candidate_facts: Fact) -> bool:
            additions = {relation: [0, 0] for relation in ("WORKS", "MOVED")}
            for fact in candidate_facts:
                if fact.relation not in additions:
                    continue
                additions[fact.relation][1] += 1
                additions[fact.relation][0] += int(
                    (fact.subject, fact.relation, fact.object)
                    in seen_relational_facts
                )
            return all(
                (restatement_counts[relation][0] + repeated)
                / (restatement_counts[relation][1] + total)
                <= p_restate
                for relation, (repeated, total) in additions.items()
                if total
            )

        if forced_pair is not None:
            add_fact(Fact(
                "SIBLING", forced_pair[0], forced_pair[1], (), _FACT_CLASSES["SIBLING"]
            ))
        if initial_conflict_entity is not None:
            add_fact(_profile_fact(initial_conflict_entity, "WORKS", profiles))
            add_fact(_profile_fact(initial_conflict_entity, "MOVED", profiles))
            partner = str(episode_rng.choice(tuple(
                entity for entity in chosen if entity != initial_conflict_entity
            )))
            relation = "TOLD" if episode_id % 2 == 0 else "MET"
            add_fact(_social_fact(
                relation,
                partner,
                initial_conflict_entity,
                profiles,
                grammar,
                float(cfg["p_redundancy"]),
                episode_rng,
            ))
        if forced_conflict is not None:
            old_company = profiles[forced_conflict]["WORKS"]
            old_city = profiles[forced_conflict]["MOVED"]
            eligible_companies = tuple(
                company
                for company in vocab.COMPANIES
                if company != old_company
                and grammar.company_home_city[company] != old_city
            )
            new_company = str(episode_rng.choice(eligible_companies))
            profiles[forced_conflict]["WORKS"] = new_company
            home = grammar.company_home_city[new_company]
            if episode_rng.random() < float(cfg["p_redundancy"]):
                new_city = home
            else:
                non_prior_cities = tuple(
                    city for city in vocab.CITIES if city not in {home, old_city}
                )
                new_city = str(episode_rng.choice(non_prior_cities))
            profiles[forced_conflict]["MOVED"] = new_city
            add_fact(_profile_fact(forced_conflict, "WORKS", profiles))
            add_fact(_profile_fact(forced_conflict, "MOVED", profiles))

        attempts = 0
        while len(facts) < n_facts and attempts < 500:
            attempts += 1
            relation = str(episode_rng.choice(relations, p=relation_weights))
            subject = str(episode_rng.choice(chosen))
            if (subject, relation) in used_keys:
                continue
            if relation == "MOVED":
                # Causal-linked MOVED facts expose the governing company in the
                # same episode, immediately before the move.
                if (
                    len(facts) + 2 > n_facts
                    or (subject, "WORKS") in used_keys
                ):
                    continue
                work_fact = _profile_fact(subject, "WORKS", profiles)
                moved_fact = _profile_fact(subject, "MOVED", profiles)
                if not admit_restatement(work_fact, moved_fact):
                    continue
                add_fact(work_fact)
                add_fact(moved_fact)
                continue
            if relation in {"TOLD", "MET"}:
                if len(chosen) < 2:
                    continue
                target = str(episode_rng.choice(tuple(e for e in chosen if e != subject)))
                fact = _social_fact(
                    relation, subject, target, profiles, grammar,
                    float(cfg["p_redundancy"]), episode_rng,
                )
            else:
                fact = _profile_fact(subject, relation, profiles)
            if not admit_restatement(fact):
                continue
            add_fact(fact)
        if len(facts) != n_facts:
            raise RuntimeError(
                f"could not generate {n_facts} unique facts for episode {episode_id}"
            )

        filler_count = int(episode_rng.integers(
            int(cfg["min_filler_tokens"]), int(cfg["max_filler_tokens"]) + 1
        ))
        filler = tuple(str(x) for x in episode_rng.choice(
            vocab.FILLER, size=filler_count, replace=True
        ))
        time = vocab.TIMES[episode_id * int(cfg["n_time_buckets"]) // n_episodes]
        episodes.append(_render_episode(episode_id, time, tuple(facts), filler))

    from .queries import generate_queries

    frozen_episodes = tuple(episodes)
    queries, mixtures = generate_queries(frozen_episodes, grammar, cfg, query_rng)
    return World(
        seed=seed,
        phase=phase,
        entities=entities,
        grammar=grammar,
        episodes=frozen_episodes,
        queries=queries,
        demand_mixtures=mixtures,
        restatement_rates=_restatement_rates(frozen_episodes),
        config=dict(cfg),
    )
