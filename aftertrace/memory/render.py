"""Deterministic token renderings for every Stage-0 memory format."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil

from aftertrace.world import vocab
from aftertrace.world.generator import Episode, Fact
from aftertrace.world.grammar import GrammarPrior


def _render_facts(prefix: Sequence[str], facts: Sequence[Fact]) -> tuple[str, ...]:
    tokens = list(prefix)
    for fact in facts:
        tokens.extend(fact.render())
        tokens.append("SEP")
    tokens.append("EOS")
    return tuple(tokens)


def render_l4(episode: Episode) -> tuple[str, ...]:
    """Render the complete episode, including filler."""

    return episode.tokens


def render_l3(episode: Episode) -> tuple[str, ...]:
    """Render every fact but omit filler tokens."""

    return _render_facts(("EP", episode.time), episode.facts)


def render_l2(
    episode: Episode,
    utility_weights: Mapping[str, float] | None = None,
    threshold: float | None = None,
) -> tuple[str, ...]:
    """Placeholder L2: retain the first ceil(0.6*n) canonical facts."""

    del utility_weights, threshold  # Reserved for calibrated utility selection.
    selected = episode.facts[: ceil(0.6 * len(episode.facts))]
    return _render_facts(("EPP", episode.time), selected)


def render_l1(episode: Episode, anchor_index: int = 0) -> tuple[str, ...]:
    """Render time and relations grouped by subject, omitting all values."""

    del anchor_index  # Retained as a compatibility argument; gist is lossless for keys.
    groups: dict[str, list[Fact]] = {}
    for fact in episode.facts:
        groups.setdefault(fact.subject, []).append(fact)
    tokens: list[str] = ["GIST", episode.time]
    for group_index, (subject, facts) in enumerate(groups.items()):
        if group_index:
            tokens.append("SEP")
        tokens.append(subject)
        for fact in facts:
            tokens.append(fact.relation)
            if (
                fact.relation in {"SIBLING", "TOLD", "MET"}
                and fact.object.startswith("E_")
            ):
                tokens.append(fact.object)
    tokens.append("EOS")
    return tuple(tokens)


def gist_entity_relation_pairs(tokens: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Decode all grouped (entity, relation) keys retained by an L1 gist."""

    if len(tokens) < 3 or tokens[0] != "GIST":
        raise ValueError("not a GIST rendering")
    body = list(tokens[2:-1])
    segments: list[list[str]] = [[]]
    for token in body:
        if token == "SEP":
            segments.append([])
        else:
            segments[-1].append(token)
    segments = [segment for segment in segments if segment]
    pairs: list[tuple[str, str]] = []
    for segment in segments:
        if not segment:
            continue
        subject, *entries = segment
        if not subject.startswith("E_"):
            raise ValueError(f"invalid GIST subject segment: {segment!r}")
        index = 0
        while index < len(entries):
            relation = entries[index]
            if relation not in vocab.RELATIONS:
                raise ValueError(f"invalid GIST relation segment: {segment!r}")
            pairs.append((subject, relation))
            index += 1
            if (
                relation in {"SIBLING", "TOLD", "MET"}
                and index < len(entries)
                and entries[index].startswith("E_")
            ):
                index += 1  # Retained partner entity, not a relation key.
    return tuple(pairs)


def render_level(
    episode: Episode,
    level: int,
    *,
    utility_weights: Mapping[str, float] | None = None,
    threshold: float | None = None,
    anchor_index: int = 0,
) -> tuple[str, ...]:
    """Render a quantized fidelity level L0--L4."""

    if level == 4:
        return render_l4(episode)
    if level == 3:
        return render_l3(episode)
    if level == 2:
        return render_l2(episode, utility_weights, threshold)
    if level == 1:
        return render_l1(episode, anchor_index)
    if level == 0:
        return ()
    raise ValueError("fidelity level must be in [0, 4]")


def quantize_fidelity(fidelity: float) -> int:
    """Map r in [0, 1] to L0--L4 using five equal-width bins."""

    # DECISION: Use equal-width bins because SPEC defines levels but no cutoffs.
    if not 0.0 <= fidelity <= 1.0:
        raise ValueError("fidelity must be in [0, 1]")
    return min(4, int(fidelity * 5))


def predicted_fact_indices(
    episode: Episode,
    grammar: GrammarPrior,
) -> tuple[int, ...]:
    """Return full facts predicted by other facts and the shared grammar."""

    works = {
        fact.subject: fact.object for fact in episode.facts if fact.relation == "WORKS"
    }
    predicted: list[int] = []
    for index, fact in enumerate(episode.facts):
        company = works.get(fact.subject)
        if (
            fact.relation == "MOVED"
            and company is not None
            and grammar.predicts_moved(company, fact.object)
        ):
            predicted.append(index)
        elif (
            fact.relation == "JOINED_YEAR"
            and company is not None
            and grammar.predicts_joined_year(company, fact.object)
        ):
            predicted.append(index)
    return tuple(predicted)


def predicted_at_indices(
    episode: Episode,
    grammar: GrammarPrior,
) -> tuple[int, ...]:
    """Return social facts whose AT attribute follows the city-place prior."""

    cities = {
        fact.subject: fact.object for fact in episode.facts if fact.relation == "MOVED"
    }
    return tuple(
        index
        for index, fact in enumerate(episode.facts)
        if fact.relation in {"TOLD", "MET"}
        and fact.extra[:1] == ("AT",)
        and fact.subject in cities
        and grammar.predicts_place(cities[fact.subject], fact.extra[1])
    )


def render_residual(episode: Episode, grammar: GrammarPrior) -> tuple[str, ...]:
    """Replace prior-predicted values with PRED while retaining fact presence."""

    # DECISION: PRED is lossless because grammar predictions are canonical; the
    # relation key remains explicit and incidental facts are never transformed.
    predicted_facts = set(predicted_fact_indices(episode, grammar))
    predicted_at = set(predicted_at_indices(episode, grammar))
    residual: list[Fact] = []
    for index, fact in enumerate(episode.facts):
        if index in predicted_facts:
            fact = Fact(
                relation=fact.relation,
                subject=fact.subject,
                object="PRED",
                extra=(),
                fact_class=fact.fact_class,
            )
        elif index in predicted_at:
            fact = Fact(
                relation=fact.relation,
                subject=fact.subject,
                object=fact.object,
                extra=("AT", "PRED"),
                fact_class=fact.fact_class,
            )
        residual.append(fact)
    return _render_facts(("RES", episode.time), residual)


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


def decode_residual(tokens: Sequence[str], grammar: GrammarPrior) -> tuple[Fact, ...]:
    """Invert a residual rendering using its explicit keys and the grammar."""

    if len(tokens) < 3 or tokens[0] != "RES" or tokens[-1] != "EOS":
        raise ValueError("not a complete RES rendering")
    segments: list[list[str]] = [[]]
    for token in tokens[2:-1]:
        if token == "SEP":
            if segments[-1]:
                segments.append([])
        else:
            segments[-1].append(token)
    segments = [segment for segment in segments if segment]
    parsed: list[Fact] = []
    for segment in segments:
        if len(segment) not in {3, 5}:
            raise ValueError(f"invalid residual fact: {segment!r}")
        subject, relation, object_ = segment[:3]
        extra = tuple(segment[3:])
        parsed.append(Fact(relation, subject, object_, extra, _FACT_CLASSES[relation]))

    works = {fact.subject: fact.object for fact in parsed if fact.relation == "WORKS"}
    moved = {
        fact.subject: (
            grammar.company_home_city[works[fact.subject]]
            if fact.object == "PRED"
            else fact.object
        )
        for fact in parsed
        if fact.relation == "MOVED"
    }
    decoded: list[Fact] = []
    for fact in parsed:
        object_ = fact.object
        extra = fact.extra
        if object_ == "PRED":
            company = works[fact.subject]
            if fact.relation == "MOVED":
                object_ = grammar.company_home_city[company]
            elif fact.relation == "JOINED_YEAR":
                object_ = grammar.company_year_window[company][0]
            else:
                raise ValueError(f"PRED unsupported for {fact.relation}")
        if extra == ("AT", "PRED"):
            extra = ("AT", grammar.city_typical_places[moved[fact.subject]][0])
        decoded.append(Fact(fact.relation, fact.subject, object_, extra, fact.fact_class))
    return tuple(decoded)


def render_summaries(episodes: Sequence[Episode]) -> tuple[tuple[str, ...], ...]:
    """Render deterministic per-entity lines containing the latest relation values."""

    latest: dict[tuple[str, str], Fact] = {}
    for episode in episodes:
        for fact in episode.facts:
            latest[(fact.subject, fact.relation)] = fact
    entities = sorted(
        {entity for entity, _ in latest}, key=lambda token: int(token.split("_")[1])
    )
    lines: list[tuple[str, ...]] = []
    for entity in entities:
        facts = sorted(
            (fact for (subject, _), fact in latest.items() if subject == entity),
            key=lambda fact: fact.relation,
        )
        tokens: list[str] = ["SUM", entity]
        for fact in facts:
            tokens.extend((fact.relation, fact.object, *fact.extra, "SEP"))
        tokens.append("EOS")
        lines.append(tuple(tokens))
    return tuple(lines)
