"""Symbolic optimal reader for Stage-0 memory-condition frontiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from aftertrace.memory.base import ItemView, MemoryItem
from aftertrace.memory.render import decode_residual
from aftertrace.world import vocab
from aftertrace.world.grammar import GrammarPrior


@dataclass(frozen=True)
class OraclePrediction:
    answer: str
    confidence: float
    candidate_count: int = 0


@dataclass(frozen=True)
class FactEvidence:
    subject: str
    relation: str
    object: str | None
    extra: tuple[str, ...]
    episode_id: int | None
    time: str | None
    format: str
    ordinal: int


def _unpack_item(item: Any) -> tuple[tuple[str, ...], int | None]:
    if isinstance(item, ItemView):
        return item.tokens, item.item.episode_id
    if isinstance(item, MemoryItem):
        return item.tokens, item.episode_id
    tokens = tuple(item)
    episode_id = getattr(item, "episode_id", None)
    return tokens, episode_id


def _parse_regular(
    tokens: Sequence[str], episode_id: int | None, time: str | None, fmt: str
) -> list[FactEvidence]:
    facts: list[FactEvidence] = []
    index = 2
    while index < len(tokens) and tokens[index] != "EOS":
        if tokens[index] == "SEP":
            index += 1
            continue
        if index + 2 >= len(tokens):
            break
        subject, relation, value = tokens[index : index + 3]
        if not subject.startswith("E_") or relation not in vocab.RELATIONS:
            index += 1
            continue
        index += 3
        extra: tuple[str, ...] = ()
        if relation in {"TOLD", "MET"} and index + 1 < len(tokens) and tokens[index] == "AT":
            extra = ("AT", tokens[index + 1])
            index += 2
        facts.append(
            FactEvidence(subject, relation, value, extra, episode_id, time, fmt, len(facts))
        )
    return facts


def _parse_gist(
    tokens: Sequence[str], episode_id: int | None, time: str | None
) -> list[FactEvidence]:
    facts: list[FactEvidence] = []
    groups: list[list[str]] = [[]]
    for token in tokens[2:-1]:
        if token == "SEP":
            groups.append([])
        else:
            groups[-1].append(token)
    for group in groups:
        if not group or not group[0].startswith("E_"):
            continue
        subject, *entries = group
        index = 0
        while index < len(entries):
            relation = entries[index]
            index += 1
            if relation not in vocab.RELATIONS:
                continue
            object_: str | None = None
            if relation in {"SIBLING", "TOLD", "MET"} and index < len(entries) and entries[index].startswith("E_"):
                object_ = entries[index]
                index += 1
            facts.append(
                FactEvidence(subject, relation, object_, (), episode_id, time, "GIST", len(facts))
            )
    return facts


def parse_context_items(
    items: Iterable[Any], grammar: GrammarPrior
) -> tuple[FactEvidence, ...]:
    """Parse every supported representation into an item-aware fact table."""

    result: list[FactEvidence] = []
    for item in items:
        tokens, episode_id = _unpack_item(item)
        if not tokens:
            continue
        fmt = tokens[0]
        time = tokens[1] if len(tokens) > 1 and tokens[1].startswith("T_") else None
        if fmt == "RES":
            try:
                decoded = decode_residual(tokens, grammar)
            except (KeyError, ValueError, IndexError):
                continue
            result.extend(
                FactEvidence(
                    fact.subject, fact.relation, fact.object, fact.extra,
                    episode_id, time, "RES", len(result),
                )
                for fact in decoded
            )
        elif fmt in {"EP", "EPP"}:
            result.extend(_parse_regular(tokens, episode_id, time, fmt))
        elif fmt == "GIST":
            result.extend(_parse_gist(tokens, episode_id, time))
        elif fmt == "SUM":
            index = 2
            while index < len(tokens) and tokens[index] != "EOS":
                if tokens[index] == "SEP":
                    index += 1
                    continue
                if index + 1 >= len(tokens):
                    break
                relation, value = tokens[index:index + 2]
                if relation in vocab.RELATIONS:
                    extra: tuple[str, ...] = ()
                    if relation in {"TOLD", "MET"} and index + 2 < len(tokens) and tokens[index + 2] == "AT":
                        if index + 3 < len(tokens):
                            extra = ("AT", tokens[index + 3])
                            index += 2
                    result.append(FactEvidence(
                        tokens[1], relation, value, extra, episode_id, None, "SUM", len(result)
                    ))
                index += 2
    return tuple(result)


class OracleReader:
    """Rule-based reader implementing the harness prediction interface.

    predict accepts selected context items, a world Query, the shared grammar
    prior, and p_redundancy, and returns an answer plus confidence. Confidence
    is the probability assigned to the best non-abstain answer; type-E uses
    confidence one to record certainty in its known abstention.
    """

    def predict(
        self,
        context_items: Sequence[Any],
        query: Any,
        grammar: GrammarPrior,
        p_redundancy: float,
    ) -> OraclePrediction:
        facts = parse_context_items(context_items, grammar)
        query_type = query.query_type
        if query_type == "E":
            return OraclePrediction("ABSTAIN", 1.0)
        if query_type == "R":
            return self._predict_r(facts, query.tokens)
        if query_type == "B":
            return self._predict_b(facts, query.tokens)
        if query_type == "C":
            entity, relation, operator = query.tokens[1:4]
            matches = self._ordered_matches(facts, entity, relation)
            values: list[FactEvidence] = []
            for fact in matches:
                if not values or values[-1].object != fact.object:
                    values.append(fact)
            if len(values) >= (2 if operator == "PREV" else 1):
                chosen = values[-2] if operator == "PREV" else values[-1]
                return OraclePrediction(str(chosen.object), 1.0, len(values))
            return OraclePrediction("ABSTAIN", 0.0, len(values))

        if query_type == "A":
            entity, relation = query.tokens[1:3]
            matches = self._ordered_matches(facts, entity, relation)
            exact = next((fact for fact in reversed(matches) if fact.object is not None), None)
            if exact is not None:
                return OraclePrediction(str(exact.object), 1.0, len(matches))
            prediction = self._predict_from_work(
                entity, relation, facts, grammar, p_redundancy
            )
            if prediction is not None:
                value, confidence = prediction
                if confidence > 0.5:
                    return OraclePrediction(value, confidence, 1)
                return OraclePrediction("ABSTAIN", confidence, 1)
        return OraclePrediction("ABSTAIN", 0.0)

    __call__ = predict

    @staticmethod
    def _ordered_matches(
        facts: Sequence[FactEvidence], entity: str, relation: str
    ) -> list[FactEvidence]:
        matches = [fact for fact in facts if fact.subject == entity and fact.relation == relation]
        return sorted(
            matches,
            key=lambda fact: (
                fact.episode_id if fact.episode_id is not None else -1,
                fact.ordinal,
            ),
        )

    @staticmethod
    def _predict_from_work(
        entity: str,
        relation: str,
        facts: Sequence[FactEvidence],
        grammar: GrammarPrior,
        p_redundancy: float,
    ) -> tuple[str, float] | None:
        works = [
            fact.object for fact in facts
            if fact.subject == entity and fact.relation == "WORKS" and fact.object is not None
        ]
        if not works:
            return None
        company = works[-1]
        if relation == "MOVED":
            return grammar.company_home_city[company], p_redundancy
        if relation == "JOINED_YEAR":
            return grammar.company_year_window[company][0], p_redundancy
        return None

    @classmethod
    def _predict_b(cls, facts: Sequence[FactEvidence], tokens: Sequence[str]) -> OraclePrediction:
        subject, place, time = tokens[2], tokens[5], tokens[7]
        told = [
            fact for fact in facts
            if fact.subject == subject and fact.relation == "TOLD"
            and fact.object is not None and fact.extra == ("AT", place)
            and fact.time == time
        ]
        if not told:
            return OraclePrediction("ABSTAIN", 0.0)
        selected_told = max(
            told,
            key=lambda fact: fact.episode_id if fact.episode_id is not None else -1,
        )
        works = [
            fact for fact in facts
            if fact.subject == selected_told.object and fact.relation == "WORKS"
            and fact.object is not None
            and (
                fact.episode_id is None or selected_told.episode_id is None
                or fact.episode_id < selected_told.episode_id
            )
        ]
        if not works:
            return OraclePrediction("ABSTAIN", 0.0)
        answer = max(
            works,
            key=lambda fact: fact.episode_id if fact.episode_id is not None else -1,
        ).object
        return OraclePrediction(str(answer), 1.0, len(works))

    @staticmethod
    def _predict_r(
        facts: Sequence[FactEvidence], tokens: Sequence[str]
    ) -> OraclePrediction:
        partner, place, time = tokens[2], tokens[4], tokens[5]
        moved_by_episode: dict[int | None, dict[str, FactEvidence]] = {}
        for fact in facts:
            if fact.relation == "MOVED" and fact.object is not None:
                moved_by_episode.setdefault(fact.episode_id, {})[fact.subject] = fact

        candidates: list[tuple[int, str]] = []
        for interaction in facts:
            if interaction.relation not in {"TOLD", "MET"} or interaction.object is None:
                continue
            moved = moved_by_episode.get(interaction.episode_id, {}).get(interaction.object)
            if moved is None:
                continue
            matched = sum((
                interaction.subject == partner,
                interaction.extra == ("AT", place),
                interaction.time == time,
            ))
            if matched:
                candidates.append((matched, str(moved.object)))
        if not candidates:
            return OraclePrediction("ABSTAIN", 0.0)
        best_match = max(score for score, _ in candidates)
        best = [answer for score, answer in candidates if score == best_match]
        counts: dict[str, int] = {}
        for answer in best:
            counts[answer] = counts.get(answer, 0) + 1
        answer, count = max(counts.items(), key=lambda pair: pair[1])
        confidence = (count / len(best)) * (best_match / 3.0)
        if confidence <= 0.5:
            return OraclePrediction("ABSTAIN", confidence, len(best))
        return OraclePrediction(answer, confidence, len(best))
