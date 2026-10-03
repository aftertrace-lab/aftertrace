"""On-the-fly reader sequences with mixed memory representations."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from aftertrace.memory.render import (
    decode_residual,
    render_l1,
    render_l2,
    render_l3,
    render_l4,
    render_residual,
    render_summaries,
)
from aftertrace.world import vocab
from aftertrace.world.generator import Episode, Fact, World
from aftertrace.world.queries import Query


REPRESENTATION_FORMATS = ("l4", "l3", "l2", "l1", "residual", "summary")


@dataclass(frozen=True)
class TrainingExample:
    input_ids: torch.Tensor
    labels: torch.Tensor
    tokens: tuple[str, ...]
    representation: str
    query_type: str
    answer: str
    world_answer: str


@dataclass(frozen=True)
class _ContextItem:
    tokens: tuple[str, ...]
    episode: Episode
    representation: str
    is_evidence: bool


class ReaderDataGenerator:
    """Generate deterministic streams of independently remapped sequences."""

    def __init__(
        self,
        worlds: Sequence[World],
        config: Mapping[str, Any],
        seed: int,
    ) -> None:
        if not worlds:
            raise ValueError("at least one reader-training world is required")
        if any(world.phase != "reader_train" for world in worlds):
            raise ValueError("reader data may only use reader_train worlds")
        self.worlds = tuple(worlds)
        self.config = dict(config)
        self.context_length = int(config["context_length"])
        self.rng = np.random.default_rng(seed)
        self.summary_cache = {
            (world.seed, endpoint): render_summaries(world.episodes[: endpoint + 1])
            for world in worlds
            for endpoint in range(
                int(config["summary_period"]) - 1,
                len(world.episodes),
                int(config["summary_period"]),
            )
        }

    def _render_episode(
        self,
        world: World,
        episode: Episode,
        representation: str,
    ) -> tuple[str, ...]:
        if representation == "l4":
            return render_l4(episode)
        if representation == "l3":
            return render_l3(episode)
        if representation == "l2":
            return render_l2(
                episode,
                self.config["l2_utility_weights"],
                float(self.config["l2_threshold"]),
            )
        if representation == "l1":
            return render_l1(episode)
        if representation == "residual":
            return render_residual(episode, world.grammar)
        raise ValueError(f"unsupported episode representation {representation!r}")

    def _summary_item(
        self,
        world: World,
        query: Query,
        episode: Episode,
    ) -> tuple[str, ...]:
        period = int(self.config["summary_period"])
        endpoint = min(
            ((query.after_episode + 1) // period) * period - 1,
            len(world.episodes) - 1,
        )
        if endpoint < period - 1:
            lines = render_summaries(world.episodes[: query.after_episode + 1])
        else:
            lines = self.summary_cache[(world.seed, endpoint)]
        episode_entities = set(episode.entities)
        matching = [line for line in lines if len(line) > 1 and line[1] in episode_entities]
        return matching[0] if matching else tuple(lines[0])

    def _render_item(
        self,
        world: World,
        query: Query,
        episode: Episode,
        representation: str,
    ) -> tuple[str, ...]:
        if representation == "summary":
            return self._summary_item(world, query, episode)
        return self._render_episode(world, episode, representation)

    def _choose_query(self, world: World, force_abstain: bool) -> Query:
        candidates = [
            query
            for query in world.queries
            if (query.answer == "ABSTAIN") == force_abstain
        ]
        if not candidates:
            raise RuntimeError("world does not contain the requested query target")
        return candidates[int(self.rng.integers(len(candidates)))]

    def _facts_visible_or_decodable(
        self, world: World, item: _ContextItem
    ) -> tuple[Fact, ...]:
        if item.representation in {"l4", "l3"}:
            return item.episode.facts
        if item.representation == "l2":
            return item.episode.facts[: ceil(0.6 * len(item.episode.facts))]
        if item.representation == "residual":
            return decode_residual(item.tokens, world.grammar)
        if item.representation == "summary":
            subject = item.tokens[1]
            facts: list[Fact] = []
            segment: list[str] = []
            for token in item.tokens[2:]:
                if token in {"SEP", "EOS"}:
                    if segment:
                        relation, object_, *extra = segment
                        facts.append(Fact(relation, subject, object_, tuple(extra), "summary"))
                        segment = []
                else:
                    segment.append(token)
            return tuple(facts)
        # Gists expose one full anchor value; summaries expose literal values.
        return tuple(
            fact
            for fact in item.episode.facts
            if fact.object in item.tokens and fact.relation in item.tokens
        )

    def _target_for_context(
        self, world: World, query: Query, included: Sequence[_ContextItem]
    ) -> str:
        """Choose a label from world truth and the retained predictive evidence."""

        if query.answer == "ABSTAIN":
            return "ABSTAIN"
        evidence = [item for item in included if item.is_evidence]
        facts = [
            fact
            for item in evidence
            for fact in self._facts_visible_or_decodable(world, item)
        ]
        if query.answer in {fact.object for fact in facts}:
            return query.answer
        if query.query_type == "A" and query.tokens[2] in {"MOVED", "JOINED_YEAR"}:
            entity, relation = query.tokens[1], query.tokens[2]
            works = [
                fact for fact in facts if fact.subject == entity and fact.relation == "WORKS"
            ]
            if works:
                company = works[-1].object
                predicted = (
                    world.grammar.company_home_city[company]
                    if relation == "MOVED"
                    else world.grammar.company_year_window[company][0]
                )
                if predicted == query.answer:
                    return query.answer
        return "ABSTAIN"

    def _entity_remap(self) -> dict[str, str]:
        permutation = self.rng.permutation(vocab.ENTITIES)
        return dict(zip(vocab.ENTITIES, (str(token) for token in permutation), strict=True))

    def sample_example(
        self,
        *,
        force_format: str | None = None,
        force_abstain: bool = False,
    ) -> TrainingExample:
        if force_format is not None and force_format not in REPRESENTATION_FORMATS:
            raise ValueError(f"unknown representation {force_format!r}")
        world = self.worlds[int(self.rng.integers(len(self.worlds)))]
        query = self._choose_query(world, force_abstain)

        items: list[_ContextItem] = []
        evidence_ids = list(query.evidence_episode_ids)
        for index, episode_id in enumerate(evidence_ids):
            representation = (
                force_format
                if index == 0 and force_format is not None
                else str(self.rng.choice(REPRESENTATION_FORMATS))
            )
            episode = world.episodes[episode_id]
            items.append(_ContextItem(
                self._render_item(world, query, episode, representation),
                episode,
                representation,
                True,
            ))

        n_distractors = int(self.rng.integers(
            int(self.config["min_distractors"]),
            int(self.config["max_distractors"]) + 1,
        ))
        distractor_pool = [
            episode.id
            for episode in world.episodes[: query.after_episode + 1]
            if episode.id not in query.evidence_episode_ids
        ]
        if distractor_pool:
            selected = self.rng.choice(
                distractor_pool,
                size=min(n_distractors, len(distractor_pool)),
                replace=False,
            )
            for episode_id in selected:
                representation = str(self.rng.choice(REPRESENTATION_FORMATS))
                episode = world.episodes[int(episode_id)]
                items.append(_ContextItem(
                    self._render_item(world, query, episode, representation),
                    episode,
                    representation,
                    False,
                ))
        self.rng.shuffle(items)

        remap = self._entity_remap()
        mapped_query = tuple(remap.get(token, token) for token in query.tokens)
        tokens: list[str] = ["BOS"]
        included: list[_ContextItem] = []
        for item in items:
            mapped_item = tuple(remap.get(token, token) for token in item.tokens)
            # Reserve the maximum fixed query/answer suffix length.
            if len(tokens) + len(mapped_item) + len(mapped_query) + 2 > self.context_length:
                continue
            tokens.extend(mapped_item)
            included.append(item)
        target = self._target_for_context(world, query, included)
        mapped_answer = remap.get(target, target)
        suffix = (*mapped_query, mapped_answer, "EOS")
        tokens.extend(suffix)
        answer_index = len(tokens) - 2
        if len(tokens) > self.context_length:
            raise RuntimeError("query and answer exceed context length")
        tokens.extend(("PAD",) * (self.context_length - len(tokens)))

        input_ids = torch.tensor(
            [vocab.TOKEN_TO_ID[token] for token in tokens], dtype=torch.long
        )
        labels = torch.full((self.context_length,), -100, dtype=torch.long)
        labels[answer_index] = vocab.TOKEN_TO_ID[mapped_answer]
        representation = force_format or "mixed"
        return TrainingExample(
            input_ids=input_ids,
            labels=labels,
            tokens=tuple(tokens),
            representation=representation,
            query_type=query.query_type,
            answer=mapped_answer,
            world_answer=query.answer,
        )

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        examples = [self.sample_example() for _ in range(batch_size)]
        return (
            torch.stack([example.input_ids for example in examples]),
            torch.stack([example.labels for example in examples]),
        )
