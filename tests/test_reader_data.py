from __future__ import annotations

import yaml

from aftertrace.reader.data import ReaderDataGenerator
from aftertrace.world.generator import make_world
from aftertrace.world.vocab import TOKEN_TO_ID


def _reader_config() -> dict:
    with open("configs/reader_r5m.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_reader_examples_are_deterministic_and_answer_only(world_config: dict) -> None:
    train_config = {**world_config, "phase": "reader_train"}
    world = make_world(0, train_config)
    first = ReaderDataGenerator((world,), _reader_config(), seed=17)
    second = ReaderDataGenerator((world,), _reader_config(), seed=17)
    left = first.sample_example(force_format="l1")
    right = second.sample_example(force_format="l1")
    assert left.tokens == right.tokens
    assert left.input_ids.equal(right.input_ids)
    assert left.labels.equal(right.labels)
    assert (left.labels != -100).sum().item() == 1
    assert "GIST" in left.tokens


def test_reader_formats_and_abstain_examples(world_config: dict) -> None:
    train_config = {**world_config, "phase": "reader_train"}
    world = make_world(1, train_config)
    data = ReaderDataGenerator((world,), _reader_config(), seed=23)
    residual = data.sample_example(force_format="residual")
    abstain = data.sample_example(force_abstain=True)
    assert residual.representation == "residual"
    answer_positions = (abstain.labels != -100).nonzero().flatten()
    assert len(answer_positions) == 1
    assert abstain.labels[answer_positions[0]].item() == TOKEN_TO_ID["ABSTAIN"]


def test_context_missing_world_fact_uses_only_world_value_or_abstain(
    world_config: dict,
) -> None:
    train_config = {**world_config, "phase": "reader_train"}
    world = make_world(2, train_config)
    config = {**_reader_config(), "context_length": 16, "min_distractors": 0, "max_distractors": 0}
    data = ReaderDataGenerator((world,), config, seed=29)
    values = {fact.object for episode in world.episodes for fact in episode.facts}
    observed_context_abstain = False
    for _ in range(100):
        example = data.sample_example(force_abstain=False)
        assert example.world_answer in values or example.world_answer.startswith("N_")
        assert (
            example.answer in {example.world_answer, "ABSTAIN"}
            or example.answer.startswith("E_") and example.world_answer.startswith("E_")
        )
        observed_context_abstain |= example.answer == "ABSTAIN"
    assert observed_context_abstain
