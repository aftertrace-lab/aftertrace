from __future__ import annotations

import pytest

from aftertrace.memory.render import (
    decode_residual,
    gist_entity_relation_pairs,
    predicted_at_indices,
    predicted_fact_indices,
    quantize_fidelity,
    render_l1,
    render_l2,
    render_l3,
    render_l4,
    render_residual,
    render_summaries,
)
from aftertrace.world.generator import Episode, Fact
from aftertrace.world.generator import make_world
from aftertrace.world.grammar import make_grammar


def _episode(facts: tuple[Fact, ...], filler: tuple[str, ...] = ("F_1",)) -> Episode:
    tokens = ["EP", "T_2"]
    for fact in facts:
        tokens.extend(fact.render())
        tokens.append("SEP")
    tokens.extend(filler)
    tokens.append("EOS")
    return Episode(7, "T_2", facts, filler, tuple(tokens))


def test_all_rendering_formats_are_deterministic() -> None:
    grammar = make_grammar(1729, 4, 8)
    company = "O_0"
    city = grammar.company_home_city[company]
    facts = (
        Fact("WORKS", "E_1", company, (), "core_relational"),
        Fact("MOVED", "E_1", city, (), "causal_linked"),
        Fact("COLOR", "E_1", "K_2", (), "incidental"),
    )
    episode = _episode(facts)
    weights = {"core_relational": 1.0, "causal_linked": 0.8, "incidental": 0.1}
    renderers = (
        lambda: render_l4(episode),
        lambda: render_l3(episode),
        lambda: render_l2(episode, weights, 0.5),
        lambda: render_l1(episode),
        lambda: render_residual(episode, grammar),
        lambda: render_summaries((episode,)),
    )
    for render in renderers:
        assert render() == render()
    assert render_l4(episode)[0] == "EP"
    assert render_l3(episode)[0] == "EP"
    assert render_l2(episode, weights, 0.5)[0] == "EPP"
    assert render_residual(episode, grammar)[0] == "RES"
    assert render_l1(episode)[0] == "GIST"
    assert render_summaries((episode,))[0][0] == "SUM"
    assert [quantize_fidelity(value) for value in (0.0, 0.2, 0.4, 0.6, 1.0)] == [0, 1, 2, 3, 4]


def test_residual_marks_predicted_values_and_preserves_incidental_facts() -> None:
    grammar = make_grammar(1729, 4, 8)
    company = "O_0"
    city = grammar.company_home_city[company]
    year = grammar.company_year_window[company][0]
    place = grammar.city_typical_places[city][0]
    facts = (
        Fact("WORKS", "E_1", company, (), "core_relational"),
        Fact("MOVED", "E_1", city, (), "causal_linked"),
        Fact("JOINED_YEAR", "E_1", year, (), "temporal"),
        Fact("TOLD", "E_1", "E_2", ("AT", place), "social"),
        Fact("COLOR", "E_1", "K_2", (), "incidental"),
    )
    episode = _episode(facts, ())
    assert predicted_fact_indices(episode, grammar) == (1, 2)
    assert predicted_at_indices(episode, grammar) == (3,)
    residual = render_residual(episode, grammar)
    assert city not in residual and year not in residual and place not in residual
    assert residual.count("PRED") == 3
    assert "E_1 TOLD E_2 AT PRED" in " ".join(residual)
    assert "WORKS" in residual and "COLOR" in residual
    assert decode_residual(residual, grammar) == facts


def test_gist_drops_values_and_round_trips_grouped_relation_keys() -> None:
    facts = (
        Fact("WORKS", "E_1", "O_1", (), "core_relational"),
        Fact("COLOR", "E_1", "K_1", (), "incidental"),
        Fact("WORKS", "E_2", "O_2", (), "core_relational"),
        Fact("OWNS", "E_2", "J_2", (), "incidental"),
        Fact("MOVED", "E_1", "C_1", (), "causal_linked"),
        Fact("MET", "E_1", "E_2", ("AT", "P_1"), "social"),
    )
    gist = render_l1(_episode(facts))
    assert set(gist_entity_relation_pairs(gist)) == {
        (fact.subject, fact.relation) for fact in facts
    }
    assert gist[0:2] == ("GIST", "T_2")
    assert not any(token.startswith(("C_", "O_", "P_", "Y_", "K_", "J_")) for token in gist)
    assert "MASK" not in gist
    assert gist.index("WORKS") < gist.index("MOVED") < gist.index("MET")
    assert "E_2" in gist


def test_l2_uses_canonical_prefix_sixty_percent_placeholder() -> None:
    facts = tuple(
        Fact("COLOR", "E_1", f"K_{i}", (), "incidental") for i in range(5)
    )
    episode = _episode(facts)
    rendered = render_l2(episode, {"incidental": 0.0}, 99.0)
    assert rendered[0] == "EPP"
    assert rendered.count("COLOR") == 3
    assert "K_0" in rendered and "K_1" in rendered and "K_2" in rendered
    assert "K_3" not in rendered and "K_4" not in rendered


def test_gist_for_world_episodes_contains_no_value_tokens(world_config: dict) -> None:
    world = make_world(2_000_061, world_config)
    value_prefixes = ("C_", "O_", "P_", "Y_", "K_", "J_")
    for episode in world.episodes:
        gist = render_l1(episode)
        assert not any(token.startswith(value_prefixes) for token in gist)
        assert set(episode.entities).issubset(set(gist))
        assert set(gist_entity_relation_pairs(gist)) == {
            (fact.subject, fact.relation) for fact in episode.facts
        }


def test_summaries_are_latest_values_through_inclusive_prefix() -> None:
    earlier = _episode((Fact("WORKS", "E_1", "O_1", (), "core_relational"),), ())
    later = Episode(
        1,
        "T_3",
        (Fact("WORKS", "E_1", "O_2", (), "core_relational"),),
        (),
        ("EP", "T_3", "E_1", "WORKS", "O_2", "SEP", "EOS"),
    )
    before = render_summaries((earlier,))[0]
    through_later = render_summaries((earlier, later))[0]
    assert "O_1" in before
    assert "O_2" in through_later and "O_1" not in through_later


@pytest.mark.slow
def test_residual_decodes_every_episode_in_three_worlds(world_config: dict) -> None:
    for seed in range(2_000_050, 2_000_053):
        world = make_world(seed, world_config)
        for episode in world.episodes:
            encoded = render_residual(episode, world.grammar)
            assert decode_residual(encoded, world.grammar) == episode.facts
