"""Fixed shared grammar prior for correlated facts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import vocab


@dataclass(frozen=True)
class GrammarPrior:
    company_home_city: dict[str, str]
    city_typical_places: dict[str, tuple[str, ...]]
    company_year_window: dict[str, tuple[str, ...]]

    def predicts_moved(self, company: str, city: str) -> bool:
        return self.company_home_city[company] == city

    def predicts_joined_year(self, company: str, year: str) -> bool:
        return year == self.company_year_window[company][0]

    def predicts_place(self, city: str, place: str) -> bool:
        return place == self.city_typical_places[city][0]


def make_grammar(
    seed: int,
    typical_places_per_city: int,
    hiring_window_years: int,
) -> GrammarPrior:
    """Build the shared prior solely from its fixed grammar seed."""

    if not 1 <= typical_places_per_city <= len(vocab.PLACES):
        raise ValueError("typical_places_per_city is out of range")
    if not 1 <= hiring_window_years <= len(vocab.YEARS):
        raise ValueError("hiring_window_years is out of range")

    rng = np.random.default_rng(seed)
    homes = {
        company: str(rng.choice(vocab.CITIES)) for company in vocab.COMPANIES
    }
    places = {
        city: tuple(str(x) for x in rng.choice(
            vocab.PLACES, size=typical_places_per_city, replace=False
        ))
        for city in vocab.CITIES
    }
    windows: dict[str, tuple[str, ...]] = {}
    last_start = len(vocab.YEARS) - hiring_window_years
    for company in vocab.COMPANIES:
        start = int(rng.integers(0, last_start + 1))
        windows[company] = vocab.YEARS[start : start + hiring_window_years]
    return GrammarPrior(homes, places, windows)


def redundant_or_uniform(
    rng: np.random.Generator,
    redundant_values: tuple[str, ...],
    all_values: tuple[str, ...],
    p_redundancy: float,
) -> str:
    """Sample from the prior with probability p, otherwise uniformly outside it."""

    if not 0.0 <= p_redundancy <= 1.0:
        raise ValueError("p_redundancy must be in [0, 1]")
    # DECISION: "else uniform" is interpreted as uniform over non-prior values,
    # so the measured prior-match probability is exactly p_redundancy in expectation.
    if rng.random() < p_redundancy:
        return redundant_values[0]
    redundant = set(redundant_values)
    alternatives = tuple(value for value in all_values if value not in redundant)
    return str(rng.choice(alternatives))
