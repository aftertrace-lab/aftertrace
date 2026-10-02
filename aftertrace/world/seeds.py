"""Disjoint world-seed ranges and phase validation."""

from __future__ import annotations

from collections.abc import Mapping

# DECISION: Million-wide, adjacent ranges are memorable, disjoint, and leave
# ample room for all planned runs; the SPEC does not prescribe numeric bounds.
SEED_RANGES: Mapping[str, range] = {
    "reader_train": range(0, 1_000_000),
    "policy_tune": range(1_000_000, 2_000_000),
    "eval": range(2_000_000, 3_000_000),
}


def assert_phase(seed: int, phase: str) -> None:
    """Raise when *seed* is not assigned to *phase*."""

    if phase not in SEED_RANGES:
        raise ValueError(f"unknown phase {phase!r}; expected one of {tuple(SEED_RANGES)}")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed not in SEED_RANGES[phase]:
        allowed = SEED_RANGES[phase]
        raise ValueError(
            f"seed {seed!r} is not valid for phase {phase!r}; "
            f"expected [{allowed.start}, {allowed.stop})"
        )

