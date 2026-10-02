#!/usr/bin/env python3
"""Print a small, human-readable sample from a generated world."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import yaml

# Allow direct execution from a source checkout without requiring installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aftertrace.world.generator import make_world


def print_query_with_evidence(world, query, *, include_episodes: bool = True) -> None:
    evidence = ",".join(str(index) for index in query.evidence_episode_ids) or "none"
    print(
        f"[{query.id:04d} after={query.after_episode:04d} "
        f"type={query.query_type} evidence={evidence} "
        f"competitor_count={query.competitor_count}] "
        f"{' '.join(query.tokens)} -> {query.answer}"
    )
    if include_episodes:
        for episode_id in query.evidence_episode_ids:
            episode = world.episodes[episode_id]
            print(f"  evidence [{episode.id:04d}] {' '.join(episode.tokens)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/world.yaml")
    parser.add_argument("--seed", type=int, default=2_000_000)
    parser.add_argument("--phase", choices=("reader_train", "policy_tune", "eval"), default="eval")
    parser.add_argument("--p-redundancy", type=float, default=0.6)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["phase"] = args.phase
    config["p_redundancy"] = args.p_redundancy
    world = make_world(args.seed, config)

    print(
        f"world seed={world.seed} phase={world.phase} "
        f"p_redundancy={config['p_redundancy']} "
        f"episodes={len(world.episodes)} queries={len(world.queries)}"
    )
    print(
        "restatement_rates "
        + " ".join(
            f"{relation}={rate:.4f}"
            for relation, rate in world.restatement_rates.items()
        )
    )
    print("\nEPISODES (first 5)")
    for episode in world.episodes[:5]:
        print(f"[{episode.id:04d}] {' '.join(episode.tokens)}")
    print("\nQUERIES (first 10)")
    for query in world.queries[:10]:
        print_query_with_evidence(world, query, include_episodes=False)

    selections = (
        ("TYPE B (3)", lambda query: query.query_type == "B"),
        (
            "TYPE C — MOVED CURRENT (3)",
            lambda query: query.query_type == "C"
            and query.tokens[2:4] == ("MOVED", "CURRENT"),
        ),
        (
            "TYPE C — MOVED PREV (3)",
            lambda query: query.query_type == "C"
            and query.tokens[2:4] == ("MOVED", "PREV"),
        ),
        ("TYPE R (3)", lambda query: query.query_type == "R"),
    )
    for heading, predicate in selections:
        selected = [query for query in world.queries if predicate(query)][:3]
        if len(selected) != 3:
            raise RuntimeError(f"world does not contain three samples for {heading}")
        print(f"\n{heading}")
        for query in selected:
            print_query_with_evidence(world, query)

    histogram = Counter(
        query.competitor_count for query in world.queries if query.query_type == "R"
    )
    print("\nTYPE R competitor_count histogram")
    for competitor_count in sorted(histogram):
        print(f"{competitor_count}: {histogram[competitor_count]}")


if __name__ == "__main__":
    main()
