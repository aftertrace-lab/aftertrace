#!/usr/bin/env python3
"""Evaluate the Stage-0 information frontier with the symbolic oracle."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aftertrace.eval.harness import evaluate_condition, tune_parameters
from aftertrace.world.generator import make_world

CONDITIONS = (
    "none", "full_oracle", "raw_offload", "raw_offload_qrender",
    "residual_offload", "evict_fifo", "evict_lru", "evict_freq",
    "evict_learned", "progressive", "progressive_nogate",
    "progressive_uniform", "summary",
)


def _frontier_k(rows, condition, p, target):
    selected = [row for row in rows if row["condition"] == condition and row["p_redundancy"] == p]
    full = [row["utility"] for row in rows if row["condition"] == "full_oracle" and row["p_redundancy"] == p]
    if not selected or not full:
        return float("inf")
    goal = target * sum(full) / len(full)
    by_k = {}
    for row in selected:
        by_k.setdefault(row["K"], []).append(row["utility"])
    points = sorted((k, sum(values) / len(values)) for k, values in by_k.items())
    # Monotonize utility as K grows, then linearly interpolate the first crossing.
    best = float("-inf")
    monotone = []
    for k, utility in points:
        best = max(best, utility)
        monotone.append((k, best))
    for (k0, u0), (k1, u1) in zip(monotone, monotone[1:]):
        if u1 >= goal:
            if u1 == u0:
                return float(k1)
            return k0 + (goal - u0) * (k1 - k0) / (u1 - u0)
    return float("inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-config", type=Path, default=REPO_ROOT / "configs/world.yaml")
    parser.add_argument("--conditions-config", type=Path, default=REPO_ROOT / "configs/conditions.yaml")
    parser.add_argument("--sweep-config", type=Path, default=REPO_ROOT / "configs/sweep.yaml")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs/oracle_frontier")
    parser.add_argument("--worlds", type=int, default=None, help="override eval worlds per redundancy level")
    args = parser.parse_args()
    world_cfg = yaml.safe_load(args.world_config.read_text(encoding="utf-8"))
    condition_cfg = yaml.safe_load(args.conditions_config.read_text(encoding="utf-8"))
    sweep = yaml.safe_load(args.sweep_config.read_text(encoding="utf-8"))
    levels = sweep["p_redundancy_levels"]
    worlds_per_level = args.worlds or int(sweep["eval_worlds_per_level"])
    budgets = [int(k) for k in sweep["K"]]
    tune_cfg = dict(world_cfg)
    tune_cfg["phase"] = "policy_tune"
    policy_worlds = []
    for index in range(int(sweep["policy_tune_worlds"])):
        seed = 1_000_000 + index
        policy_worlds.append(make_world(seed, tune_cfg))
    tuned = tune_parameters(policy_worlds, condition_cfg, sweep["tuning"])

    rows = []
    details = []
    eval_index = 0
    for p in levels:
        for world_index in range(worlds_per_level):
            seed = 2_000_000 + eval_index
            eval_index += 1
            cfg = dict(world_cfg)
            cfg.update(phase="eval", p_redundancy=float(p))
            world = make_world(seed, cfg)
            sample = list(world.queries)
            progressive_budgets = {}
            for K in budgets:
                progressive_eval = evaluate_condition(
                    world, "progressive", K, condition_config=condition_cfg,
                    tuned=tuned, queries=sample,
                )
                progressive_budgets[K] = int(progressive_eval.mean_accounting["store_tokens"])
                for name in CONDITIONS:
                    if name == "progressive":
                        result = progressive_eval
                    else:
                        result = evaluate_condition(
                            world, name, K, condition_config=condition_cfg,
                            tuned=tuned, store_budget_tokens=progressive_budgets[K],
                            queries=sample,
                        )
                    rows.append({
                        "seed": seed, "p_redundancy": float(p), "world_index": world_index,
                        "condition": name, "K": K, "n_queries": len(result.results),
                        "utility": result.utility,
                        "context_tokens": result.mean_accounting.get("context_tokens", 0.0),
                        "retrieval_ops": result.mean_accounting.get("retrieval_ops", 0.0),
                        "store_tokens": result.mean_accounting.get("store_tokens", 0.0),
                    })
                    for qtype, metrics in result.per_type.items():
                        details.append({
                            "seed": seed, "p_redundancy": float(p), "condition": name,
                            "K": K, "query_type": qtype, **metrics,
                        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "frontier.csv", rows)
    _write_csv(args.output_dir / "per_type.csv", details)
    _plot(rows, levels, budgets, args.output_dir / "fig1_utility_vs_k.png")

    # DECISION: Express headroom as K*_raw / K*_progressive so values above 1
    # directly mean progressive needs less context for the same utility.
    print("K* ratios (best raw baseline divided by progressive; >1 favors progressive)")
    ratios = []
    for p in levels:
        for target in sweep["frontier_targets"]:
            kp = _frontier_k(rows, "progressive", float(p), float(target))
            kr = min(
                _frontier_k(rows, "raw_offload", float(p), float(target)),
                _frontier_k(rows, "raw_offload_qrender", float(p), float(target)),
            )
            ratio = kr / kp if kp and kp != float("inf") else float("inf")
            ratios.append(ratio)
            print(f"p={p:.1f} target={target:.2f} K*_raw/K*_progressive={ratio:.3f} (raw={kr:.2f}, progressive={kp:.2f})")
    # Apply the amendment's three-way decision to the preregistered 0.9 frontier.
    primary = [
        min(
            _frontier_k(rows, "raw_offload", float(p), 0.9),
            _frontier_k(rows, "raw_offload_qrender", float(p), 0.9),
        ) / _frontier_k(rows, "progressive", float(p), 0.9)
        for p in levels
    ]
    if all(value < 1.2 for value in primary):
        decision = "STOP"
    elif any(value >= 1.5 for value in primary):
        decision = "PASS"
    else:
        decision = "REDUCED"
    print(f"{decision}: oracle-reader pre-check")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows, levels, budgets, path):
    fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 4), sharey=True)
    if len(levels) == 1:
        axes = [axes]
    for ax, p in zip(axes, levels, strict=True):
        for condition in CONDITIONS:
            selected = [row for row in rows if row["p_redundancy"] == p and row["condition"] == condition]
            if selected:
                by_k = {}
                for row in selected:
                    by_k.setdefault(row["K"], []).append(row["utility"])
                means, lower, upper = [], [], []
                rng = np.random.default_rng(1207)
                for k in budgets:
                    values = np.asarray(by_k.get(k, [0.0]), dtype=float)
                    means.append(float(values.mean()))
                    draws = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
                    lower.append(float(np.quantile(draws, 0.025)))
                    upper.append(float(np.quantile(draws, 0.975)))
                ax.plot(budgets, means, marker="o", label=condition)
                ax.fill_between(budgets, lower, upper, alpha=0.12)
        ax.set_xscale("log", base=2)
        ax.set_title(f"p_redundancy={p}")
        ax.set_xlabel("K tokens")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Utility U")
    axes[-1].legend(fontsize=6, bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
