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
from aftertrace.run_metadata import record_run_provenance, require_clean_tree
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
        return float("nan"), True
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
    if monotone and monotone[0][1] >= goal:
        return float(monotone[0][0]), False
    for (k0, u0), (k1, u1) in zip(monotone, monotone[1:]):
        if u1 >= goal:
            if u1 == u0:
                return float(k1), False
            return k0 + (goal - u0) * (k1 - k0) / (u1 - u0), False
    return float(max(k for k, _ in monotone)), True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-config", type=Path, default=REPO_ROOT / "configs/world.yaml")
    parser.add_argument("--conditions-config", type=Path, default=REPO_ROOT / "configs/conditions.yaml")
    parser.add_argument("--sweep-config", type=Path, default=REPO_ROOT / "configs/sweep.yaml")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs/oracle_frontier")
    parser.add_argument("--worlds", type=int, default=None, help="override eval worlds per redundancy level")
    parser.add_argument("--resume", action="store_true", help="resume complete worlds from partial CSVs")
    parser.add_argument("--allow-dirty", action="store_true", help="run from a dirty tree and record that provenance")
    args = parser.parse_args()
    provenance = require_clean_tree(REPO_ROOT, allow_dirty=args.allow_dirty)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record_run_provenance(args.output_dir, provenance)
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
    tuned = tune_parameters(policy_worlds, condition_cfg, sweep["tuning"], budgets)

    rows = _load_partial(args.output_dir / "frontier.partial.csv") if args.resume else []
    details = _load_partial(args.output_dir / "per_type.partial.csv") if args.resume else []
    complete_seeds = {
        int(seed) for seed in {row["seed"] for row in rows}
        if sum(int(row["seed"]) == int(seed) for row in rows)
        == len(CONDITIONS) * len(budgets)
    }
    eval_index = 0
    for p in levels:
        for world_index in range(worlds_per_level):
            seed = 2_000_000 + eval_index
            eval_index += 1
            if seed in complete_seeds:
                print(f"[oracle-frontier] resuming past completed seed={seed}", flush=True)
                continue
            cfg = dict(world_cfg)
            cfg.update(phase="eval", p_redundancy=float(p))
            world = make_world(seed, cfg)
            sample = _stratified_unique_queries(
                world, int(sweep["stratified_queries_per_type"])
            )
            progressive_budgets = {}
            for K in budgets:
                tuned_k = tuned["by_K"][str(K)]
                progressive_eval = evaluate_condition(
                    world, "progressive", K, condition_config=condition_cfg,
                    tuned=tuned_k["progressive"], queries=sample,
                )
                progressive_budgets[K] = int(progressive_eval.mean_accounting["store_tokens"])
                for name in CONDITIONS:
                    if name == "progressive":
                        result = progressive_eval
                    else:
                        condition_tuning = (
                            tuned_k["progressive"] if name.startswith("progressive")
                            else tuned_k["evict_learned"] if name == "evict_learned"
                            else None
                        )
                        result = evaluate_condition(
                            world, name, K, condition_config=condition_cfg,
                            tuned=condition_tuning, store_budget_tokens=progressive_budgets[K],
                            queries=sample,
                        )
                    rows.append({
                        "seed": seed, "p_redundancy": float(p), "world_index": world_index,
                        "condition": name, "K": K, "n_queries": len(result.results),
                        "utility": result.utility,
                        "context_tokens": result.mean_accounting.get("context_tokens", 0.0),
                        "retrieval_ops": result.mean_accounting.get("retrieval_ops", 0.0),
                        "store_tokens": result.mean_accounting.get("store_tokens", 0.0),
                        "mean_tokens_per_item": result.mean_accounting.get("mean_rendered_tokens_per_item", 0.0),
                        "mean_raw_l4_tokens_per_item": result.mean_accounting.get("mean_raw_l4_tokens_per_item", 0.0),
                    })
                    for qtype, metrics in result.per_type.items():
                        details.append({
                            "seed": seed, "p_redundancy": float(p), "condition": name,
                            "K": K, "query_type": qtype, **metrics,
                        })
                print(
                    f"[oracle-frontier] p={p:.1f} world={world_index + 1}/{worlds_per_level} K={K} complete",
                    flush=True,
                )
            _write_csv(args.output_dir / "frontier.partial.csv", rows)
            _write_csv(args.output_dir / "per_type.partial.csv", details)
            print(
                "[oracle-frontier] sampled n="
                + ",".join(
                    f"{kind}:{sum(query.query_type == kind for query in sample)}"
                    for kind in ("A", "B", "C", "E", "R")
                ),
                flush=True,
            )
    _write_csv(args.output_dir / "frontier.csv", rows)
    _write_csv(args.output_dir / "per_type.csv", details)
    _plot(rows, levels, budgets, args.output_dir / "fig1_utility_vs_k.png")

    kstar_rows = []
    pair_ratio_rows = []
    print("K* ratios by p and utility target (best raw / progressive; >1 favors progressive)")
    for p in levels:
        for target in sweep["frontier_targets"]:
            kp, kp_censored = _frontier_k(rows, "progressive", float(p), float(target))
            raw_pairs = {
                name: _frontier_k(rows, name, float(p), float(target))
                for name in ("raw_offload", "raw_offload_qrender")
            }
            # The best raw baseline is the one reaching the target at lower K;
            # a censored baseline is conservatively treated as > max K.
            kr, kr_censored = min(
                raw_pairs.items(),
                key=lambda pair: float("inf") if pair[1][1] else pair[1][0],
            )[1]
            ratio, relation = _ratio_bound(kr, kr_censored, kp, kp_censored, max(budgets))
            kstar_rows.extend([
                {"p_redundancy": p, "target": target, "condition": "progressive", "K_star": kp, "censored": kp_censored},
                {"p_redundancy": p, "target": target, "condition": "best_raw", "K_star": kr, "censored": kr_censored},
            ])
            kp_text = f">{max(budgets)}" if kp_censored else f"{kp:.2f}"
            kr_text = f">{max(budgets)}" if kr_censored else f"{kr:.2f}"
            ratio_text = (
                f"{relation}{ratio:.3f}" if relation in {"=", ">", "<"}
                else "indeterminate (both censored)"
            )
            print(f"p={p:.1f} target={target:.2f} raw/progressive={ratio_text} (raw K*={kr_text}, progressive K*={kp_text})")
            for condition, (raw_k, raw_censored) in raw_pairs.items():
                pair_ratio, pair_bound = _ratio_bound(
                    raw_k, raw_censored, kp, kp_censored, max(budgets)
                )
                pair_ratio_rows.append({
                    "p_redundancy": p, "target": target,
                    "raw_condition": condition, "ratio": pair_ratio,
                    "bound": pair_bound,
                })
                pair_text = (
                    f"{pair_bound}{pair_ratio:.3f}" if pair_bound in {"=", ">", "<"}
                    else "indeterminate (both censored)"
                )
                print(f"  pair {condition}/progressive={pair_text}")
    _write_csv(args.output_dir / "kstar.csv", kstar_rows)
    _write_csv(args.output_dir / "kstar_ratios_by_pair.csv", pair_ratio_rows)
    matched_k_rows = _matched_k_bootstrap(
        rows, levels, budgets,
        replicates=int(sweep["matched_k_bootstrap_replicates"]),
        seed=int(sweep["matched_k_bootstrap_seed"]),
    )
    _write_csv(args.output_dir / "matched_k_bootstrap.csv", matched_k_rows)
    _print_utility_table(rows, levels, budgets)
    _print_selected_per_type(details, levels)
    _print_qrender_size_table(rows, levels, budgets)
    _print_matched_k_bootstrap(matched_k_rows, levels)

    decision = _decision_from_pair_ratios(pair_ratio_rows)
    censored_count = sum(bool(row["censored"]) for row in kstar_rows)
    print(f"Original K*-ratio rule: {decision}: oracle-reader pre-check (censored K* cases={censored_count}; included conservatively)")
    fallback = _matched_k_fallback(kstar_rows, matched_k_rows)
    print(f"After-v2 matched-K fallback: {fallback}")


def _ratio_bound(raw, raw_censored, progressive, progressive_censored, max_k):
    if raw_censored and progressive_censored:
        return float("nan"), "interval"
    if raw_censored:
        return max_k / progressive, ">"
    if progressive_censored:
        return raw / max_k, "<"
    return raw / progressive, "="


def _decision_from_pair_ratios(rows):
    """Apply the pre-check conservatively to every raw/progressive pair and target."""

    if not rows:
        return "REDUCED"
    if any(
        float(row["ratio"]) >= 1.5 and row["bound"] in {"=", ">"}
        for row in rows
    ):
        return "PASS"
    no_pair_can_reach_1_2 = all(
        (row["bound"] == "=" and float(row["ratio"]) < 1.2)
        or (row["bound"] == "<" and float(row["ratio"]) <= 1.2)
        for row in rows
    )
    return "STOP" if no_pair_can_reach_1_2 else "REDUCED"


def _matched_k_bootstrap(rows, levels, budgets, *, replicates: int, seed: int):
    """Paired bootstrap over eval worlds; best raw is max of raw means per draw."""
    output = []
    for level_index, p in enumerate(levels):
        for budget in budgets:
            selected = [
                row for row in rows
                if row["p_redundancy"] == p and row["K"] == budget
            ]
            by_condition_seed = {
                (row["condition"], int(row["seed"])): float(row["utility"])
                for row in selected
            }
            conditions = ("progressive", "raw_offload", "raw_offload_qrender")
            seeds = sorted(
                set.intersection(*(
                    {world_seed for (condition, world_seed) in by_condition_seed
                     if condition == name}
                    for name in conditions
                ))
            ) if all(
                any(condition == name for condition, _ in by_condition_seed)
                for name in conditions
            ) else []
            if not seeds:
                continue
            progressive = np.asarray(
                [by_condition_seed[("progressive", world_seed)] for world_seed in seeds]
            )
            raw_offload = np.asarray(
                [by_condition_seed[("raw_offload", world_seed)] for world_seed in seeds]
            )
            raw_qrender = np.asarray(
                [by_condition_seed[("raw_offload_qrender", world_seed)] for world_seed in seeds]
            )
            best_raw_condition = (
                "raw_offload" if raw_offload.mean() >= raw_qrender.mean()
                else "raw_offload_qrender"
            )
            best_raw_mean = max(float(raw_offload.mean()), float(raw_qrender.mean()))
            difference = float(progressive.mean()) - best_raw_mean
            rng = np.random.default_rng(seed + level_index * 100_000 + int(budget))
            draws = rng.integers(0, len(seeds), size=(replicates, len(seeds)))
            prog_draw = progressive[draws].mean(axis=1)
            raw_draw = np.maximum(
                raw_offload[draws].mean(axis=1),
                raw_qrender[draws].mean(axis=1),
            )
            delta_draw = prog_draw - raw_draw
            lower, upper = np.quantile(delta_draw, (0.025, 0.975))
            output.append({
                "p_redundancy": p, "K": budget, "n_worlds": len(seeds),
                "progressive_utility": float(progressive.mean()),
                "best_raw_condition": best_raw_condition,
                "best_raw_utility": best_raw_mean,
                "difference": difference,
                "ci95_lower": float(lower), "ci95_upper": float(upper),
                "bootstrap_replicates": replicates,
            })
    return output


def _matched_k_fallback(kstar_rows, matched_rows):
    grouped = {}
    for row in kstar_rows:
        grouped.setdefault((row["p_redundancy"], row["target"]), {})[row["condition"]] = bool(row["censored"])
    if not grouped or not all(
        values.get("progressive", False) and values.get("best_raw", False)
        for values in grouped.values()
    ):
        return "NOT TRIGGERED (progressive and best-raw K* are not both censored at every p/target)"
    if matched_rows and all(float(row["ci95_upper"]) < 0.03 for row in matched_rows):
        return "STOP (upper 95% matched-K bootstrap bound < +0.03 at every p and K)"
    return "NO STOP (the matched-K STOP criterion is not met)"


def _mean(rows):
    return sum(rows) / len(rows) if rows else float("nan")


def _print_utility_table(rows, levels, budgets):
    print("\nUTILITY BY CONDITION AND K (mean over eval worlds)")
    for p in levels:
        subset = [row for row in rows if row["p_redundancy"] == p]
        names = [name for name in CONDITIONS if any(row["condition"] == name for row in subset)]
        print(f"p_redundancy={p:.1f}")
        print("K," + ",".join(names))
        for K in budgets:
            values = []
            for name in names:
                group = [row["utility"] for row in subset if row["condition"] == name and row["K"] == K]
                values.append(f"{_mean(group):.4f}")
            print(f"{K}," + ",".join(values))


def _print_selected_per_type(details, levels):
    selected = {"progressive", "raw_offload", "raw_offload_qrender", "summary"}
    print("\nPER-TYPE UTILITY (K=32,64; mean over eval worlds)")
    for p in levels:
        for K in (32, 64):
            print(f"p={p:.1f} K={K}")
            rows = [row for row in details if row["p_redundancy"] == p and row["K"] == K and row["condition"] in selected]
            keys = sorted({(row["condition"], row["query_type"]) for row in rows})
            for condition, query_type in keys:
                group = [row for row in rows if row["condition"] == condition and row["query_type"] == query_type]
                n = sum(row["n"] for row in group)
                utility = _mean([row["utility"] for row in group])
                print(f"  {condition:20s} type={query_type} n={int(n)} U={utility:.4f}")


def _print_qrender_size_table(rows, levels, budgets):
    print("\nQRENDER TOKENS PER ITEM (mean over eval worlds; raw L4 measured on the same retrieved items)")
    for p in levels:
        print(f"p_redundancy={p:.1f}")
        for budget in budgets:
            group = [
                row for row in rows
                if row["p_redundancy"] == p and row["K"] == budget
                and row["condition"] == "raw_offload_qrender"
            ]
            rendered = _mean([float(row.get("mean_tokens_per_item", 0.0)) for row in group])
            raw = _mean([float(row.get("mean_raw_l4_tokens_per_item", 0.0)) for row in group])
            print(f"  K={budget}: rendered={rendered:.2f} raw_L4={raw:.2f}")


def _print_matched_k_bootstrap(rows, levels):
    print("\nMATCHED-K UTILITY DIFFERENCE (progressive - best raw; paired world bootstrap 95% CI)")
    for p in levels:
        print(f"p_redundancy={p:.1f}")
        for row in rows:
            if row["p_redundancy"] == p:
                print(
                    f"  K={row['K']:>3}: delta={row['difference']:+.4f} "
                    f"CI=[{row['ci95_lower']:+.4f}, {row['ci95_upper']:+.4f}] "
                    f"best_raw={row['best_raw_condition']} n_worlds={row['n_worlds']}"
                )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_partial(path: Path) -> list[dict]:
    if not path.exists():
        return []
    numeric_int = {"seed", "world_index", "K", "n_queries", "n"}
    numeric_float = {"p_redundancy", "utility", "context_tokens", "retrieval_ops", "store_tokens", "accuracy", "abstain"}
    result = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key in numeric_int & row.keys():
                row[key] = int(float(row[key]))
            for key in numeric_float & row.keys():
                row[key] = float(row[key])
            result.append(row)
    return result


def _stratified_unique_queries(world, per_type):
    grouped = {query_type: [] for query_type in ("A", "B", "C", "E", "R")}
    seen = {query_type: set() for query_type in grouped}
    for query in world.queries:
        if query.query_type not in grouped:
            continue
        signature = (query.tokens, query.answer)
        if signature not in seen[query.query_type]:
            grouped[query.query_type].append(query)
            seen[query.query_type].add(signature)
    return [query for kind in grouped for query in grouped[kind][:min(per_type, len(grouped[kind]))]]


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
