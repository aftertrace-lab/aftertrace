#!/usr/bin/env python3
"""Render publication figures from the frozen Stage-0 v3 result tables."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "oracle_frontier_v3"
OUT = ROOT / "figures"
CONDITIONS = [
    ("progressive", "Progressive"),
    ("raw_offload", "Raw offload"),
    ("raw_offload_qrender", "Raw + qrender"),
    ("residual_offload", "Residual"),
    ("evict_lru", "LRU"),
    ("summary", "Summary"),
]
COLORS = ["#1768ac", "#e07a1f", "#2a9d65", "#8756a5", "#555555", "#bb4d4d"]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    frontier = pd.read_csv(RUN / "frontier.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, p in zip(axes, [0.2, 0.6, 0.9]):
        data = frontier[frontier.p_redundancy == p]
        for (condition, label), color in zip(CONDITIONS, COLORS):
            means = data[data.condition == condition].groupby("K").utility.mean()
            ax.plot(means.index, means.values, marker="o", linewidth=2, label=label, color=color)
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 32, 64, 128, 256, 384], labels=["16", "32", "64", "128", "256", "384"])
        ax.set_ylim(0.15, 1.03)
        ax.set_title(f"p_redundancy = {p:.1f}")
        ax.set_xlabel("Context budget K (tokens/query)")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Mean oracle-reader utility")
    axes[-1].legend(loc="lower right", fontsize=8, frameon=True)
    fig.suptitle("Stage 0 — utility vs context budget (4 eval worlds per p)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "stage0_fig1_utility_vs_k.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    by_type = pd.read_csv(RUN / "per_type.csv")
    selected = [condition for condition, _ in CONDITIONS]
    d = by_type[(by_type.K == 64) & by_type.condition.isin(selected)]
    # Pool query-level outcomes across the four paired evaluation worlds.
    d = d.assign(weighted_utility=d.utility * d.n)
    grouped = d.groupby(["p_redundancy", "condition", "query_type"], as_index=False).agg(
        weighted_utility=("weighted_utility", "sum"), n=("n", "sum")
    )
    grouped["utility"] = grouped.weighted_utility / grouped.n
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    query_types = ["A", "B", "C", "E", "R"]
    x = range(len(query_types))
    width = 0.12
    for ax, p in zip(axes, [0.2, 0.6, 0.9]):
        pdata = grouped[grouped.p_redundancy == p]
        for i, ((condition, label), color) in enumerate(zip(CONDITIONS, COLORS)):
            vals = [
                float(pdata[(pdata.condition == condition) & (pdata.query_type == q)].utility.iloc[0])
                for q in query_types
            ]
            offsets = [j + (i - (len(CONDITIONS) - 1) / 2) * width for j in x]
            ax.bar(offsets, vals, width=width, label=label, color=color)
        ax.set_xticks(list(x), query_types)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"p_redundancy = {p:.1f}")
        ax.set_xlabel("Query type")
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Mean utility at K = 64")
    axes[-1].legend(loc="upper left", fontsize=8, frameon=True)
    fig.suptitle("Stage 0 — per-type utility at K = 64 (query-weighted across 4 worlds)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "stage0_per_type_k64.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
