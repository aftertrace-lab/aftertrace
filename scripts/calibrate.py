#!/usr/bin/env python3
"""Train reader calibration runs and compare none/full-oracle utility."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/aftertrace-calibration-mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aftertrace.memory.render import render_l4
from aftertrace.reader.data import ReaderDataGenerator
from aftertrace.reader.model import ReaderModel, count_parameters
from aftertrace.reader.train import build_training, run_steps, select_device
from aftertrace.world.generator import World, make_world
from aftertrace.world.queries import Query
from aftertrace.world.vocab import TOKENS, TOKEN_TO_ID


SIZES = ("r2m", "r5m", "r14m", "r31m")
REDUNDANCY_LEVELS = (0.2, 0.6, 0.9)
CALIBRATION_RULES = {
    "full_oracle_within_2_points_of_r31m": 0.02,
    "between_seed_sd_below_1_point": 0.01,
    "none_within_5_points_of_chance": 0.05,
}


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.inference_mode()
def validation_loss(
    model: ReaderModel,
    data: ReaderDataGenerator,
    device: torch.device,
    *,
    batch_size: int,
    batches: int = 4,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(batches):
        inputs, labels = data.sample_batch(batch_size)
        active = (inputs != TOKEN_TO_ID["PAD"]).any(dim=0).nonzero().flatten()
        length = int(active[-1]) + 1
        output = model(inputs[:, :length].to(device), labels[:, :length].to(device))
        assert output.loss is not None
        losses.append(float(output.loss.cpu()))
    model.train()
    return float(np.mean(losses))


def train_one(
    size: str,
    seed: int,
    steps: int,
    device: torch.device,
    world_cfg: dict[str, Any],
    output_root: Path,
    *,
    batch_size: int | None = None,
) -> tuple[ReaderModel, dict[str, Any], Path, int]:
    config_path = REPO_ROOT / "configs" / f"reader_{size}.yaml"
    reader_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if batch_size is not None:
        reader_cfg["batch_size"] = batch_size
    run_dir = output_root / f"{size}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.pt"

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model, data, optimizer = build_training(reader_cfg, world_cfg, seed, device)
    validation_cfg = dict(world_cfg)
    validation_cfg["phase"] = "reader_train"
    validation_world = make_world((500_000 + seed) % 1_000_000, validation_cfg)
    validation_data = ReaderDataGenerator((validation_world,), reader_cfg, seed + 10_000)
    parameter_count = count_parameters(model)
    completed = 0
    losses: list[float] = []
    validation_losses: list[tuple[int, float]] = []
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["size"] != size or checkpoint["seed"] != seed:
            raise ValueError(f"checkpoint identity mismatch at {checkpoint_path}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        completed = int(checkpoint["completed_steps"])
        losses = list(checkpoint["losses"])
        validation_losses = list(checkpoint.get("validation_losses", []))
        data.rng.bit_generator.state = checkpoint["data_rng_state"]
        if checkpoint.get("validation_rng_state") is not None:
            validation_data.rng.bit_generator.state = checkpoint["validation_rng_state"]
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        print(f"resume {size} seed={seed} at step {completed}", flush=True)

    checkpoint_interval = 1000
    while completed < steps:
        chunk_size = min(checkpoint_interval - completed % checkpoint_interval, steps - completed)
        chunk_losses, _ = run_steps(
            model,
            data,
            optimizer,
            reader_cfg,
            device,
            chunk_size,
            progress_every=checkpoint_interval,
            start_step=completed,
            total_steps=steps,
        )
        losses.extend(chunk_losses)
        completed += chunk_size
        val_loss = validation_loss(
            model,
            validation_data,
            device,
            batch_size=int(reader_cfg["batch_size"]),
            batches=1 if batch_size is not None else 4,
        )
        validation_losses.append((completed, val_loss))
        checkpoint = {
            "size": size,
            "seed": seed,
            "completed_steps": completed,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "losses": losses,
            "validation_losses": validation_losses,
            "validation_rng_state": validation_data.rng.bit_generator.state,
            "data_rng_state": data.rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }
        _atomic_save(checkpoint, checkpoint_path)
        print(
            f"checkpoint {size} seed={seed} step={completed} val_loss={val_loss:.6f}",
            flush=True,
        )

    with (run_dir / "losses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("step", "loss"))
        writer.writerows(enumerate(losses, start=1))
    with (run_dir / "validation_losses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("step", "loss"))
        writer.writerows(validation_losses)
    model.eval()
    return model, reader_cfg, run_dir, parameter_count


def _input_for_query(
    world: World, query: Query, condition: str, context_length: int
) -> tuple[int, ...]:
    context: list[str] = []
    if condition == "full_oracle":
        for episode_id in query.evidence_episode_ids:
            context.extend(render_l4(world.episodes[episode_id]))
    elif condition != "none":
        raise ValueError(f"unknown calibration condition {condition}")
    prefix = ["BOS", *context, *query.tokens]
    # The model's configured context is the physical ceiling on oracle inputs.
    if len(prefix) > context_length:
        prefix = prefix[-context_length:]
    return tuple(TOKEN_TO_ID[token] for token in prefix)


@torch.inference_mode()
def evaluate_condition(
    model: ReaderModel,
    world: World,
    condition: str,
    device: torch.device,
    *,
    queries: Sequence[Query] | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    queries = tuple(queries) if queries is not None else world.queries
    inputs = [
        _input_for_query(world, query, condition, model.config.context_length)
        for query in queries
    ]
    predictions: list[str] = []
    model.eval()
    for offset in range(0, len(inputs), batch_size):
        batch = inputs[offset : offset + batch_size]
        max_len = max(map(len, batch))
        tensor = torch.full(
            (len(batch), max_len), TOKEN_TO_ID["PAD"], dtype=torch.long, device=device
        )
        lengths = []
        for row, tokens in enumerate(batch):
            tensor[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
            lengths.append(len(tokens) - 1)
        logits = model(tensor).logits
        row_ids = torch.arange(len(batch), device=device)
        answer_ids = logits[row_ids, torch.tensor(lengths, device=device)].argmax(dim=-1)
        predictions.extend(TOKENS[int(token_id)] for token_id in answer_ids.cpu())

    scores: list[float] = []
    correct: list[int] = []
    confabulation: list[int] = []
    by_type: dict[str, list[tuple[str, str, str]]] = {}
    for query, prediction in zip(queries, predictions, strict=True):
        target = query.answer
        score = 1.0 if prediction == target else (0.0 if prediction == "ABSTAIN" else -1.0)
        scores.append(score)
        correct.append(int(prediction == target))
        confabulation.append(int(prediction != "ABSTAIN" and prediction != target))
        by_type.setdefault(query.query_type, []).append((prediction, target, str(score)))

    # Chance is the expected SPEC utility after shuffling targets within query
    # type, preserving each type's target frequencies and the model's guesses.
    chance_by_type: dict[str, float] = {}
    for query_type, rows in by_type.items():
        n = len(rows)
        pred_counts: dict[str, int] = {}
        target_counts: dict[str, int] = {}
        for prediction, target, _ in rows:
            pred_counts[prediction] = pred_counts.get(prediction, 0) + 1
            target_counts[target] = target_counts.get(target, 0) + 1
        exact_non_abstain_probability = sum(
            pred_counts.get(token, 0) * count for token, count in target_counts.items()
            if token != "ABSTAIN"
        ) / (n * n)
        exact_abstain_probability = (
            pred_counts.get("ABSTAIN", 0) * target_counts.get("ABSTAIN", 0) / (n * n)
        )
        non_abstain_prediction_rate = 1 - pred_counts.get("ABSTAIN", 0) / n
        chance_by_type[query_type] = (
            2 * exact_non_abstain_probability
            + exact_abstain_probability
            - non_abstain_prediction_rate
        )

    type_scores = {
        query_type: float(np.mean([float(row[2]) for row in rows]))
        for query_type, rows in by_type.items()
    }
    chance_utility = sum(
        chance_by_type[query_type] * len(rows) for query_type, rows in by_type.items()
    ) / max(1, len(queries))
    return {
        "query_count": len(queries),
        "n_by_type": {query_type: len(rows) for query_type, rows in by_type.items()},
        "utility": float(np.mean(scores)),
        "accuracy": float(np.mean(correct)),
        "confabulation_rate": float(np.mean(confabulation)),
        "chance_utility": float(chance_utility),
        "chance_by_type": chance_by_type,
        "utility_by_type": type_scores,
    }


def make_eval_worlds(
    base_config: dict[str, Any], *, worlds_per_level: int, smoke: bool
) -> list[tuple[float, World]]:
    result: list[tuple[float, World]] = []
    for level_index, redundancy in enumerate(REDUNDANCY_LEVELS):
        for replicate in range(worlds_per_level):
            config = dict(base_config)
            config["phase"] = "eval"
            config["p_redundancy"] = redundancy
            seed = 2_000_000 + level_index * 100 + replicate
            result.append((redundancy, make_world(seed, config)))
    print(f"generated {len(result)} eval worlds ({'smoke' if smoke else 'full'})", flush=True)
    return result


def stratified_eval_queries(
    world: World,
    *,
    per_type: int = 300,
    seed: int,
) -> tuple[Query, ...]:
    """Select distinct query forms per type, failing if any type is undersupplied."""

    rng = np.random.default_rng(seed)
    selected: list[Query] = []
    for query_type in ("A", "B", "C", "E", "R"):
        unique: dict[tuple[str, ...], Query] = {}
        for query in world.queries:
            if query.query_type == query_type:
                unique.setdefault(query.tokens, query)
        candidates = list(unique.values())
        minimum = min(per_type, 250) if query_type == "R" else per_type
        if len(candidates) < minimum:
            raise RuntimeError(
                f"eval world {world.seed} has {len(candidates)} unique type-{query_type} "
                f"queries; calibration requires at least {minimum}"
            )
        sample_size = min(per_type, len(candidates))
        indices = rng.choice(
            len(candidates),
            size=sample_size,
            replace=False,
        )
        selected.extend(candidates[int(index)] for index in indices)
    return tuple(selected)


def write_outputs(
    raw_rows: list[dict[str, Any]],
    sizes: list[str],
    seeds: list[int],
    parameters: dict[str, int],
    output_dir: Path,
    *,
    smoke: bool,
) -> dict[str, Any]:
    raw_path = output_dir / "calibration_raw.csv"
    columns = [
        "size", "parameters", "seed", "p_redundancy", "world_seed", "condition",
        "query_count", "utility", "accuracy", "confabulation_rate", "chance_utility",
        "n_A", "n_B", "n_C", "n_E", "n_R",
        "unique_n_A", "unique_n_B", "unique_n_C", "unique_n_E", "unique_n_R",
        "utility_A", "utility_B", "utility_C", "utility_E", "utility_R",
    ]
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in raw_rows:
            writer.writerow({key: row.get(key, "") for key in columns})

    summary: list[dict[str, Any]] = []
    for size in sizes:
        size_rows = [row for row in raw_rows if row["size"] == size]
        full = [row for row in size_rows if row["condition"] == "full_oracle"]
        none = [row for row in size_rows if row["condition"] == "none"]
        by_seed: dict[int, list[float]] = {}
        for row in full:
            by_seed.setdefault(int(row["seed"]), []).append(float(row["utility"]))
        seed_utilities = [float(np.mean(values)) for values in by_seed.values()]
        b_c = {
            relation: float(np.mean(values)) if values else math.nan
            for relation, values in {
                relation: [
                row[f"utility_{relation}"]
                for row in full
                if row.get(f"utility_{relation}") not in (None, "")
                ]
                for relation in ("B", "C")
            }.items()
        }
        summary.append({
            "size": size,
            "parameters": parameters[size],
            "none_utility": float(np.mean([row["utility"] for row in none])) if none else math.nan,
            "chance_utility": float(np.mean([row["chance_utility"] for row in none])) if none else math.nan,
            "full_oracle_utility": float(np.mean([row["utility"] for row in full])) if full else math.nan,
            "full_oracle_B": b_c["B"],
            "full_oracle_C": b_c["C"],
            "between_seed_sd": float(np.std(seed_utilities, ddof=1)) if len(seed_utilities) > 1 else math.nan,
            **{
                f"n_{query_type}": sum(int(row.get(f"n_{query_type}", 0)) for row in full)
                for query_type in ("A", "B", "C", "E", "R")
            },
            **{
                f"unique_n_{query_type}": sum(
                    int(row.get(f"unique_n_{query_type}", 0)) for row in full
                )
                for query_type in ("A", "B", "C", "E", "R")
            },
        })

    r31 = next((row for row in summary if row["size"] == "r31m"), None)
    for row in summary:
        complete = r31 is not None and not smoke and len(seeds) > 1
        if complete:
            row["criterion_full_oracle"] = abs(row["full_oracle_utility"] - r31["full_oracle_utility"]) <= 0.02
            row["criterion_type_B"] = abs(row["full_oracle_B"] - r31["full_oracle_B"]) <= 0.02
            row["criterion_type_C"] = abs(row["full_oracle_C"] - r31["full_oracle_C"]) <= 0.02
            row["criterion_seed_sd"] = row["between_seed_sd"] < 0.01
            row["criterion_none_chance"] = abs(row["none_utility"] - row["chance_utility"]) <= 0.05
            row["meets_all"] = all(row[key] for key in (
                "criterion_full_oracle", "criterion_type_B", "criterion_type_C",
                "criterion_seed_sd", "criterion_none_chance",
            ))
        else:
            for key in (
                "criterion_full_oracle", "criterion_type_B", "criterion_type_C",
                "criterion_seed_sd", "criterion_none_chance", "meets_all",
            ):
                row[key] = None

    summary_path = output_dir / "calibration_table.csv"
    summary_columns = list(summary[0]) if summary else []
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        writer.writeheader()
        writer.writerows(summary)

    selected = None
    selection_message = "Selection not estimable from smoke/subset run."
    if not smoke and set(sizes) == set(SIZES) and len(seeds) >= 2 and r31 is not None:
        eligible = [row for row in summary if row["meets_all"]]
        if eligible:
            selected = min(eligible, key=lambda row: SIZES.index(row["size"]))["size"]
            selection_message = f"Selected {selected}; next size up is {SIZES[SIZES.index(selected) + 1] if selected != SIZES[-1] else 'none'}."
        else:
            selection_message = "No reader size meets every preregistered calibration criterion."
    selection = {
        "selection": selected,
        "message": selection_message,
        "rules": CALIBRATION_RULES,
        "summary": summary,
    }
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    x = [parameters[size] / 1e6 for size in sizes]
    for condition, marker in (("none", "o"), ("full_oracle", "s")):
        ys = []
        for size in sizes:
            matching = [row["utility"] for row in raw_rows if row["size"] == size and row["condition"] == condition]
            ys.append(float(np.mean(matching)) if matching else math.nan)
        axis.plot(x, ys, marker=marker, label=condition)
    axis.set(xlabel="Non-embedding parameters (millions)", ylabel="Utility", title="Reader calibration")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "calibration_plot.png", dpi=160)
    plt.close(figure)
    return selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--steps", type=int, default=15_000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "calibration")
    args = parser.parse_args()

    sizes = ["r2m"] if args.smoke else list(args.sizes)
    seeds = [11] if args.smoke else list(args.seeds)
    steps = 200 if args.smoke else args.steps
    device = select_device("cpu" if args.smoke else args.device)
    eval_worlds_per_level = 1 if args.smoke else 2
    per_type = 2 if args.smoke else 300
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    world_cfg = yaml.safe_load((REPO_ROOT / "configs/world.yaml").read_text(encoding="utf-8"))
    model_parameters: dict[str, int] = {}
    raw_rows: list[dict[str, Any]] = []
    worlds = make_eval_worlds(world_cfg, worlds_per_level=eval_worlds_per_level, smoke=args.smoke)
    for size in sizes:
        for seed in seeds:
            print(f"train {size} seed={seed} steps={steps} device={device}", flush=True)
            model, reader_cfg, run_dir, parameter_count = train_one(
                size,
                seed,
                steps,
                device,
                world_cfg,
                output_dir / "runs",
                batch_size=8 if args.smoke else None,
            )
            model_parameters[size] = parameter_count
            torch.save(model.state_dict(), run_dir / "model.pt")
            for redundancy, world in worlds:
                eval_queries = stratified_eval_queries(
                    world,
                    per_type=per_type,
                    seed=world.seed + 1,
                )
                for condition in ("none", "full_oracle"):
                    metrics = evaluate_condition(
                        model,
                        world,
                        condition,
                        device,
                        queries=eval_queries,
                    )
                    raw_rows.append({
                        "size": size,
                        "parameters": parameter_count,
                        "seed": seed,
                        "p_redundancy": redundancy,
                        "world_seed": world.seed,
                        "condition": condition,
                        **metrics,
                        **{
                            f"n_{query_type}": metrics["n_by_type"].get(query_type, 0)
                            for query_type in ("A", "B", "C", "E", "R")
                        },
                        **{
                            f"unique_n_{query_type}": len({
                                query.id for query in eval_queries if query.query_type == query_type
                            })
                            for query_type in ("A", "B", "C", "E", "R")
                        },
                        **{f"utility_{key}": value for key, value in metrics["utility_by_type"].items()},
                    })
                    print(
                        f"eval {size} seed={seed} p={redundancy:.1f} world={world.seed} "
                        f"{condition} U={metrics['utility']:.3f} n={metrics['query_count']} "
                        + " ".join(
                            f"n_{query_type}={metrics['n_by_type'].get(query_type, 0)}"
                            for query_type in ("A", "B", "C", "E", "R")
                        ),
                        flush=True,
                    )
    parameters = model_parameters or {size: 0 for size in sizes}
    selection = write_outputs(raw_rows, sizes, seeds, parameters, output_dir, smoke=args.smoke)
    print(selection["message"])
    print(f"calibration table: {output_dir / 'calibration_table.csv'}")
    print(f"calibration plot: {output_dir / 'calibration_plot.png'}")


if __name__ == "__main__":
    main()
