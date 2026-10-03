#!/usr/bin/env python3
"""Run the requested reader timing benchmark and save its loss curve."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/aftertrace-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aftertrace.reader.data import ReaderDataGenerator
from aftertrace.reader.model import ReaderModel, config_for_size, count_parameters
from aftertrace.run_metadata import record_run_provenance, require_clean_tree
from aftertrace.world.generator import make_world
from aftertrace.world.vocab import TOKENS


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1024**2
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024**2 if platform.system() == "Darwin" else 1024
    return maximum_rss / divisor


def visible_tokens(tokens: tuple[str, ...]) -> str:
    return " ".join(token for token in tokens if token != "PAD")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=("r2m", "r5m", "r14m", "r31m"), default="r5m")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-dirty", action="store_true", help="run from a dirty tree and record that provenance")
    args = parser.parse_args()
    provenance = require_clean_tree(REPO_ROOT, allow_dirty=args.allow_dirty)

    with (REPO_ROOT / "configs" / "world.yaml").open(encoding="utf-8") as handle:
        world_config = yaml.safe_load(handle)
    with (REPO_ROOT / "configs" / f"reader_{args.size}.yaml").open(encoding="utf-8") as handle:
        reader_config = yaml.safe_load(handle)
    world_config["phase"] = "reader_train"
    batch_size = args.batch_size or int(reader_config["batch_size"])

    torch.manual_seed(args.seed)
    torch.set_num_threads(min(os.cpu_count() or 1, 8))
    device = select_device()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    world = make_world(args.seed, world_config)
    data = ReaderDataGenerator((world,), reader_config, seed=args.seed)
    model_config = config_for_size(
        args.size, len(TOKENS), int(reader_config["context_length"])
    )
    model = ReaderModel(model_config).to(device)
    parameter_count = count_parameters(model)
    if args.compile:
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(reader_config["peak_lr"]),
        betas=(float(reader_config["beta1"]), float(reader_config["beta2"])),
        weight_decay=float(reader_config["weight_decay"]),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or REPO_ROOT / "runs" / f"{timestamp}_{args.size}_timing"
    output_dir.mkdir(parents=True, exist_ok=False)
    resolved = {
        "reader": reader_config,
        "world": world_config,
        "steps": args.steps,
        "batch_size": batch_size,
        "seed": args.seed,
        "device": str(device),
    }
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=True)
    record_run_provenance(output_dir, provenance)
    (output_dir / "seeds.json").write_text(
        json.dumps({"run_seed": args.seed, "world_seed": args.seed}, indent=2) + "\n",
        encoding="utf-8",
    )

    pad_id = TOKENS.index("PAD")
    loss_tensors: list[torch.Tensor] = []
    lengths: list[int] = []
    start = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        input_ids, labels = data.sample_batch(batch_size)
        active_columns = (input_ids != pad_id).any(dim=0).nonzero().flatten()
        active_length = int(active_columns[-1]) + 1
        input_ids = input_ids[:, :active_length].to(device)
        labels = labels[:, :active_length].to(device)
        optimizer.zero_grad(set_to_none=True)
        learning_rate = float(reader_config["peak_lr"]) * min(
            1.0, step / int(reader_config["warmup_steps"])
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(input_ids, labels)
        assert output.loss is not None
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(reader_config["grad_clip"]))
        optimizer.step()
        loss_tensors.append(output.loss.detach())
        lengths.append(active_length)
        if step == 1 or step % 10 == 0 or step == args.steps:
            elapsed = time.perf_counter() - start
            print(
                f"step={step} loss={float(loss_tensors[-1].float().cpu()):.6f} "
                f"steps_per_sec={step / elapsed:.4f} active_length={active_length}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    losses = torch.stack(loss_tensors).float().cpu().tolist()
    training_peak_memory_mb = peak_memory_mb(device)

    metrics = pd.DataFrame({
        "step": range(1, args.steps + 1),
        "loss": losses,
        "active_length": lengths,
    })
    metrics.to_csv(output_dir / "losses.csv", index=False)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(metrics["step"], metrics["loss"], linewidth=1)
    axis.set(xlabel="Step", ylabel="Answer-token loss", title=f"{args.size} timing run")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close(figure)

    examples = {
        "l1_gist": visible_tokens(data.sample_example(force_format="l1").tokens),
        "residual": visible_tokens(data.sample_example(force_format="residual").tokens),
        "abstain": visible_tokens(data.sample_example(force_abstain=True).tokens),
    }
    (output_dir / "examples.json").write_text(
        json.dumps(examples, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "size": args.size,
        "steps": args.steps,
        "batch_size": batch_size,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "steps_per_second": args.steps / elapsed,
        "seconds_per_1000_steps": elapsed * 1000 / args.steps,
        "peak_memory_mb": training_peak_memory_mb,
        "non_embedding_parameters": parameter_count,
        "mean_active_length": sum(lengths) / len(lengths),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
