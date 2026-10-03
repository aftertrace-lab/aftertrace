#!/usr/bin/env python3
"""Time reader data generation and model-update stages separately."""

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

# Disable silent CPU fallback before importing torch on MPS-capable hosts.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aftertrace.reader.data import ReaderDataGenerator
from aftertrace.reader.model import ReaderModel, config_for_size
from aftertrace.reader.train import autocast_context, peak_memory_mb, select_device
from aftertrace.run_metadata import record_run_provenance, require_clean_tree
from aftertrace.world.generator import make_world
from aftertrace.world.vocab import TOKENS


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _seconds_since(start: float, device: torch.device) -> float:
    _sync(device)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/reader_r5m.yaml")
    parser.add_argument("--world-config", type=Path, default=REPO_ROOT / "configs/world.yaml")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-dirty", action="store_true", help="run from a dirty tree and record that provenance")
    args = parser.parse_args()
    provenance = require_clean_tree(REPO_ROOT, allow_dirty=args.allow_dirty)
    run_dir = args.output_dir or REPO_ROOT / "runs" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_bench"
    run_dir.mkdir(parents=True, exist_ok=False)
    record_run_provenance(run_dir, provenance)

    reader_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    world_cfg = yaml.safe_load(args.world_config.read_text(encoding="utf-8"))
    device = select_device(args.device or str(reader_cfg.get("device", "auto")))
    batch_size = args.batch_size or int(reader_cfg["batch_size"])
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    started = time.perf_counter()
    world_cfg["phase"] = "reader_train"
    world = make_world(args.seed, world_cfg)
    world_generation = time.perf_counter() - started

    started = time.perf_counter()
    data = ReaderDataGenerator((world,), reader_cfg, seed=args.seed)
    data_setup = time.perf_counter() - started

    started = time.perf_counter()
    model_cfg = config_for_size(
        str(reader_cfg["name"]), len(TOKENS), int(reader_cfg["context_length"])
    )
    model = ReaderModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(reader_cfg["peak_lr"]),
        betas=(float(reader_cfg["beta1"]), float(reader_cfg["beta2"])),
        weight_decay=float(reader_cfg["weight_decay"]),
    )
    model_setup = _seconds_since(started, device)

    stages = {
        key: []
        for key in ("batch_sampling", "trim_and_transfer", "forward", "backward", "optimizer")
    }
    measured_step_seconds: list[float] = []
    model.train()
    total_steps = args.warmup_steps + args.steps
    for step in range(total_steps):
        whole_start = time.perf_counter()

        stage_start = time.perf_counter()
        input_ids, labels = data.sample_batch(batch_size)
        data_elapsed = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        active = (input_ids != TOKENS.index("PAD")).any(dim=0).nonzero().flatten()
        active_length = int(active[-1]) + 1
        input_ids = input_ids[:, :active_length].to(device)
        labels = labels[:, :active_length].to(device)
        transfer_elapsed = _seconds_since(stage_start, device)

        optimizer.zero_grad(set_to_none=True)
        stage_start = time.perf_counter()
        with autocast_context(device):
            output = model(input_ids, labels)
        forward_elapsed = _seconds_since(stage_start, device)
        assert output.loss is not None

        stage_start = time.perf_counter()
        output.loss.backward()
        backward_elapsed = _seconds_since(stage_start, device)

        stage_start = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(reader_cfg["grad_clip"]))
        optimizer.step()
        optimizer_elapsed = _seconds_since(stage_start, device)
        whole_elapsed = _seconds_since(whole_start, device)

        if step >= args.warmup_steps:
            for key, value in zip(
                stages,
                (data_elapsed, transfer_elapsed, forward_elapsed, backward_elapsed, optimizer_elapsed),
                strict=True,
            ):
                stages[key].append(value)
            measured_step_seconds.append(whole_elapsed)

    print(f"device: {device}")
    print(f"MPS fallback: {os.environ['PYTORCH_ENABLE_MPS_FALLBACK']}")
    print(f"threads: {args.threads}; batch_size: {batch_size}; measured_steps: {args.steps}")
    print("setup seconds:")
    print(f"  world_generation: {world_generation:.3f}")
    print(f"  data_setup_and_summary_cache: {data_setup:.3f}")
    print(f"  model_and_optimizer: {model_setup:.3f}")
    total_measured = sum(measured_step_seconds)
    print("per-step mean seconds (share of measured step time):")
    for key, values in stages.items():
        mean = sum(values) / max(1, len(values))
        print(f"  {key}: {mean:.4f} ({100 * mean / (total_measured / args.steps):.1f}%)")
    mean_step = total_measured / max(1, args.steps)
    print(f"  step_total: {mean_step:.4f} s ({1000 * mean_step:.2f} ms)")
    print(f"steps/sec: {1 / mean_step:.4f}")
    memory = peak_memory_mb(device)
    print(f"peak memory (MiB): {memory:.1f}")
    summary = {
        "device": str(device),
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "batch_size": batch_size,
        "threads": args.threads,
        "steps_per_second": 1 / mean_step,
        "peak_memory_mib": memory,
        "setup_seconds": {
            "world_generation": world_generation,
            "data_setup_and_summary_cache": data_setup,
            "model_and_optimizer": model_setup,
        },
        "mean_step_seconds": {
            key: sum(values) / max(1, len(values)) for key, values in stages.items()
        },
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
