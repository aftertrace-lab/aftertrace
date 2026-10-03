"""Config-driven reader training, also runnable as a source-checkout module."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Mapping

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aftertrace.reader.data import ReaderDataGenerator
from aftertrace.reader.model import ReaderModel, config_for_size
from aftertrace.run_metadata import record_run_provenance, require_clean_tree
from aftertrace.world.generator import make_world
from aftertrace.world.vocab import TOKENS


def select_device(requested: str = "auto") -> torch.device:
    """Resolve auto/cpu/cuda/mps and reject unavailable explicit devices."""

    if requested == "auto":
        requested = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    return torch.device(requested)


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1024**2
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**2 if platform.system() == "Darwin" else 1024)


def autocast_context(device: torch.device):
    """Use bf16 autocast only on CUDA; keep CPU and MPS training in fp32."""

    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )


def build_training(
    reader_cfg: Mapping[str, Any], world_cfg: Mapping[str, Any], seed: int, device: torch.device
) -> tuple[ReaderModel, ReaderDataGenerator, torch.optim.Optimizer]:
    cfg = dict(world_cfg)
    cfg["phase"] = "reader_train"
    world = make_world(seed, cfg)
    data = ReaderDataGenerator((world,), reader_cfg, seed)
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
    return model, data, optimizer


def run_steps(
    model: ReaderModel,
    data: ReaderDataGenerator,
    optimizer: torch.optim.Optimizer,
    cfg: Mapping[str, Any],
    device: torch.device,
    steps: int,
    *,
    progress_every: int = 0,
    start_step: int = 0,
    total_steps: int | None = None,
) -> tuple[list[float], float]:
    losses: list[float] = []
    pending_losses: list[torch.Tensor] = []
    start = time.perf_counter()
    for local_step in range(1, steps + 1):
        step = start_step + local_step
        input_ids, labels = data.sample_batch(int(cfg["batch_size"]))
        active = (input_ids != TOKENS.index("PAD")).any(dim=0).nonzero().flatten()
        length = int(active[-1]) + 1
        input_ids, labels = input_ids[:, :length].to(device), labels[:, :length].to(device)
        optimizer.zero_grad(set_to_none=True)
        warmup = int(cfg["warmup_steps"])
        warmup_scale = min(1.0, step / warmup)
        if total_steps is not None and step > warmup and total_steps > warmup:
            progress = min(1.0, (step - warmup) / (total_steps - warmup))
            decay_scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            decay_scale = 1.0
        lr = float(cfg["peak_lr"]) * warmup_scale * decay_scale
        for group in optimizer.param_groups:
            group["lr"] = lr
        with autocast_context(device):
            output = model(input_ids, labels)
        assert output.loss is not None
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
        optimizer.step()
        pending_losses.append(output.loss.detach())
        should_log = progress_every and (
            step == 1 or step % progress_every == 0 or local_step == steps
        )
        if should_log:
            losses.extend(torch.stack(pending_losses).float().cpu().tolist())
            pending_losses.clear()
            elapsed = time.perf_counter() - start
            print(
                f"step={step} loss={losses[-1]:.6f} "
                f"steps/sec={local_step / elapsed:.4f}",
                flush=True,
            )
    if pending_losses:
        losses.extend(torch.stack(pending_losses).float().cpu().tolist())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    return losses, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/reader_r5m.yaml")
    parser.add_argument("--world-config", type=Path, default=REPO_ROOT / "configs/world.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "runs/checkpoints")
    parser.add_argument("--allow-dirty", action="store_true", help="run from a dirty tree and record that provenance")
    args = parser.parse_args()
    provenance = require_clean_tree(REPO_ROOT, allow_dirty=args.allow_dirty)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    record_run_provenance(args.checkpoint_dir, provenance)
    reader_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    world_cfg = yaml.safe_load(args.world_config.read_text(encoding="utf-8"))
    device = select_device(args.device or str(reader_cfg.get("device", "auto")))
    torch.manual_seed(args.seed)
    model, data, optimizer = build_training(reader_cfg, world_cfg, args.seed, device)
    losses, elapsed = run_steps(
        model, data, optimizer, reader_cfg, device, args.steps, progress_every=10
    )
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": args.steps},
        args.checkpoint_dir / f"{reader_cfg['name']}_step_{args.steps}.pt",
    )
    print(
        f"device={device} steps/sec={args.steps / elapsed:.4f} "
        f"peak_memory_mb={peak_memory_mb(device):.1f} final_loss={losses[-1]:.6f}"
    )


if __name__ == "__main__":
    main()
