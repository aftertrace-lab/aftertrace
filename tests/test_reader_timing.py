from __future__ import annotations

import math

import pytest
import torch

from aftertrace.reader.model import ReaderConfig, ReaderModel
from aftertrace.reader.train import run_steps
from aftertrace.world.vocab import TOKENS


class FixedBatch:
    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.randint(0, len(TOKENS), (batch_size, 12))
        labels = torch.full_like(inputs, -100)
        labels[:, -1] = inputs[:, -1]
        return inputs, labels


def _timed_training_step(device: torch.device) -> tuple[torch.dtype, float]:
    config = ReaderConfig(
        vocab_size=len(TOKENS), n_layers=1, d_model=48, n_heads=4, ffn_dim=96,
        context_length=16,
    )
    model = ReaderModel(config).to(device)
    observed_logits: list[torch.dtype] = []
    model.register_forward_hook(
        lambda _module, _inputs, output: observed_logits.append(output.logits.dtype)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_config = {
        "batch_size": 2,
        "warmup_steps": 1,
        "peak_lr": 1e-3,
        "grad_clip": 1.0,
    }
    losses, elapsed = run_steps(
        model,
        FixedBatch(),  # type: ignore[arg-type]
        optimizer,
        train_config,
        device,
        1,
        progress_every=1,
    )
    assert len(losses) == 1 and math.isfinite(losses[0])
    assert elapsed > 0
    assert len(observed_logits) == 1
    return observed_logits[0], elapsed


def test_cpu_timing_path_keeps_fp32() -> None:
    logits_dtype, _elapsed = _timed_training_step(torch.device("cpu"))
    assert logits_dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_timing_path_uses_bf16_autocast() -> None:
    logits_dtype, elapsed = _timed_training_step(torch.device("cuda"))
    assert logits_dtype == torch.bfloat16
    assert elapsed > 0
