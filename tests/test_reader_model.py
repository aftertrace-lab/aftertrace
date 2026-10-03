from __future__ import annotations

import pytest
import torch
import math

from aftertrace.reader.model import (
    CausalSelfAttention,
    ReaderConfig,
    ReaderModel,
    config_for_size,
    count_parameters,
)
from aftertrace.world.vocab import TOKENS


@pytest.mark.parametrize(
    ("name", "target"),
    (("r2m", 1.8e6), ("r5m", 4.7e6), ("r14m", 14e6), ("r31m", 31e6)),
)
def test_reader_parameter_counts_are_within_spec(name: str, target: float) -> None:
    model = ReaderModel(config_for_size(name, len(TOKENS)))
    count = count_parameters(model)
    assert abs(count - target) / target < 0.15


def test_reader_forward_ties_embeddings_and_masks_non_answers() -> None:
    config = config_for_size("r2m", len(TOKENS), context_length=16)
    model = ReaderModel(config)
    input_ids = torch.randint(0, len(TOKENS), (2, 16))
    labels = torch.full_like(input_ids, -100)
    labels[:, 12] = input_ids[:, 12]
    output = model(input_ids, labels)
    assert output.logits.shape == (2, 16, len(TOKENS))
    assert output.loss is not None and torch.isfinite(output.loss)
    assert model.output_weight.data_ptr() == model.token_embedding.weight.data_ptr()


def test_sdpa_matches_previous_causal_attention_on_fixed_cpu_batch() -> None:
    torch.manual_seed(314159)
    config = ReaderConfig(
        vocab_size=len(TOKENS), n_layers=1, d_model=48, n_heads=4, ffn_dim=96,
        context_length=16,
    )
    attention = CausalSelfAttention(config).cpu().eval()
    inputs = torch.randn(2, 13, config.d_model, device="cpu")

    with torch.inference_mode():
        actual = attention(inputs)
        batch, length, width = inputs.shape
        qkv = attention.qkv(inputs).view(
            batch, length, 3, attention.n_heads, attention.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query, key = attention.rope(query), attention.rope(key)
        scores = query @ key.transpose(-2, -1) / math.sqrt(attention.head_dim)
        causal_mask = torch.ones(length, length, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        previous = (weights @ value).transpose(1, 2).contiguous().view(batch, length, width)
        expected = attention.output(previous)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
