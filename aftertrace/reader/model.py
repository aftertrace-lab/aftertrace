"""Small decoder-only readers with RMSNorm, RoPE, and tied embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ReaderConfig:
    vocab_size: int
    n_layers: int
    d_model: int
    n_heads: int
    ffn_dim: int
    context_length: int = 384

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE head dimension must be even")


SIZE_CONFIGS = {
    "r2m": dict(n_layers=4, d_model=192, n_heads=6, ffn_dim=768),
    "r5m": dict(n_layers=6, d_model=256, n_heads=8, ffn_dim=1024),
    "r14m": dict(n_layers=8, d_model=384, n_heads=8, ffn_dim=1536),
    "r31m": dict(n_layers=10, d_model=512, n_heads=8, ffn_dim=2048),
}


def config_for_size(name: str, vocab_size: int, context_length: int = 384) -> ReaderConfig:
    if name not in SIZE_CONFIGS:
        raise ValueError(f"unknown reader size {name!r}")
    return ReaderConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        **SIZE_CONFIGS[name],
    )


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = inputs.float() * torch.rsqrt(
            inputs.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(inputs.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_length: int) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            10_000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_length, dtype=torch.float32)
        angles = torch.outer(positions, inverse_frequency)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        length = tensor.shape[-2]
        cos = self.cos[:length].to(dtype=tensor.dtype)[None, None, :, :]
        sin = self.sin[:length].to(dtype=tensor.dtype)[None, None, :, :]
        even, odd = tensor[..., 0::2], tensor[..., 1::2]
        return torch.stack(
            (even * cos - odd * sin, odd * cos + even * sin), dim=-1
        ).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ReaderConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.context_length)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, length, width = inputs.shape
        qkv = self.qkv(inputs).view(
            batch, length, 3, self.n_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query = self.rope(query)
        key = self.rope(key)
        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=True
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(attended)


class FeedForward(nn.Module):
    def __init__(self, config: ReaderConfig) -> None:
        super().__init__()
        self.up = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.down = nn.Linear(config.ffn_dim, config.d_model, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # DECISION: Use GELU for the unspecified FFN nonlinearity; the two
        # projection matrices preserve the calibration-table parameter counts.
        return self.down(F.gelu(self.up(inputs), approximate="tanh"))


class DecoderBlock(nn.Module):
    def __init__(self, config: ReaderConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.feed_forward = FeedForward(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.attention(self.attention_norm(inputs))
        return inputs + self.feed_forward(self.ffn_norm(inputs))


@dataclass
class ReaderOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class ReaderModel(nn.Module):
    def __init__(self, config: ReaderConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            DecoderBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.d_model)
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def output_weight(self) -> nn.Parameter:
        return self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> ReaderOutput:
        if input_ids.shape[1] > self.config.context_length:
            raise ValueError("input exceeds configured context length")
        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        logits = F.linear(self.final_norm(hidden), self.output_weight)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.config.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return ReaderOutput(logits=logits, loss=loss)


def count_parameters(model: ReaderModel, *, exclude_embeddings: bool = True) -> int:
    """Count unique trainable parameters, optionally excluding token embeddings."""

    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not exclude_embeddings or name != "token_embedding.weight"
    )
