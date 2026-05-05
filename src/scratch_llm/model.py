from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from scratch_llm.systems.flash_attention import FlashAttentionPytorch


@dataclass
class TransformerConfig:
    vocab_size: int
    context_length: int
    num_layers: int
    num_heads: int
    d_model: int
    d_ff: int
    rope_theta: float = 10000.0
    dropout: float = 0.0
    attention_impl: str = "manual"


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x_fp32 = x.float()
        inv_rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_fp32 * inv_rms * self.weight).to(original_dtype)


class RoPE(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE dimension must be even.")
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", torch.cos(freqs), persistent=False)
        self.register_buffer("sin", torch.sin(freqs), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = x.shape[-2]
        if token_positions is None:
            cos = self.cos[:seq_len]
            sin = self.sin[:seq_len]
            while cos.ndim < x.ndim:
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)
        else:
            cos = self.cos[token_positions]
            sin = self.sin[token_positions]
            if x.ndim == 4 and cos.ndim == 3:
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=dim, keepdim=True)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    max_logits = logits.max(dim=-1, keepdim=True).values
    logsumexp = max_logits + torch.log(torch.exp(logits - max_logits).sum(dim=-1, keepdim=True))
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1))
    return (logsumexp - target_logits).mean()


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)
    return softmax(scores, dim=-1) @ v


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.num_heads = config.num_heads
        self.d_head = config.d_model // config.num_heads
        self.attention_impl = config.attention_impl
        self.q_proj = Linear(config.d_model, config.d_model)
        self.k_proj = Linear(config.d_model, config.d_model)
        self.v_proj = Linear(config.d_model, config.d_model)
        self.output_proj = Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.rope = RoPE(self.d_head, config.rope_theta, config.context_length)
        causal = torch.tril(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool)
        )
        self.register_buffer(
            "causal_mask",
            causal.view(1, 1, config.context_length, config.context_length),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        if self.attention_impl == "flash_pytorch":
            y = FlashAttentionPytorch.apply(q, k, v, True)
        elif self.attention_impl == "manual":
            y = scaled_dot_product_attention(q, k, v, mask=mask)
        else:
            raise ValueError(f"Unknown attention implementation: {self.attention_impl}")
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.dropout(self.output_proj(y))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.d_ff, config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_final = RMSNorm(config.d_model)
        self.lm_head = Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.shape
        if seq_len > self.config.context_length:
            raise ValueError(
                f"Cannot forward sequence length {seq_len}; "
                f"context length is {self.config.context_length}."
            )

        token_positions = torch.arange(seq_len, device=idx.device).unsqueeze(0).expand_as(idx)
        x = self.token_embeddings(idx)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        loss = (
            cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            if targets is not None
            else None
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        stop_token_id: int | None = None,
        stop_token_ids: int | Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if stop_token_id is not None:
            if stop_token_ids is not None:
                raise ValueError("Pass either stop_token_id or stop_token_ids, not both.")
            stop_token_ids = stop_token_id

        stop_ids: torch.Tensor | None = None
        if stop_token_ids is not None:
            if isinstance(stop_token_ids, torch.Tensor):
                stop_ids = stop_token_ids.to(device=idx.device, dtype=idx.dtype).flatten()
            elif isinstance(stop_token_ids, int):
                stop_ids = torch.tensor([stop_token_ids], device=idx.device, dtype=idx.dtype)
            else:
                stop_ids = torch.tensor(list(stop_token_ids), device=idx.device, dtype=idx.dtype)
            if stop_ids.numel() == 0:
                stop_ids = None

        done = (
            torch.zeros(idx.size(0), device=idx.device, dtype=torch.bool)
            if stop_ids is not None
            else None
        )
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.context_length
                else idx[:, -self.config.context_length :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature <= 0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < values[:, [-1]]] = -torch.inf
                probs = softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            if done is not None and done.any():
                idx_next = idx_next.clone()
                idx_next[done] = stop_ids[0]
            idx = torch.cat((idx, idx_next), dim=1)
            if done is not None:
                done |= (idx_next == stop_ids.view(1, -1)).any(dim=1)
                if done.all():
                    break
        return idx

    def parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def config_dict(self) -> dict[str, int | float | str]:
        return asdict(self.config)
