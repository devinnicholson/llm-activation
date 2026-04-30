from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScalingModelConfig:
    name: str
    vocab_size: int
    seq_len: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    head_dim: int
    intermediate_size: int

    def validate(self) -> None:
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.intermediate_size < self.hidden_size:
            raise ValueError("intermediate_size should usually be >= hidden_size")

    @property
    def parameter_estimate(self) -> int:
        embedding = self.vocab_size * self.hidden_size
        attention = self.num_layers * 4 * self.hidden_size * self.hidden_size
        mlp = self.num_layers * 3 * self.hidden_size * self.intermediate_size
        norms = self.num_layers * 2 * self.hidden_size + self.hidden_size
        lm_head = self.vocab_size * self.hidden_size
        return embedding + attention + mlp + norms + lm_head


@dataclass(frozen=True)
class ScalingRunConfig:
    model: ScalingModelConfig
    train_batch_size: int
    total_train_tokens: int
    max_runtime_seconds: float

    @property
    def tokens_per_optimizer_step(self) -> int:
        return self.model.seq_len * self.train_batch_size

    @property
    def total_optimizer_steps(self) -> int:
        if self.total_train_tokens % self.tokens_per_optimizer_step != 0:
            raise ValueError("total_train_tokens must divide seq_len * train_batch_size")
        return self.total_train_tokens // self.tokens_per_optimizer_step

    @property
    def approximate_training_flops(self) -> int:
        # Common rough estimate for dense decoder-only training.
        return 6 * self.model.parameter_estimate * self.total_train_tokens


MODEL_SWEEP = [
    ScalingModelConfig(
        name="tiny",
        vocab_size=32000,
        seq_len=512,
        hidden_size=256,
        num_layers=6,
        num_attention_heads=4,
        head_dim=64,
        intermediate_size=768,
    ),
    ScalingModelConfig(
        name="small",
        vocab_size=32000,
        seq_len=512,
        hidden_size=448,
        num_layers=9,
        num_attention_heads=7,
        head_dim=64,
        intermediate_size=1280,
    ),
    ScalingModelConfig(
        name="base",
        vocab_size=32000,
        seq_len=512,
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        head_dim=64,
        intermediate_size=3072,
    ),
]
