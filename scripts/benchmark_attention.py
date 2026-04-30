#!/usr/bin/env python
from __future__ import annotations

import argparse
import time

import torch

from scratch_llm.systems.flash_attention import FlashAttentionPytorch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark torch SDPA vs the project PyTorch FlashAttention."
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def torch_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)


def timed(fn, params: tuple[torch.Tensor, ...], *, device: torch.device, iters: int) -> float:
    for _ in range(3):
        for param in params:
            param.grad = None
        out = fn()
        out.sum().backward()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        for param in params:
            param.grad = None
        out = fn()
        out.sum().backward()
    synchronize(device)
    return (time.perf_counter() - start) / iters


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    shape = (args.batch_size, args.heads, args.seq_len, args.head_dim)

    def make_inputs():
        q = torch.randn(shape, device=device, requires_grad=True)
        k = torch.randn(shape, device=device, requires_grad=True)
        v = torch.randn(shape, device=device, requires_grad=True)
        return q, k, v

    q, k, v = make_inputs()
    sdpa_s = timed(
        lambda: torch_sdpa(q, k, v),
        (q, k, v),
        device=device,
        iters=args.iters,
    )

    q, k, v = make_inputs()
    flash_s = timed(
        lambda: FlashAttentionPytorch.apply(q, k, v, True),
        (q, k, v),
        device=device,
        iters=args.iters,
    )

    print(f"device={device}")
    print(f"shape={shape}")
    print(f"torch_sdpa_forward_backward_s={sdpa_s:.6f}")
    print(f"flash_pytorch_forward_backward_s={flash_s:.6f}")


if __name__ == "__main__":
    main()
