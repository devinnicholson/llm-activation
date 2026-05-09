#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scratch_llm.config import load_config
from scratch_llm.data import load_token_array
from scratch_llm.model import TransformerConfig, TransformerLM
from scratch_llm.tokenizer import ScratchTokenizer
from scratch_llm.train import autocast_context, estimate_loss, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one checkpoint with stable settings.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=4242)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(args.seed)

    device = resolve_device(config["system"].get("device", "auto"))
    train_data = load_token_array(config["paths"]["train_data"])
    val_data = load_token_array(config["paths"]["val_data"])
    tokenizer = ScratchTokenizer.from_file(config["paths"]["tokenizer"])

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = TransformerLM(TransformerConfig(**checkpoint["model_args"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    train_cfg = dict(config["training"])
    train_cfg["eval_iters"] = args.eval_iters
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size

    dtype_name = config["system"].get("dtype", "float32")
    ctx = autocast_context(device, dtype_name)
    losses = estimate_loss(
        model=model,
        train_data=train_data,
        val_data=val_data,
        train_cfg=train_cfg,
        context_length=model.config.context_length,
        device=device,
        ctx=ctx,
    )

    print(f"config={args.config}")
    print(f"checkpoint={args.checkpoint}")
    print(f"checkpoint_iter={checkpoint.get('iter_num')}")
    print(f"checkpoint_best_val_loss={checkpoint.get('best_val_loss')}")
    print(f"parameters={model.parameter_count():,}")
    print(f"eval_iters={args.eval_iters}")
    print(f"batch_size={train_cfg['batch_size']}")
    print(f"seed={args.seed}")
    print(f"train_loss={losses['train']:.6f}")
    print(f"val_loss={losses['val']:.6f}")


if __name__ == "__main__":
    main()
