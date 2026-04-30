#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch

from scratch_llm.config import load_config
from scratch_llm.model import TransformerConfig, TransformerLM
from scratch_llm.tokenizer import ScratchTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from a trained scratch GPT checkpoint."
    )
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default="High performance computing")
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
    checkpoint_path = args.checkpoint or f"{config['paths']['checkpoint_dir']}/ckpt.pt"
    device = resolve_device(config["system"].get("device", "auto"))

    tokenizer = ScratchTokenizer.from_file(config["paths"]["tokenizer"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TransformerLM(TransformerConfig(**checkpoint["model_args"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    idx = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=int(config["generation"]["max_new_tokens"]),
            temperature=float(config["generation"]["temperature"]),
            top_k=int(config["generation"]["top_k"]),
        )
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
