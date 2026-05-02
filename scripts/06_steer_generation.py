#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch

from scratch_llm.activations import register_residual_steering_hook
from scratch_llm.config import load_config
from scratch_llm.model import TransformerConfig, TransformerLM
from scratch_llm.tokenizer import ScratchTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate with residual-stream emotion steering.")
    parser.add_argument("--config", default="configs/tinystories_60m_overnight.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--vectors", default="benchmarks/results/emotion_vectors_60m.pt")
    parser.add_argument("--emotion", default="happy")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--position", choices=["last", "all"], default="last")
    parser.add_argument("--prompt", default="Once upon a time there was a little robot")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


@torch.no_grad()
def generate_text(
    *,
    model: TransformerLM,
    tokenizer: ScratchTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    idx = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.decode(out[0].tolist())


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    config = load_config(args.config)
    checkpoint_path = args.checkpoint or f"{config['paths']['checkpoint_dir']}/ckpt.pt"
    device = resolve_device(config["system"].get("device", "auto"))

    tokenizer = ScratchTokenizer.from_file(config["paths"]["tokenizer"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TransformerLM(TransformerConfig(**checkpoint["model_args"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    layer = args.layer if args.layer >= 0 else model.config.num_layers + args.layer
    vectors = torch.load(args.vectors, map_location="cpu")
    vector = vectors["vectors"][args.emotion][layer]

    generation_cfg = config["generation"]
    max_new_tokens = args.max_new_tokens or int(generation_cfg["max_new_tokens"])
    temperature = args.temperature or float(generation_cfg["temperature"])
    top_k = args.top_k if args.top_k is not None else int(generation_cfg["top_k"])

    baseline = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    torch.manual_seed(args.seed)
    handle = register_residual_steering_hook(
        model=model,
        layer=layer,
        vector=vector,
        alpha=args.alpha,
        position=args.position,
    )
    try:
        steered = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    finally:
        handle.remove()

    print(f"emotion={args.emotion} layer={layer} alpha={args.alpha} position={args.position}")
    print("\n[baseline]\n")
    print(baseline)
    print("\n[steered]\n")
    print(steered)


if __name__ == "__main__":
    main()
