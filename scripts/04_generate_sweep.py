#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from scratch_llm.config import load_config
from scratch_llm.model import TransformerConfig, TransformerLM
from scratch_llm.tokenizer import ScratchTokenizer

MODEL_REGISTRY = {
    "8m": {
        "config": "configs/tinystories_8m.yaml",
        "checkpoint": "checkpoints/tinystories_8m/ckpt.pt",
    },
    "24m": {
        "config": "configs/tinystories_24m.yaml",
        "checkpoint": "checkpoints/tinystories_24m/ckpt.pt",
    },
    "60m": {
        "config": "configs/tinystories_60m_overnight.yaml",
        "checkpoint": "checkpoints/tinystories_60m_overnight/ckpt.pt",
    },
}

DEFAULT_PROMPTS = [
    "Once upon a time there was a little robot",
    "Lily found a strange machine in the garden",
    "Tom wanted to help his friend, but",
    "The little dog was scared of the dark",
    "Mia learned that sharing was important because",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comparable samples from checkpoints.")
    parser.add_argument(
        "--models",
        default="8m,24m,60m",
        help="Comma-separated model keys from: 8m,24m,60m",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results/generation_sweep.jsonl",
        help="JSONL output path.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt to include. Repeat for multiple prompts. Defaults to a fixed prompt set.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def load_model(
    config_path: str,
    checkpoint_path: str,
) -> tuple[dict, ScratchTokenizer, TransformerLM, torch.device]:
    config = load_config(config_path)
    device = resolve_device(config["system"].get("device", "auto"))
    tokenizer = ScratchTokenizer.from_file(config["paths"]["tokenizer"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TransformerLM(TransformerConfig(**checkpoint["model_args"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return config, tokenizer, model, device


def main() -> None:
    args = parse_args()
    requested_models = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = sorted(set(requested_models) - set(MODEL_REGISTRY))
    if unknown:
        raise ValueError(f"Unknown model keys: {', '.join(unknown)}")

    prompts = args.prompt or DEFAULT_PROMPTS
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for model_key in requested_models:
            entry = MODEL_REGISTRY[model_key]
            checkpoint_path = Path(entry["checkpoint"])
            if not checkpoint_path.exists():
                print(f"Skipping {model_key}: missing {checkpoint_path}")
                continue

            print(f"Loading {model_key}: {checkpoint_path}")
            config, tokenizer, model, device = load_model(entry["config"], str(checkpoint_path))
            generation_cfg = config["generation"]
            max_new_tokens = args.max_new_tokens or int(generation_cfg["max_new_tokens"])
            temperature = args.temperature or float(generation_cfg["temperature"])
            top_k = args.top_k if args.top_k is not None else int(generation_cfg["top_k"])
            parameter_count = model.parameter_count()

            for prompt_idx, prompt in enumerate(prompts):
                torch.manual_seed(args.seed + prompt_idx)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(args.seed + prompt_idx)
                    torch.cuda.synchronize()
                start = time.time()
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                idx = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
                with torch.no_grad():
                    out = model.generate(
                        idx,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed_s = time.time() - start
                text = tokenizer.decode(out[0].tolist())
                record = {
                    "model": model_key,
                    "parameters": parameter_count,
                    "checkpoint": str(checkpoint_path),
                    "prompt": prompt,
                    "text": text,
                    "elapsed_s": elapsed_s,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_k": top_k,
                    "device": str(device),
                }
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                print(f"\n[{model_key}] {prompt}\n{text}\n")

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
