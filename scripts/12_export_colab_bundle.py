#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from scratch_llm.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an inference-only Colab demo bundle.")
    parser.add_argument("--config", default="configs/tinystories_100m_full_ctx512.yaml")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/tinystories_100m_full_ctx512/best.pt",
        help="Training checkpoint to strip down to model weights only.",
    )
    parser.add_argument(
        "--vectors",
        default="benchmarks/results/playful_serious_vectors_100m_full_ctx512.pt",
        help="Activation vector payload to copy into the bundle.",
    )
    parser.add_argument(
        "--output-dir",
        default="colab_bundle/playful_ctx512",
        help="Directory to write model.pt, tokenizer.json, vectors.pt, and manifest.json.",
    )
    parser.add_argument(
        "--dtype",
        choices=["preserve", "float16", "bfloat16", "float32"],
        default="float16",
        help="Storage dtype for floating-point model weights. float16 makes upload smaller.",
    )
    parser.add_argument("--emotion", default="playful")
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--position", choices=["last", "all"], default="all")
    parser.add_argument("--prompt", default="Once upon a time there was a little robot")
    parser.add_argument("--max-new-tokens", type=int, default=350)
    return parser.parse_args()


def convert_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    dtype_name: str,
) -> dict[str, torch.Tensor]:
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name == "preserve":
        return {key: value.detach().cpu() for key, value in state_dict.items()}
    dtype = dtype_map[dtype_name]
    converted = {}
    for key, value in state_dict.items():
        tensor = value.detach().cpu()
        converted[key] = tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor
    return converted


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    vectors_path = Path(args.vectors)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not vectors_path.exists():
        raise FileNotFoundError(f"Vector payload not found: {vectors_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_payload: dict[str, Any] = {
        "model": convert_state_dict(checkpoint["model"], dtype_name=args.dtype),
        "model_args": checkpoint["model_args"],
        "source_checkpoint": str(checkpoint_path),
        "source_iter_num": checkpoint.get("iter_num"),
        "source_best_val_loss": checkpoint.get("best_val_loss"),
        "export_dtype": args.dtype,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": float(config.get("generation", {}).get("temperature", 0.8)),
            "top_k": int(config.get("generation", {}).get("top_k", 100)),
            "stop_at_eos": bool(config.get("generation", {}).get("stop_at_eos", True)),
        },
    }
    model_output = output_dir / "model.pt"
    torch.save(model_payload, model_output)

    tokenizer_source = Path(config["paths"]["tokenizer"])
    tokenizer_output = output_dir / "tokenizer.json"
    shutil.copy2(tokenizer_source, tokenizer_output)

    vectors_output = output_dir / "vectors.pt"
    shutil.copy2(vectors_path, vectors_output)

    manifest = {
        "model": "model.pt",
        "tokenizer": "tokenizer.json",
        "vectors": "vectors.pt",
        "config": str(args.config),
        "checkpoint": str(checkpoint_path),
        "vector_source": str(vectors_path),
        "default_demo": {
            "emotion": args.emotion,
            "layer": args.layer,
            "alpha": args.alpha,
            "position": args.position,
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
        },
        "model_args": checkpoint["model_args"],
        "source_iter_num": checkpoint.get("iter_num"),
        "source_best_val_loss": checkpoint.get("best_val_loss"),
        "export_dtype": args.dtype,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"wrote {model_output}")
    print(f"wrote {tokenizer_output}")
    print(f"wrote {vectors_output}")
    print(f"wrote {output_dir / 'manifest.json'}")
    print(f"bundle_dir={output_dir}")


if __name__ == "__main__":
    main()
