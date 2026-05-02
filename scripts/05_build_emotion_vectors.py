#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scratch_llm.activations import build_contrast_vectors, parse_layers
from scratch_llm.config import load_config
from scratch_llm.model import TransformerConfig, TransformerLM
from scratch_llm.tokenizer import ScratchTokenizer

PROMPT_BANK = {
    "neutral": [
        "Once there was a child walking through the room.",
        "A small animal sat near the tree and looked around.",
        "The toy was on the table while the sun was outside.",
        "A person opened the door and saw a box.",
        "The day started and everyone went to the park.",
    ],
    "happy": [
        "The little girl smiled because she felt happy and loved.",
        "The child laughed with joy after finding a wonderful toy.",
        "Everyone cheered and felt bright, warm, and excited.",
        "The puppy wagged its tail because it was so glad.",
        "The friends danced together and had a very happy day.",
    ],
    "sad": [
        "The little boy felt sad because his favorite toy was gone.",
        "The child cried softly and missed her friend.",
        "Everyone felt lonely when the rain ruined the party.",
        "The puppy looked down because it felt sad and forgotten.",
        "The girl sat quietly with tears in her eyes.",
    ],
    "scared": [
        "The child was scared when the dark room made a loud sound.",
        "The little dog trembled because it heard thunder outside.",
        "Everyone felt afraid when the shadow moved near the door.",
        "The boy hid under the blanket because he was frightened.",
        "The girl held her toy tightly and felt very scared.",
    ],
    "calm": [
        "The child felt calm while sitting beside the quiet lake.",
        "The puppy rested softly and breathed slowly.",
        "Everyone spoke gently and the room felt peaceful.",
        "The girl smiled calmly and listened to the wind.",
        "The boy felt safe, quiet, and relaxed.",
    ],
    "curious": [
        "The child felt curious and wanted to know what was inside.",
        "The little robot wondered how the strange machine worked.",
        "Everyone asked questions about the shiny new box.",
        "The girl looked closely because she wanted to learn more.",
        "The boy explored the garden with curious eyes.",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build contrastive emotion activation vectors.")
    parser.add_argument("--config", default="configs/tinystories_60m_overnight.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="benchmarks/results/emotion_vectors_60m.pt")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--emotions", default="happy,sad,scared,calm,curious")
    parser.add_argument("--baseline", default="neutral")
    parser.add_argument("--pooling", choices=["mean", "last"], default="mean")
    parser.add_argument("--no-normalize", action="store_true")
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

    layers = parse_layers(args.layers, num_layers=model.config.num_layers)
    emotions = [item.strip() for item in args.emotions.split(",") if item.strip()]
    vectors = build_contrast_vectors(
        model=model,
        tokenizer=tokenizer,
        prompt_bank=PROMPT_BANK,
        emotions=emotions,
        baseline_key=args.baseline,
        layers=layers,
        device=device,
        pooling=args.pooling,
        normalize=not args.no_normalize,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vectors": vectors,
        "config": args.config,
        "checkpoint": checkpoint_path,
        "layers": layers,
        "emotions": emotions,
        "baseline": args.baseline,
        "pooling": args.pooling,
        "normalized": not args.no_normalize,
        "d_model": model.config.d_model,
        "num_layers": model.config.num_layers,
        "parameter_count": model.parameter_count(),
        "prompt_bank": PROMPT_BANK,
    }
    torch.save(payload, output_path)
    print(f"wrote {output_path}")
    print(f"layers={layers}")
    print(f"emotions={emotions}")


if __name__ == "__main__":
    main()
