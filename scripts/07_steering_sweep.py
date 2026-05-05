#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from scratch_llm.activations import parse_layers, register_residual_steering_hook
from scratch_llm.config import load_config
from scratch_llm.experiment_manifest import (
    AtomicTextFile,
    ensure_outputs_available,
    git_commit,
    manifest_path_for_output,
    slurm_job_id,
    utc_timestamp,
    write_json_atomic,
)
from scratch_llm.generation import EOS_TOKEN, resolve_eos_token_id
from scratch_llm.model import TransformerConfig, TransformerLM
from scratch_llm.tokenizer import ScratchTokenizer

DEFAULT_PROMPTS = [
    "Once upon a time there was a little robot",
    "Lily found a strange machine in the garden",
    "The little dog was scared of the dark",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a structured activation steering sweep.")
    parser.add_argument("--config", default="configs/tinystories_60m_overnight.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--vectors", default="benchmarks/results/emotion_vectors_60m.pt")
    parser.add_argument(
        "--output",
        default="benchmarks/results/steering_sweep_60m.jsonl",
        help="JSONL output path.",
    )
    parser.add_argument("--emotions", default="happy,sad,scared,calm,curious")
    parser.add_argument("--layers", default="6,9,11")
    parser.add_argument("--alphas", default="2,3")
    parser.add_argument("--positions", default="last,all")
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--stop-at-eos",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Stop generation after sampling <|endoftext|>. Defaults to "
            "generation.stop_at_eos when configured, otherwise false."
        ),
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing sweep output and manifest.",
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


def parse_csv_strings(spec: str) -> list[str]:
    values = [item.strip() for item in spec.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return values


def parse_csv_floats(spec: str) -> list[float]:
    return [float(item) for item in parse_csv_strings(spec)]


def seed_everything(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


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
    stop_token_id: int | None,
) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    idx = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        stop_token_id=stop_token_id,
    )
    return tokenizer.decode(out[0].tolist())


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    manifest_path = manifest_path_for_output(output_path)
    ensure_outputs_available([output_path, manifest_path], overwrite=args.overwrite)

    config = load_config(args.config)
    checkpoint_path = args.checkpoint or f"{config['paths']['checkpoint_dir']}/ckpt.pt"
    device = resolve_device(config["system"].get("device", "auto"))

    tokenizer = ScratchTokenizer.from_file(config["paths"]["tokenizer"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TransformerLM(TransformerConfig(**checkpoint["model_args"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    vector_payload = torch.load(args.vectors, map_location="cpu")
    emotions = parse_csv_strings(args.emotions)
    layers = parse_layers(args.layers, num_layers=model.config.num_layers)
    alphas = parse_csv_floats(args.alphas)
    positions = parse_csv_strings(args.positions)
    prompts = args.prompt or DEFAULT_PROMPTS

    generation_cfg = config["generation"]
    temperature = (
        args.temperature if args.temperature is not None else float(generation_cfg["temperature"])
    )
    top_k = args.top_k if args.top_k is not None else int(generation_cfg["top_k"])
    stop_at_eos = (
        args.stop_at_eos
        if args.stop_at_eos is not None
        else bool(generation_cfg.get("stop_at_eos", False))
    )
    stop_token_id = resolve_eos_token_id(tokenizer) if stop_at_eos else None
    if stop_at_eos and stop_token_id is None:
        raise ValueError(f"Could not resolve {EOS_TOKEN!r} to a single token id.")
    parameter_count = model.parameter_count()

    manifest = {
        "timestamp": utc_timestamp(),
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
        "slurm_job_id": slurm_job_id(),
        "config": args.config,
        "config_values": config,
        "checkpoint": checkpoint_path,
        "vectors": args.vectors,
        "output": str(output_path),
        "manifest": str(manifest_path),
        "layers": layers,
        "alphas": alphas,
        "positions": positions,
        "emotions": emotions,
        "prompts": prompts,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "seed": args.seed,
            "stop_at_eos": stop_at_eos,
            "stop_token_id": stop_token_id,
        },
        "parameter_count": parameter_count,
        "device": str(device),
        "requested_device": config["system"].get("device", "auto"),
    }

    baseline_cache: dict[tuple[int, str], str] = {}
    with AtomicTextFile(output_path, overwrite=args.overwrite) as handle:
        for prompt_idx, prompt in enumerate(prompts):
            seed = args.seed + prompt_idx
            seed_everything(seed, device)
            baseline = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                stop_token_id=stop_token_id,
            )
            baseline_cache[(prompt_idx, prompt)] = baseline

        total = len(prompts) * len(emotions) * len(layers) * len(alphas) * len(positions)
        completed = 0
        for emotion in emotions:
            for layer in layers:
                vector = vector_payload["vectors"][emotion][layer]
                for alpha in alphas:
                    for position in positions:
                        for prompt_idx, prompt in enumerate(prompts):
                            seed = args.seed + prompt_idx
                            seed_everything(seed, device)
                            handle_obj = register_residual_steering_hook(
                                model=model,
                                layer=layer,
                                vector=vector,
                                alpha=alpha,
                                position=position,
                            )
                            start = time.time()
                            try:
                                steered = generate_text(
                                    model=model,
                                    tokenizer=tokenizer,
                                    prompt=prompt,
                                    device=device,
                                    max_new_tokens=args.max_new_tokens,
                                    temperature=temperature,
                                    top_k=top_k,
                                    stop_token_id=stop_token_id,
                                )
                            finally:
                                handle_obj.remove()
                            if device.type == "cuda":
                                torch.cuda.synchronize()
                            elapsed_s = time.time() - start
                            record = {
                                "emotion": emotion,
                                "layer": layer,
                                "alpha": alpha,
                                "position": position,
                                "prompt_idx": prompt_idx,
                                "prompt": prompt,
                                "seed": seed,
                                "baseline_text": baseline_cache[(prompt_idx, prompt)],
                                "steered_text": steered,
                                "elapsed_s": elapsed_s,
                                "max_new_tokens": args.max_new_tokens,
                                "temperature": temperature,
                                "top_k": top_k,
                                "stop_at_eos": stop_at_eos,
                                "config": args.config,
                                "checkpoint": checkpoint_path,
                                "vectors": args.vectors,
                                "parameters": parameter_count,
                            }
                            handle.write(json.dumps(record) + "\n")
                            handle.flush()
                            completed += 1
                            print(
                                f"{completed}/{total} emotion={emotion} layer={layer} "
                                f"alpha={alpha} position={position} prompt={prompt_idx}"
                            )

    write_json_atomic(manifest, manifest_path, overwrite=args.overwrite)
    print(f"wrote {output_path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
