from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.utils.hooks import RemovableHandle

from scratch_llm.model import TransformerLM
from scratch_llm.tokenizer import TokenizerLike

PoolingMode = str


def parse_layers(spec: str, *, num_layers: int) -> list[int]:
    if spec == "all":
        return list(range(num_layers))

    layers: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        layer = int(item)
        if layer < 0:
            layer = num_layers + layer
        if not 0 <= layer < num_layers:
            raise ValueError(f"Layer {item} is outside [0, {num_layers}).")
        layers.append(layer)

    if not layers:
        raise ValueError("At least one layer is required.")
    return sorted(set(layers))


def _pool_activation(activation: torch.Tensor, pooling: PoolingMode) -> torch.Tensor:
    if activation.ndim != 3:
        raise ValueError(
            "Expected activation with shape [batch, seq, d_model], "
            f"got {activation.shape}."
        )
    if pooling == "mean":
        return activation.mean(dim=(0, 1))
    if pooling == "last":
        return activation[:, -1, :].mean(dim=0)
    raise ValueError(f"Unknown pooling mode: {pooling}")


@torch.no_grad()
def mean_layer_activations(
    *,
    model: TransformerLM,
    tokenizer: TokenizerLike,
    prompts: Iterable[str],
    layers: list[int],
    device: torch.device,
    pooling: PoolingMode = "mean",
) -> dict[int, torch.Tensor]:
    sums = {layer: torch.zeros(model.config.d_model, dtype=torch.float32) for layer in layers}
    counts = {layer: 0 for layer in layers}
    captures: dict[int, torch.Tensor] = {}
    handles: list[RemovableHandle] = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            captures[layer_idx] = output.detach().float().cpu()

        return hook

    for layer in layers:
        handles.append(model.layers[layer].register_forward_hook(make_hook(layer)))

    try:
        model.eval()
        for prompt in prompts:
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            if not ids:
                continue
            idx = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
            captures.clear()
            model(idx)
            for layer in layers:
                if layer not in captures:
                    raise RuntimeError(f"Layer {layer} was not captured.")
                sums[layer] += _pool_activation(captures[layer], pooling)
                counts[layer] += 1
    finally:
        for handle in handles:
            handle.remove()

    means = {}
    for layer in layers:
        if counts[layer] == 0:
            raise ValueError("No non-empty prompts were provided.")
        means[layer] = sums[layer] / counts[layer]
    return means


def normalize_vector(vector: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    return vector / vector.norm().clamp_min(eps)


def build_contrast_vectors(
    *,
    model: TransformerLM,
    tokenizer: TokenizerLike,
    prompt_bank: dict[str, list[str]],
    emotions: list[str],
    baseline_key: str,
    layers: list[int],
    device: torch.device,
    pooling: PoolingMode = "mean",
    normalize: bool = True,
) -> dict[str, dict[int, torch.Tensor]]:
    if baseline_key not in prompt_bank:
        raise KeyError(f"Missing baseline prompt key: {baseline_key}")

    baseline = mean_layer_activations(
        model=model,
        tokenizer=tokenizer,
        prompts=prompt_bank[baseline_key],
        layers=layers,
        device=device,
        pooling=pooling,
    )

    vectors: dict[str, dict[int, torch.Tensor]] = {}
    for emotion in emotions:
        if emotion not in prompt_bank:
            raise KeyError(f"Missing emotion prompt key: {emotion}")
        emotion_means = mean_layer_activations(
            model=model,
            tokenizer=tokenizer,
            prompts=prompt_bank[emotion],
            layers=layers,
            device=device,
            pooling=pooling,
        )
        vectors[emotion] = {}
        for layer in layers:
            vector = emotion_means[layer] - baseline[layer]
            vectors[emotion][layer] = normalize_vector(vector) if normalize else vector
    return vectors


def register_residual_steering_hook(
    *,
    model: TransformerLM,
    layer: int,
    vector: torch.Tensor,
    alpha: float,
    position: str = "last",
) -> RemovableHandle:
    if not 0 <= layer < len(model.layers):
        raise ValueError(f"Layer {layer} is outside [0, {len(model.layers)}).")

    def hook(_module, _inputs, output):
        steer = vector.to(device=output.device, dtype=output.dtype).view(1, 1, -1)
        if position == "last":
            steered = output.clone()
            steered[:, -1:, :] = steered[:, -1:, :] + alpha * steer
            return steered
        if position == "all":
            return output + alpha * steer
        raise ValueError(f"Unknown steering position: {position}")

    return model.layers[layer].register_forward_hook(hook)
