from __future__ import annotations

import argparse
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from scratch_llm.config import load_config
from scratch_llm.data import load_token_array
from scratch_llm.model import TransformerConfig, TransformerLM
from scratch_llm.optim import AdamW, clip_grad_norm, cosine_lr
from scratch_llm.tokenizer import ScratchTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer LM from scratch.")
    parser.add_argument("--config", default="configs/tiny.yaml")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed(requested_device: str) -> tuple[bool, int, int, int, torch.device]:
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        return True, rank, local_rank, world_size, device

    if requested_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(requested_device)
    return False, 0, 0, 1, device


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "float32":
        return nullcontext()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def get_batch(
    data: np.ndarray,
    *,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - context_length - 1, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(np.array(data[i : i + context_length], dtype=np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy(np.array(data[i + 1 : i + 1 + context_length], dtype=np.int64))
            for i in ix
        ]
    )
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    *,
    model: TransformerLM | DDP,
    train_data: np.ndarray,
    val_data: np.ndarray,
    train_cfg: dict[str, Any],
    context_length: int,
    device: torch.device,
    ctx,
) -> dict[str, float]:
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(int(train_cfg["eval_iters"]))
        for idx in range(int(train_cfg["eval_iters"])):
            x, y = get_batch(
                data,
                batch_size=int(train_cfg["batch_size"]),
                context_length=context_length,
                device=device,
            )
            with ctx:
                _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("Model returned no loss during evaluation.")
            losses[idx] = loss.item()
        out[split] = float(losses.mean())
    model.train()
    return out


def configure_optimizer(model: TransformerLM, train_cfg: dict[str, Any]) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        betas=(float(train_cfg["beta1"]), float(train_cfg["beta2"])),
        weight_decay=float(train_cfg["weight_decay"]),
    )


def save_checkpoint(
    *,
    raw_model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    best_val_loss: float,
    config: dict[str, Any],
    checkpoint_dir: Path,
) -> None:
    checkpoint = {
        "model": raw_model.state_dict(),
        "model_args": raw_model.config_dict(),
        "optimizer": optimizer.state_dict(),
        "iter_num": iteration,
        "best_val_loss": best_val_loss,
        "config": config,
    }
    torch.save(checkpoint, checkpoint_dir / "ckpt.pt")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    system_cfg = config["system"]
    train_cfg = config["training"]
    paths = config["paths"]

    ddp, rank, local_rank, world_size, device = setup_distributed(system_cfg.get("device", "auto"))
    master_process = rank == 0
    seed_everything(int(config["project"].get("seed", 1337)) + rank)

    train_data = load_token_array(paths["train_data"])
    val_data = load_token_array(paths["val_data"])
    tokenizer = ScratchTokenizer.from_file(paths["tokenizer"])

    model_cfg = dict(config["model"])
    if model_cfg["vocab_size"] == "auto":
        model_cfg["vocab_size"] = tokenizer.vocab_size
    transformer_config = TransformerConfig(**model_cfg)
    model = TransformerLM(transformer_config).to(device)

    raw_model = model
    if bool(system_cfg.get("compile", False)):
        model = torch.compile(model)

    if ddp:
        device_ids = [local_rank] if device.type == "cuda" else None
        model = DDP(model, device_ids=device_ids)

    optimizer = configure_optimizer(raw_model, train_cfg)
    dtype_name = system_cfg.get("dtype", "float32")
    ctx = autocast_context(device, dtype_name)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and dtype_name == "float16"),
    )

    checkpoint_dir = Path(paths["checkpoint_dir"])
    if master_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"device={device} world_size={world_size}")
        print(f"parameters={raw_model.parameter_count():,}")
        print(f"train_tokens={len(train_data):,} val_tokens={len(val_data):,}")

    x, y = get_batch(
        train_data,
        batch_size=int(train_cfg["batch_size"]),
        context_length=transformer_config.context_length,
        device=device,
    )
    t0 = time.time()
    best_val_loss = float("inf")

    for iter_num in range(int(train_cfg["max_iters"])):
        lr = cosine_lr(
            it=iter_num,
            max_learning_rate=float(train_cfg["learning_rate"]),
            min_learning_rate=float(train_cfg["min_lr"]),
            warmup_iters=int(train_cfg["warmup_iters"]),
            cosine_cycle_iters=int(train_cfg["lr_decay_iters"]),
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % int(train_cfg["eval_interval"]) == 0 and master_process:
            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                train_cfg=train_cfg,
                context_length=transformer_config.context_length,
                device=device,
                ctx=ctx,
            )
            print(
                f"step={iter_num} train_loss={losses['train']:.4f} "
                f"val_loss={losses['val']:.4f}"
            )
            should_save = losses["val"] < best_val_loss or bool(
                train_cfg.get("always_save_checkpoint", False)
            )
            if should_save:
                best_val_loss = losses["val"]
                save_checkpoint(
                    raw_model=raw_model,
                    optimizer=optimizer,
                    iteration=iter_num,
                    best_val_loss=best_val_loss,
                    config=config,
                    checkpoint_dir=checkpoint_dir,
                )

        for micro_step in range(int(train_cfg["gradient_accumulation_steps"])):
            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == int(train_cfg["gradient_accumulation_steps"]) - 1
                )
            with ctx:
                _, loss = model(x, y)
                if loss is None:
                    raise RuntimeError("Model returned no loss during training.")
                loss = loss / int(train_cfg["gradient_accumulation_steps"])

            x, y = get_batch(
                train_data,
                batch_size=int(train_cfg["batch_size"]),
                context_length=transformer_config.context_length,
                device=device,
            )
            scaler.scale(loss).backward()

        if float(train_cfg["grad_clip"]) != 0.0:
            scaler.unscale_(optimizer)
            clip_grad_norm(model.parameters(), float(train_cfg["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if iter_num % int(train_cfg["log_interval"]) == 0 and master_process:
            elapsed = time.time() - t0
            tokens = (
                int(train_cfg["batch_size"])
                * transformer_config.context_length
                * int(train_cfg["gradient_accumulation_steps"])
                * world_size
            )
            reported_loss = loss.item() * int(train_cfg["gradient_accumulation_steps"])
            print(
                f"iter={iter_num} loss={reported_loss:.4f} "
                f"lr={lr:.2e} tokens_per_s={tokens / max(elapsed, 1e-9):.1f}"
            )
            t0 = time.time()

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
