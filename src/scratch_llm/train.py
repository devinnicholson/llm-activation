from __future__ import annotations

import argparse
import csv
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume model, optimizer, and scaler state from a checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path to save to, and to load from when --resume is set.",
    )
    parser.add_argument(
        "--metrics-path",
        default=None,
        help="CSV path for training/evaluation metrics. Defaults to checkpoint_dir/metrics.csv.",
    )
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
    scaler: torch.amp.GradScaler,
    iteration: int,
    best_val_loss: float,
    config: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    checkpoint = {
        "model": raw_model.state_dict(),
        "model_args": raw_model.config_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "iter_num": iteration,
        "best_val_loss": best_val_loss,
        "config": config,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    *,
    checkpoint_path: Path,
    raw_model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    raw_model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["iter_num"]) + 1, float(checkpoint.get("best_val_loss", "inf"))


def resolve_best_checkpoint_path(
    checkpoint_dir: Path,
    train_cfg: dict[str, Any],
) -> Path | None:
    if not bool(train_cfg.get("save_best_checkpoint", False)):
        return None
    return checkpoint_dir / str(train_cfg.get("best_checkpoint_name", "best.pt"))


METRIC_FIELDS = [
    "elapsed_s",
    "iter",
    "event",
    "loss",
    "train_loss",
    "val_loss",
    "lr",
    "tokens_per_s",
    "tokens_per_step",
    "world_size",
    "parameters",
]


def prepare_metrics_file(metrics_path: Path, *, append: bool) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if append and metrics_path.exists():
        return
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()


def write_metrics_row(metrics_path: Path, row: dict[str, int | float | str | None]) -> None:
    with metrics_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})


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
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else checkpoint_dir / "ckpt.pt"
    best_checkpoint_path = resolve_best_checkpoint_path(checkpoint_dir, train_cfg)
    metrics_path = Path(args.metrics_path) if args.metrics_path else checkpoint_dir / "metrics.csv"
    if master_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        prepare_metrics_file(metrics_path, append=args.resume)
        print(f"device={device} world_size={world_size}")
        print(f"parameters={raw_model.parameter_count():,}")
        print(f"train_tokens={len(train_data):,} val_tokens={len(val_data):,}")

    start_iter = 0
    best_val_loss = float("inf")
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Cannot resume; checkpoint not found: {checkpoint_path}")
        start_iter, best_val_loss = load_checkpoint(
            checkpoint_path=checkpoint_path,
            raw_model=raw_model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        if master_process:
            print(f"resumed_from={checkpoint_path} start_iter={start_iter}")

    x, y = get_batch(
        train_data,
        batch_size=int(train_cfg["batch_size"]),
        context_length=transformer_config.context_length,
        device=device,
    )
    last_log_time = time.time()
    train_start_time = last_log_time
    last_iter = start_iter - 1
    last_log_iter = start_iter - 1

    for iter_num in range(start_iter, int(train_cfg["max_iters"])):
        last_iter = iter_num
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
            val_improved = losses["val"] < best_val_loss
            best_val_loss = min(best_val_loss, losses["val"])
            write_metrics_row(
                metrics_path,
                {
                    "elapsed_s": time.time() - train_start_time,
                    "iter": iter_num,
                    "event": "eval",
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "lr": lr,
                    "world_size": world_size,
                    "parameters": raw_model.parameter_count(),
                },
            )
            should_save = val_improved or bool(train_cfg.get("always_save_checkpoint", False))
            if best_checkpoint_path is not None and val_improved:
                save_checkpoint(
                    raw_model=raw_model,
                    optimizer=optimizer,
                    scaler=scaler,
                    iteration=iter_num,
                    best_val_loss=best_val_loss,
                    config=config,
                    checkpoint_path=best_checkpoint_path,
                )
                print(f"saved_best_checkpoint={best_checkpoint_path} iter={iter_num}")
            if should_save:
                save_checkpoint(
                    raw_model=raw_model,
                    optimizer=optimizer,
                    scaler=scaler,
                    iteration=iter_num,
                    best_val_loss=best_val_loss,
                    config=config,
                    checkpoint_path=checkpoint_path,
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
            elapsed = time.time() - last_log_time
            steps_since_log = max(iter_num - last_log_iter, 1)
            tokens_per_step = (
                int(train_cfg["batch_size"])
                * transformer_config.context_length
                * int(train_cfg["gradient_accumulation_steps"])
                * world_size
            )
            tokens = tokens_per_step * steps_since_log
            reported_loss = loss.item() * int(train_cfg["gradient_accumulation_steps"])
            print(
                f"iter={iter_num} loss={reported_loss:.4f} "
                f"lr={lr:.2e} tokens_per_s={tokens / max(elapsed, 1e-9):.1f}"
            )
            write_metrics_row(
                metrics_path,
                {
                    "elapsed_s": time.time() - train_start_time,
                    "iter": iter_num,
                    "event": "train",
                    "loss": reported_loss,
                    "lr": lr,
                    "tokens_per_s": tokens / max(elapsed, 1e-9),
                    "tokens_per_step": tokens_per_step,
                    "world_size": world_size,
                    "parameters": raw_model.parameter_count(),
                },
            )
            last_log_time = time.time()
            last_log_iter = iter_num

    if master_process and last_iter >= start_iter:
        save_checkpoint(
            raw_model=raw_model,
            optimizer=optimizer,
            scaler=scaler,
            iteration=last_iter,
            best_val_loss=best_val_loss,
            config=config,
            checkpoint_path=checkpoint_path,
        )
        print(f"saved_final_checkpoint={checkpoint_path} iter={last_iter}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
