from __future__ import annotations

import torch
import torch.distributed as dist


class SimpleDDP(torch.nn.Module):
    """Small custom DDP wrapper matching the Assignment 2 direction.

    It broadcasts initial parameters from rank 0 and averages gradients after backward.
    The later systems milestone is to replace post-backward synchronization with async
    per-parameter hooks that overlap communication with backpropagation.
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        if dist.is_available() and dist.is_initialized():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        world_size = dist.get_world_size()
        for param in self.module.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)


def get_ddp(module: torch.nn.Module) -> torch.nn.Module:
    return SimpleDDP(module)


def ddp_on_after_backward(ddp_model: torch.nn.Module, _optimizer: torch.optim.Optimizer) -> None:
    ddp_model.finish_gradient_synchronization()


def get_fsdp(
    _module: torch.nn.Module,
    _compute_dtype: torch.dtype | None = None,
) -> torch.nn.Module:
    raise NotImplementedError("Custom FSDP is an Assignment 2 milestone.")


def fsdp_on_after_backward(
    _fsdp_model: torch.nn.Module,
    _optimizer: torch.optim.Optimizer,
) -> None:
    raise NotImplementedError("Custom FSDP is an Assignment 2 milestone.")


def fsdp_gather_full_params(_fsdp_model: torch.nn.Module) -> dict[str, torch.Tensor]:
    raise NotImplementedError("Custom FSDP is an Assignment 2 milestone.")


def get_sharded_optimizer(
    _params,
    _optimizer_cls: type[torch.optim.Optimizer],
    **_kwargs,
) -> torch.optim.Optimizer:
    raise NotImplementedError("Sharded optimizer is an Assignment 2 milestone.")
