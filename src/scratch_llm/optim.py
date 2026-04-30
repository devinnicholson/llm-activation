from __future__ import annotations

import math
from collections.abc import Iterable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0:
            raise ValueError("lr must be non-negative.")
        if eps < 0:
            raise ValueError("eps must be non-negative.")
        beta1, beta2 = betas
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("betas must be in [0, 1).")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                param.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-step_size)
                if weight_decay != 0:
                    param.add_(param, alpha=-lr * weight_decay)
        return loss


def cosine_lr(
    *,
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)


def clip_grad_norm(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    params = [param for param in parameters if param.grad is not None]
    if not params:
        return
    total = torch.sqrt(sum(torch.sum(param.grad.detach() ** 2) for param in params))
    if total > max_l2_norm:
        scale = max_l2_norm / (total + 1e-6)
        for param in params:
            param.grad.mul_(scale)
