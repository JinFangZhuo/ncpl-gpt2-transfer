"""PyTorch implementation of Marin/Levanter's Cautious optimizer."""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


class Cautious(Optimizer):
    """Adam-style update masked by gradient/update agreement.

    This follows ``levanter.optim.cautious.scale_by_cautious``: compute the
    bias-corrected Adam update, retain elements whose update agrees in sign
    with the current gradient, normalize the mask by its per-tensor mean, add
    decoupled weight decay, then apply the scheduled learning rate.
    """

    def __init__(self, params, lr=1e-3, betas=(0.95, 0.95), eps=1e-8):
        if lr < 0:
            raise ValueError(f"invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"invalid betas: {betas}")
        if eps < 0:
            raise ValueError(f"invalid epsilon: {eps}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group.get("weight_decay", 0.0)
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("Cautious does not support sparse gradients")

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                adam_update = (exp_avg / bias_correction1) / (
                    (exp_avg_sq / bias_correction2).sqrt().add_(eps)
                )
                mask = (adam_update * gradient > 0).to(adam_update.dtype)
                cautious_update = adam_update * mask / (mask.mean() + eps)
                if weight_decay:
                    cautious_update.add_(parameter, alpha=weight_decay)
                parameter.add_(cautious_update, alpha=-lr)

        return loss
