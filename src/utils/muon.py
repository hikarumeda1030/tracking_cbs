"""Muon optimizer implementation.

Muon applies Nesterov momentum in the spectral domain by orthogonalizing
the gradient via Newton-Schulz iterations. 2D+ parameters (weight matrices)
are updated with Muon; 1D parameters (biases, norms) use AdamW.

Reference: Keller Jordan et al., "Muon: An optimizer for hidden layers in
neural networks" (2024). https://github.com/KellerJordan/Muon
"""
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power (orthogonalization) of G.
    Operates in bfloat16 for efficiency, then converts back to original dtype.
    """
    assert G.ndim == 2, f"Expected 2D tensor, got {G.ndim}D"
    a, b, c = 3.4445, -4.7750, 2.0315
    orig_dtype = G.dtype
    X = G.to(torch.bfloat16)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    # Normalize so spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(orig_dtype)


class Muon(Optimizer):
    """
    Muon optimizer for weight matrices (2D+ parameters).

    Args:
        params: iterable of 2D+ tensors (weight matrices)
        lr: learning rate (default: 0.02)
        momentum: Nesterov momentum coefficient (default: 0.95)
        nesterov: whether to use Nesterov-style update (default: True)
        ns_steps: number of Newton-Schulz iterations (default: 5)
    """
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                # Nesterov: use g + momentum * buf
                update = g.add(buf, alpha=momentum) if nesterov else buf.clone()

                # Orthogonalize: flatten to 2D, apply NS, reshape back
                orig_shape = update.shape
                update_2d = update.reshape(orig_shape[0], -1)
                update_2d = zeropower_via_newtonschulz5(update_2d, steps=ns_steps)
                update = update_2d.reshape(orig_shape)

                p.add_(update, alpha=-lr)


def make_muon_optimizers(model, cfg, config_module):
    """
    Split model parameters and create (muon_optimizer, adamw_optimizer).

    Muon handles ndim >= 2 params; AdamW handles the rest.
    Config keys (all under 'train'):
        lr             - Muon learning rate (required)
        muon_momentum  - Nesterov momentum (default: 0.95)
        muon_ns_steps  - Newton-Schulz steps (default: 5)
        adamw_lr       - AdamW lr for 1D params (default: 3e-4)
        adamw_beta1    - (default: 0.95)
        adamw_beta2    - (default: 0.95)
        adamw_eps      - (default: 1e-8)
        adamw_wd       - weight decay for AdamW (default: 0.0)
    """
    muon_params = [p for p in model.parameters() if p.ndim >= 2]
    adamw_params = [p for p in model.parameters() if p.ndim < 2]

    lr = config_module.require(cfg, "train.lr")
    momentum = config_module.optional(cfg, "train.muon_momentum", 0.95)
    ns_steps = config_module.optional(cfg, "train.muon_ns_steps", 5)

    adamw_lr = config_module.optional(cfg, "train.adamw_lr", 3e-4)
    adamw_beta1 = config_module.optional(cfg, "train.adamw_beta1", 0.95)
    adamw_beta2 = config_module.optional(cfg, "train.adamw_beta2", 0.95)
    adamw_eps = config_module.optional(cfg, "train.adamw_eps", 1e-8)
    adamw_wd = config_module.optional(cfg, "train.adamw_wd", 0.0)

    muon_opt = Muon(muon_params, lr=lr, momentum=momentum, ns_steps=ns_steps)
    adamw_opt = torch.optim.AdamW(
        adamw_params,
        lr=adamw_lr,
        betas=(adamw_beta1, adamw_beta2),
        eps=adamw_eps,
        weight_decay=adamw_wd,
    )
    return muon_opt, adamw_opt
