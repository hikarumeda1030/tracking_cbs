import torch.optim as optim
from . import config


def exp_growth_lr_lambda(steps, exp_rate):
    return exp_rate ** steps


def lr_scheduler(
        optimizer,
        cfg: dict,
) -> None:
    scheduler = config.require(cfg, "train.lr_scheduler")
    if scheduler == "constant":
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    elif scheduler == "cosine":
        T_max = config.require(cfg, "train.epochs")
        eta_min = cfg.get("train", {}).get("lr_min", 0.0)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
        )
    elif scheduler == "exp_growth":
        exp_rate = config.require(cfg, "train.lr_exp_rate")
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda steps: exp_growth_lr_lambda(steps, exp_rate=exp_rate))
    else:
        raise ValueError(f"Unknown learning rate method: {scheduler}")

    return lr_scheduler
