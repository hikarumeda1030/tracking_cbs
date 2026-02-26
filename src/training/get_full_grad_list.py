import torch
import torch.nn as nn
from tqdm import tqdm


@torch.no_grad()
def _count_seen(y):
    return int(y.numel())


def get_full_grad_list(model, trainloader, device="cuda"):
    params = [p for p in model.parameters() if p.requires_grad]
    was_training = model.training
    model.eval()

    grads_acc = [torch.zeros_like(p, dtype=torch.float32, device=device) for p in params]
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    seen = 0
    try:
        for x, y in tqdm(trainloader, total=len(trainloader), desc="Full grad (train empirical loss)", leave=False):
            x, y = x.to(device), y.to(device)
            seen += _count_seen(y)

            logits = model(x)
            loss = loss_fn(logits, y)

            g_list = torch.autograd.grad(
                loss, params,
                retain_graph=False, create_graph=False,
                allow_unused=True
            )
            for gacc, g in zip(grads_acc, g_list):
                if g is not None:
                    gacc.add_(g.detach().float())
    finally:
        model.train(was_training)

    # average over training samples (empirical risk)
    inv_seen = 1.0 / float(seen)
    total_sq = torch.zeros((), device=device)
    for g in grads_acc:
        g.mul_(inv_seen)
        total_sq.add_(g.square().sum())

    return total_sq.sqrt().item()
