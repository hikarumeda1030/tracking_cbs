from tqdm import tqdm
import torch
from .get_full_grad_list import get_full_grad_list


def _get_optimizers(optimizer):
    """optimizerがtupleの場合はリストに、単体の場合はそのままリストに変換する。"""
    return list(optimizer) if isinstance(optimizer, tuple) else [optimizer]


def train(
        model,
        optimizer,
        device,
        criterion,
        trainloader,
        trainloader_full,
        scaler=None,
        amp_enabled=False,
        compute_full_grad=True,
):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    amp_enabled = (amp_enabled and device.type == "cuda")
    optimizers = _get_optimizers(optimizer)

    for images, labels in tqdm(trainloader, total=len(trainloader), desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
            for opt in optimizers:
                scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            for opt in optimizers:
                opt.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        num_batches += 1

    norm = get_full_grad_list(model, trainloader_full, device) if compute_full_grad else None

    train_accuracy = 100.0 * correct / total
    train_result = [train_loss / max(1, num_batches), train_accuracy]

    return norm, train_result
