from tqdm import tqdm
from .get_full_grad_list import get_full_grad_list


def train(
        model,
        optimizer,
        device,
        criterion,
        trainloader,
        trainloader_full,
):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(
        tqdm(trainloader, total=len(trainloader), desc="Training", leave=False)
    ):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    norm = get_full_grad_list(model, trainloader_full, device)

    train_accuracy = 100. * correct / total
    train_result = [train_loss / (batch_idx + 1), train_accuracy]

    return norm, train_result
