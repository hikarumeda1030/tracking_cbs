import torch
from tqdm import tqdm


def test(model, device, testloader, criterion, amp_enabled=False):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    amp_enabled = (amp_enabled and device.type == "cuda")

    with torch.no_grad():
        for images, labels in tqdm(testloader, total=len(testloader), desc="Testing", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            num_batches += 1

    test_accuracy = 100.0 * correct / total
    test_result = [test_loss / max(1, num_batches), test_accuracy]
    return test_result
