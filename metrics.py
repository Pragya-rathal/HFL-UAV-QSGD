from typing import Dict

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def evaluate(model, dataloader, criterion) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    with torch.no_grad():
        for xb, yb in dataloader:
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item()
            total_acc += accuracy(logits, yb)
            total_batches += 1

    if total_batches == 0:
        return {"loss": 0.0, "accuracy": 0.0}

    return {
        "loss": total_loss / total_batches,
        "accuracy": total_acc / total_batches,
    }
