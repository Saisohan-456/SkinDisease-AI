"""
Trains the skin disease classifier on SkinDisease/train, validates on a
held-out split, and saves the best-performing weights to saved_models/.

Run:
    python train.py
"""

import os
import json
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from models.model import build_model
from utils.dataset import get_train_val_loaders
from utils.metrics import AverageMeter, accuracy


def run_epoch(model, loader, criterion, optimizer=None):
    """If optimizer is provided, runs a training pass; otherwise evaluation only."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    torch.set_grad_enabled(is_train)
    for images, labels in loader:
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

        if is_train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(accuracy(outputs, labels), images.size(0))

    return loss_meter.avg, acc_meter.avg


def main():
    os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)

    print(f"Device: {config.DEVICE}")
    print(f"Loading data from: {config.TRAIN_DIR}")

    train_loader, val_loader, class_names = get_train_val_loaders()
    print(f"Classes found ({len(class_names)}): {class_names}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Save the exact class order ImageFolder used, so predict.py never has
    # to guess or risk a mismatch with config.CLASS_NAMES ordering.
    with open(config.CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)

    model = build_model(num_classes=len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_acc = 0.0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        start = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

        scheduler.step(val_acc)
        elapsed = time.time() - start

        print(
            f"Epoch {epoch:02d}/{config.NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "val_acc": val_acc,
                "epoch": epoch,
            }, config.BEST_MODEL_PATH)
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val_acc={best_val_acc:.4f}")
    print(f"Best model saved to: {config.BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
