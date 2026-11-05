"""
Metrics helpers: accuracy tracking during training, and a fuller
classification report for test.py.
"""

import torch
from sklearn.metrics import classification_report, confusion_matrix


class AverageMeter:
    """Tracks running average of a value (loss or accuracy) across batches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0


def accuracy(outputs, labels):
    """Returns accuracy for a single batch (0-1 float)."""
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def full_classification_report(all_labels, all_preds, class_names):
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=3, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return report, cm
