"""
Dataset loading utilities. Wraps torchvision's ImageFolder (since the data
is already organized as train/<class_name>/*.jpg) and provides a helper to
split the train set into train/val subsets.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

import config
from utils.transforms import get_train_transforms, get_eval_transforms


def get_train_val_loaders():
    """
    Loads SkinDisease/train, splits it into train/val, and returns DataLoaders
    for both. Validation uses eval transforms (no augmentation) even though
    it's drawn from the same folder as training.
    """
    full_train_dataset = ImageFolder(config.TRAIN_DIR, transform=get_train_transforms())

    val_size = int(len(full_train_dataset) * config.VAL_SPLIT)
    train_size = len(full_train_dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size], generator=generator)

    # val_subset currently uses train transforms (inherited from full_train_dataset);
    # swap in eval transforms for a clean validation signal.
    val_dataset_eval = ImageFolder(config.TRAIN_DIR, transform=get_eval_transforms())
    val_subset.dataset = val_dataset_eval

    train_loader = DataLoader(
        train_subset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )

    class_names = full_train_dataset.classes  # alphabetical order, matches ImageFolder's internal label indices
    return train_loader, val_loader, class_names


def get_test_loader():
    test_dataset = ImageFolder(config.TEST_DIR, transform=get_eval_transforms())
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    return test_loader, test_dataset.classes
