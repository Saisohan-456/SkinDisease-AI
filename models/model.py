"""
Model definition: ResNet18 pretrained on ImageNet, fine-tuned for skin
disease classification by replacing the final fully-connected layer.

ResNet18 is chosen over heavier backbones (ResNet50/EfficientNet) because it
trains fast on CPU/modest GPUs while still giving strong results via
transfer learning -- a good default for a dataset of this size. Swap
`build_model`'s backbone if you have a bigger GPU and want more accuracy.
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

import config


def build_model(num_classes: int = config.NUM_CLASSES, freeze_backbone: bool = False):
    """
    Returns a ResNet18 with its final layer replaced for num_classes outputs.

    freeze_backbone=True freezes all convolutional layers and only trains
    the new classifier head -- faster but less accurate. Default (False)
    fine-tunes the whole network, which generally works better once you
    have a few hundred images per class.
    """
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )

    return model.to(config.DEVICE)
