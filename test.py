"""
Evaluates the best saved model on SkinDisease/test and prints a full
per-class classification report + confusion matrix.

Run:
    python test.py
"""

import torch

import config
from models.model import build_model
from utils.dataset import get_test_loader
from utils.metrics import full_classification_report


def main():
    print(f"Device: {config.DEVICE}")
    print(f"Loading test data from: {config.TEST_DIR}")

    test_loader, test_class_names = get_test_loader()

    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    saved_class_names = checkpoint["class_names"]

    if saved_class_names != test_class_names:
        print("WARNING: class order in test/ does not match the order the model was trained with.")
        print(f"  Trained order: {saved_class_names}")
        print(f"  Test order:    {test_class_names}")
        print("Proceeding using the trained model's class order for label names.")

    model = build_model(num_classes=len(saved_class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    report, cm = full_classification_report(all_labels, all_preds, saved_class_names)

    print("\nClassification Report:\n")
    print(report)
    print("\nConfusion Matrix:\n")
    print(cm)


if __name__ == "__main__":
    main()
