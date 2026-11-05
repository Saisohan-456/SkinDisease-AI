"""
Prints the values for the report tables:
  - Experimental detail (dataset counts, device, best epoch, accuracies, duration)
  - Overall Model Performance (macro / weighted precision, recall, F1)

Run from the project folder, after training:
    python get_report_values.py

Optional: save your training console output as train_log.txt to also get
the total training duration.
"""
import os
import re

import torch
from sklearn.metrics import classification_report

import config
from models.model import build_model
from utils.dataset import get_train_val_loaders, get_test_loader

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def count_images(root):
    total = 0
    for _, _, files in os.walk(root):
        total += sum(1 for f in files if f.lower().endswith(IMG_EXT))
    return total


def training_duration():
    log = "train_log.txt"
    if not os.path.exists(log):
        return None
    pat = re.compile(r"Epoch\s+\d+/\d+.*\|\s*([\d.]+)s\s*$")
    secs, epochs = 0.0, 0
    with open(log, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line.strip())
            if m:
                secs += float(m.group(1))
                epochs += 1
    if not epochs:
        return None
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s ({epochs} epochs)"


def main():
    ckpt = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    class_names = ckpt["class_names"]

    train_loader, val_loader, _ = get_train_val_loaders()
    test_loader, _ = get_test_loader()
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)

    model = build_model(num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(config.DEVICE).eval()

    preds, labels = [], []
    with torch.no_grad():
        for images, y in test_loader:
            out = model(images.to(config.DEVICE))
            preds.extend(torch.argmax(out, 1).cpu().tolist())
            labels.extend(y.tolist())

    rep = classification_report(labels, preds, labels=list(range(len(class_names))),
                                target_names=class_names, output_dict=True, zero_division=0)
    ma, wa = rep["macro avg"], rep["weighted avg"]

    device = str(config.DEVICE)
    if device.startswith("cuda"):
        device += f" ({torch.cuda.get_device_name(0)})"

    dur = training_duration() or "not available (save console output as train_log.txt, or note it yourself)"

    print("\n===== Experimental detail =====")
    print("Dataset version and source :  (write this yourself: dataset name + link/author)")
    print(f"Total images               : {count_images(config.TRAIN_DIR) + count_images(config.TEST_DIR)}")
    print(f"Train / validation / test  : {n_train} / {n_val} / {n_test}")
    print(f"Actual device              : {device}")
    print(f"Best epoch                 : {ckpt.get('epoch', '?')}")
    print(f"Best validation accuracy   : {ckpt.get('val_acc', 0) * 100:.2f}%")
    print(f"Test accuracy              : {rep['accuracy'] * 100:.2f}%")
    print(f"Training duration          : {dur}")

    print("\n===== Overall Model Performance =====")
    print(f"Best validation accuracy   : {ckpt.get('val_acc', 0) * 100:.2f}%")
    print(f"Test accuracy              : {rep['accuracy'] * 100:.2f}%")
    print(f"Macro precision            : {ma['precision']:.4f}")
    print(f"Macro recall               : {ma['recall']:.4f}")
    print(f"Macro F1-score             : {ma['f1-score']:.4f}")
    print(f"Weighted precision         : {wa['precision']:.4f}")
    print(f"Weighted recall            : {wa['recall']:.4f}")
    print(f"Weighted F1-score          : {wa['f1-score']:.4f}")


if __name__ == "__main__":
    main()