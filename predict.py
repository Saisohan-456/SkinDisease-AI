"""
Loads the trained model and predicts the class of a single image.
Can be run standalone from the command line, or imported by app.py.

Run:
    python predict.py path/to/image.jpg
"""

import sys
import json

import torch
from PIL import Image

import config
from models.model import build_model
from utils.transforms import get_eval_transforms

_model = None
_class_names = None
_transform = get_eval_transforms()


def load_model():
    """Loads the model once and caches it (so Streamlit doesn't reload on every interaction)."""
    global _model, _class_names

    if _model is not None:
        return _model, _class_names

    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    _class_names = checkpoint["class_names"]

    _model = build_model(num_classes=len(_class_names))
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()

    return _model, _class_names


def predict_image(image: Image.Image, top_k: int = 3):
    """
    Predicts the class of a PIL Image. Returns a list of (class_name, probability)
    tuples, sorted descending, for the top_k predictions.
    """
    model, class_names = load_model()

    image = image.convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)

    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(class_names)))

    results = [
        (class_names[idx], float(prob))
        for idx, prob in zip(top_indices.tolist(), top_probs.tolist())
    ]
    return results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    img = Image.open(image_path)
    predictions = predict_image(img, top_k=3)

    print(f"\nTop predictions for {image_path}:")
    for class_name, prob in predictions:
        print(f"  {class_name}: {prob * 100:.2f}%")
