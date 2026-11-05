"""
Central configuration for the SkinDiseaseAI project.
Edit DATA_ROOT to point at wherever your SkinDisease/train and
SkinDisease/test folders actually live on your machine.
"""
import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "SkinDisease")

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

SAVED_MODELS_DIR = "saved_models"
BEST_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "best_model.pth")
CLASS_NAMES_PATH = os.path.join(SAVED_MODELS_DIR, "class_names.json")

# ---------------------------------------------------------------------------
# Classes (must match your folder names exactly, alphabetically sorted
# is how ImageFolder will order them — train.py will save the real order
# it used to class_names.json, so predict.py never has to guess)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "Acne",
    "Actinic_Keratosis",
    "Benign_tumors",
    "Bullous",
    "Candidiasis",
    "DrugEruption",
    "Eczema",
    "Infestations_Bites",
    "Lichen",
    "Lupus",
    "Moles",
    "Psoriasis",
    "Rosacea",
    "Seborrh_Keratoses",
    "SkinCancer",
    "Sun_Sunlight_Damage",
    "Tinea",
    "Unknown_Normal",
    "Vascular_Tumors",
    "Vasculitis",
    "Vitiligo",
    "Warts",
]

NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Model / training hyperparameters
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224          # standard input size for ResNet
BATCH_SIZE = 8
NUM_EPOCHS = 35
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0        # set to 0 on Windows if you hit multiprocessing issues
VAL_SPLIT = 0.15          # fraction of train/ held out for validation during training

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
