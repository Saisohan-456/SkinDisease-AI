# 🩺 SkinDiseaseAI

A Streamlit web app that classifies skin conditions from a photo using a ResNet18 model (transfer learning), then shows care tips and commonly used medications for the predicted condition. Users create an account, log in, and can review their activity history on a profile page.

> **Disclaimer:** This is a student / personal machine-learning project, **not a medical device**. It cannot diagnose any condition and is not a substitute for a doctor. Medication information is general and is not a prescription. For anything concerning, see a dermatologist or doctor.

## Features

- **Login and registration** with a clean white and blue interface
- **Skin condition prediction** across 22 classes, with the top 3 results and confidence scores
- **Care tips and common medications** for every class, labelled as OTC, prescription, procedure or self-care
- **Urgency indicator** on each result (self-care, confirm with a doctor, or see a doctor promptly)
- **Profile page** with your username, member-since date, stats, a full activity history (logins, logouts, analyses) and a logout button
- **Safe password storage** using salted PBKDF2-SHA256 (200,000 iterations) in a local SQLite database

## Supported classes

Acne, Actinic keratosis, Benign tumors, Bullous, Candidiasis, Drug eruption, Eczema, Infestations and bites, Lichen, Lupus, Moles, Psoriasis, Rosacea, Seborrheic keratoses, Skin cancer, Sun damage, Tinea, Normal skin, Vascular tumors, Vasculitis, Vitiligo, Warts.

## Project structure

```
SkinDiseaseAI/
├── app.py               # Streamlit app (header, analyze page, profile page)
├── auth.py              # Accounts, activity history, login/sign-up/profile screens
├── ui.py                # White and blue theme (CSS) and HTML helpers
├── recommendation.py    # Care tips and medications for every class
├── predict.py           # Loads the model and predicts a single image
├── train.py             # Trains the classifier
├── test.py              # Evaluates on the test set (report + confusion matrix)
├── config.py            # Paths and hyperparameters
├── requirements.txt
├── .streamlit/
│   └── config.toml      # Streamlit theme
├── models/
│   └── model.py         # build_model()
├── utils/
│   ├── transforms.py    # get_eval_transforms()
│   ├── dataset.py       # get_train_val_loaders(), get_test_loader()
│   └── metrics.py       # AverageMeter, accuracy, full_classification_report()
├── SkinDisease/         # dataset (not in the repo)
│   ├── train/<class folders>
│   └── test/<class folders>
└── saved_models/        # created by train.py (not in the repo)
    ├── best_model.pth
    └── class_names.json
```

## Getting started

### 1. Requirements

- Python **3.11 or 3.12** (the pinned `torch==2.3.1` does not support Python 3.13)
- A skin disease image dataset arranged as `SkinDisease/train/<class>/*.jpg` and `SkinDisease/test/<class>/*.jpg`

### 2. Install

```powershell
# Windows (PowerShell)
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train.py
```

This saves the best weights to `saved_models/best_model.pth`. Adjust epochs, batch size and learning rate in `config.py`.

### 4. Evaluate (optional)

```bash
python test.py
```

### 5. Run the app

```bash
python -m streamlit run app.py
```

Open http://localhost:8501, create an account, and upload a skin photo.

You can also test the model from the command line:

```bash
python predict.py path/to/image.jpg
```

## How it works

1. **Authentication:** `auth.py` stores users in `users.db` (created automatically on first run). Passwords are never stored in plain text.
2. **Prediction:** the uploaded image is resized and normalised, then passed to a fine-tuned ResNet18. Softmax probabilities give the top 3 conditions.
3. **Recommendations:** `recommendation.py` maps each class to care tips, medications and an urgency level. Serious conditions (skin cancer, lupus, vasculitis and others) show the treatments a doctor may discuss, clearly marked as not for self-treatment.
4. **History:** every login, logout and analysis is saved per user and shown on the profile page. Analysis history can be cleared from the profile.

## Notes

- `users.db`, `saved_models/` and the dataset are excluded from git via `.gitignore`.
- Sessions live in the browser tab: refreshing the page logs you out.
- Model accuracy depends on the dataset. Treat every result as a rough hint, not a diagnosis.

## Tech stack

Python, PyTorch, torchvision, Streamlit, SQLite, Pillow, scikit-learn.

## License

Add a license of your choice (for example MIT) before publishing.
