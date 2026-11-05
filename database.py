"""
SQLite storage for users and prediction history.
Passwords are never stored in plain text: they are salted and hashed
with PBKDF2 (standard library only, no new dependencies).
"""
import os
import re
import json
import sqlite3
import hashlib
import hmac
import secrets
from datetime import datetime

import config

DATA_DIR = os.path.join(config.BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "app.db")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")


def _conn():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with _conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                image_path TEXT,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                top3 TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            """
        )


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000).hex()


def create_user(username: str, email: str, password: str):
    """Returns (ok, message)."""
    username, email = username.strip(), email.strip().lower()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return False, "Please enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    salt = secrets.token_bytes(16)
    try:
        with _conn() as conn:
            conn.execute(
                "INSERT INTO users (username, email, password_hash, salt, created_at) VALUES (?,?,?,?,?)",
                (username, email, _hash(password, salt), salt.hex(), datetime.now().isoformat()),
            )
    except sqlite3.IntegrityError:
        return False, "That username or email is already registered."
    return True, "Account created! You can log in now."


def authenticate(username: str, password: str):
    """Returns user dict (id, username, email) or None."""
    with _conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username.strip(),)).fetchone()
    if row is None:
        return None
    expected = _hash(password, bytes.fromhex(row["salt"]))
    if hmac.compare_digest(expected, row["password_hash"]):
        return {"id": row["id"], "username": row["username"], "email": row["email"]}
    return None


def save_prediction(user_id: int, image, predictions):
    """Saves the uploaded PIL image to disk and logs the result."""
    user_dir = os.path.join(UPLOAD_DIR, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(user_dir, f"{stamp}.jpg")
    image.convert("RGB").save(path, "JPEG", quality=90)

    top_class, top_prob = predictions[0]
    with _conn() as conn:
        conn.execute(
            "INSERT INTO history (user_id, image_path, predicted_class, confidence, top3, created_at) VALUES (?,?,?,?,?,?)",
            (user_id, path, top_class, top_prob, json.dumps(predictions), datetime.now().isoformat()),
        )


def get_history(user_id: int):
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM history WHERE user_id = ? ORDER BY id DESC", (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]
