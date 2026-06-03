import argparse
import numpy as np
import torch
import os
import random

from sklearn.preprocessing import LabelEncoder

from config import *

from models.mlp_model import LanguageMLP, train_model
from evaluate_saved_model import evaluate_and_plot_model

from utils.embedding_utils import WhisperEmbedder
from utils.save_utils import load_label_encoder, save_label_encoder, save_model, load_model

import librosa

device = "cuda" if torch.cuda.is_available() else "cpu"

def build_embeddings(data_dir, embedder, max_per_lang=None):
    X = []
    y = []

    for lang in sorted(os.listdir(data_dir)):
        lang_folder = os.path.join(data_dir, lang)
        if not os.path.isdir(lang_folder):
            continue

        files = [f for f in os.listdir(lang_folder) if f.endswith((".wav", ".mp3"))]
        if max_per_lang is not None:
            files = random.sample(files, min(max_per_lang, len(files)))

        for file in files:
            path = os.path.join(lang_folder, file)
            try:
                audio, _ = librosa.load(path, sr=16000, mono=True)
                emb = embedder.get_embedding(audio)
                X.append(emb)
                y.append(lang)
            except Exception as e:
                print(f"Skipping {path}: {e}")

    return np.array(X), np.array(y)


def train_and_evaluate(data_train, data_test, device):
    embedder = WhisperEmbedder(MODEL_ID, device)

    X_train, y_train = build_embeddings(data_train, embedder, max_per_lang=MAX_PER_LANG)
    X_test, y_test = build_embeddings(data_test, embedder)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_enc, dtype=torch.long)

    model = LanguageMLP(
        input_dim=X_train.shape[1],
        hidden_dim=HIDDEN_DIM,
        num_classes=len(le.classes_)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_model(
        model,
        optimizer,
        X_train_t,
        y_train_t,
        device,
        epochs=EPOCHS
    )

    os.makedirs("outputs/checkpoints", exist_ok=True)
    save_model(model, "outputs/checkpoints/mlp_model_test.pt")
    save_label_encoder(le, "outputs/checkpoints/label_encoder.pkl")

    evaluate_and_plot_model(model, le, data_test, device, output_dir="outputs", embedder=embedder, X_test=X_test, y_test=y_test)


def evaluate_only(data_test, device, checkpoint_path="outputs/checkpoints/mlp_model_test.pt", encoder_path="outputs/checkpoints/label_encoder.pkl"):
    le = load_label_encoder(encoder_path)
    embedder = WhisperEmbedder(MODEL_ID, device)
    X_test, y_test = build_embeddings(data_test, embedder)

    model = LanguageMLP(
        input_dim=X_test.shape[1],
        hidden_dim=HIDDEN_DIM,
        num_classes=len(le.classes_)
    ).to(device)

    model = load_model(model, checkpoint_path, device)
    model.eval()

    evaluate_and_plot_model(model, le, data_test, device, output_dir="outputs", embedder=embedder, X_test=X_test, y_test=y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and/or evaluate the Whisper language MLP model.")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip training and evaluate an existing saved model.")
    args = parser.parse_args()

    if args.evaluate_only:
        evaluate_only(TEST_DIR, device)
    else:
        train_and_evaluate(TRAIN_DIR, TEST_DIR, device)