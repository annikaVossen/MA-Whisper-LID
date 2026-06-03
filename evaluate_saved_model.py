import os
import librosa
import numpy as np
import torch
from sklearn.metrics import classification_report

from config import TEST_DIR, MODEL_ID, HIDDEN_DIM
from models.mlp_model import LanguageMLP
from utils.embedding_utils import WhisperEmbedder
from utils.eval_utils import evaluate_model
from utils.plotting_utils import plot_pca, plot_tsne, plot_cm
from utils.save_utils import load_label_encoder, load_model


def build_test_dataset(test_dir, embedder):
    X_test = []
    y_test = []

    for lang in sorted(os.listdir(test_dir)):
        lang_folder = os.path.join(test_dir, lang)
        if not os.path.isdir(lang_folder):
            continue

        for file_name in sorted(os.listdir(lang_folder)):
            if not file_name.lower().endswith((".wav", ".mp3")):
                continue

            path = os.path.join(lang_folder, file_name)
            try:
                audio, _ = librosa.load(path, sr=16000, mono=True)
                X_test.append(embedder.get_embedding(audio))
                y_test.append(lang)
            except Exception as e:
                print(f"Skipping {path}: {e}")

    return np.array(X_test), np.array(y_test)


def evaluate_and_plot_model(model, le, test_dir, device, output_dir="outputs", embedder=None, X_test=None, y_test=None):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    if embedder is None:
        embedder = WhisperEmbedder(MODEL_ID, device)

    if X_test is None or y_test is None:
        X_test, y_test = build_test_dataset(test_dir, embedder)

    y_test_enc = le.transform(y_test)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    preds = evaluate_model(model, X_test_t, device)

    report = classification_report(y_test_enc, preds, target_names=le.classes_)
    report_path = os.path.join(output_dir, "classification_report_test.txt")
    with open(report_path, "w") as f:
        f.write(report)

    plot_pca(X_test, y_test, le.classes_, "Sample Whisper Embeddings by Language", os.path.join(output_dir, "plots", "pca_test.png"))
    plot_tsne(X_test, y_test, le.classes_, "Sample Embeddings tSNE", os.path.join(output_dir, "plots", "tsne_test.png"))
    plot_cm(y_test_enc, preds, le.classes_, "Sample Confusion Matrix", os.path.join(output_dir, "plots", "confusion_test.png"))

    print(f"Saved report to {report_path}")
    print(f"Saved plots to {os.path.join(output_dir, 'plots')}")

    return preds, X_test, y_test, y_test_enc


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    le = load_label_encoder("outputs/checkpoints/label_encoder.pkl")
    label_classes = le.classes_

    embedder = WhisperEmbedder(MODEL_ID, device)

    X_test, _ = build_test_dataset(TEST_DIR, embedder)
    model = LanguageMLP(
        input_dim=X_test.shape[1],
        hidden_dim=HIDDEN_DIM,
        num_classes=len(label_classes)
    ).to(device)

    model = load_model(model, "outputs/checkpoints/mlp_model_test.pt", device)
    model.eval()

    evaluate_and_plot_model(model, le, TEST_DIR, device, embedder=embedder, X_test=X_test, y_test=_)
