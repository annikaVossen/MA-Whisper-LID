import os
import librosa
import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, accuracy_score
import pandas as pd

from config import TEST_DIR, MODEL_ID, HIDDEN_DIM, RUN_NAME, LR, EPOCHS
from models.mlp_model import LanguageMLP
from utils.embedding_utils import WhisperEmbedder
from utils.eval_utils import evaluate_model
from utils.plotting_utils import plot_pca, plot_tsne, plot_cm, plot_umap
from utils.save_utils import load_label_encoder, load_model
import time



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
            
            audio, _ = librosa.load(path, sr=16000, mono=True)
            X_test.append(embedder.get_embedding(audio))
            y_test.append(lang)


    return np.array(X_test), np.array(y_test)


def evaluate_and_plot_model(model, le, test_dir, device, output_dir="outputs", embedder=None, X_test=None, y_test=None, timing=False):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)


    if embedder is None:
        embedder = WhisperEmbedder(MODEL_ID, device)

    if X_test is None or y_test is None:
        X_test, y_test = build_test_dataset(test_dir, embedder)

    y_test_enc = le.transform(y_test)

    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    preds = evaluate_model(model, X_test_t, device)

    if timing:
        return preds, X_test, y_test, y_test_enc, None


    predicted_labels = [le.classes_[i] for i in preds]

    report = classification_report(y_test_enc, preds, target_names=le.classes_)
    report_path = os.path.join(output_dir, "classification_report_" + RUN_NAME + ".txt")
    with open(report_path, "w") as f:
        f.write(report)

    plot_pca(X_test, y_test, le.classes_, " Whisper Embeddings by Language for Medium Model", os.path.join(output_dir, "plots", "pca_" + RUN_NAME + ".png"))
    plot_tsne(X_test, y_test, le.classes_, "Embeddings tSNE for Medium Whisper Model", os.path.join(output_dir, "plots", "tsne_" + RUN_NAME + ".png"))
    plot_umap(X_test, y_test, le.classes_, "Embeddings UMAP for Medium Whisper Model", os.path.join(output_dir, "plots", "umap_" + RUN_NAME + ".png"))
    plot_cm(y_test, predicted_labels, le.classes_, "Confusion Matrix for Model Trained from whisper-medium Embeddings", os.path.join(output_dir, "plots", "confusion_" + RUN_NAME + ".png"))


    print(f"Saved report to {report_path}")
    print(f"Saved plots to {os.path.join(output_dir, 'plots')}")

    macro_f1 = f1_score(y_test_enc, preds, average='macro')

    result = {
        "run": RUN_NAME,
        "model": MODEL_ID,
        "hidden_dim": HIDDEN_DIM,
        "epochs": EPOCHS,
        "lr": LR,
        "accuracy": accuracy_score(y_test_enc, preds),
        "macro_f1": macro_f1,
        "weighted_f1": f1_score(y_test_enc, preds, average='weighted')
    }
    df = pd.DataFrame([result])
    # csvpath = "outputs/compare/experiments.csv"

    # if os.path.exists(csvpath):
    #     df.to_csv(csvpath, mode='a', header=False, index=False)
    # else:
    #     df.to_csv(csvpath, index=False)

    return preds, X_test, y_test, y_test_enc, df


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    le = load_label_encoder("outputs/checkpoints/label_encoder_" + RUN_NAME + ".pkl")
    label_classes = le.classes_

    embedder = WhisperEmbedder(MODEL_ID, device)

    X_test, _ = build_test_dataset(TEST_DIR, embedder)
    model = LanguageMLP(
        input_dim=X_test.shape[1],
        hidden_dim=HIDDEN_DIM,
        num_classes=len(label_classes)
    ).to(device)

    model = load_model(model, "outputs/checkpoints/mlp_model_" + RUN_NAME + ".pt", device)
    model.eval()

    evaluate_and_plot_model(model, le, TEST_DIR, device, embedder=embedder, X_test=X_test, y_test=_)
