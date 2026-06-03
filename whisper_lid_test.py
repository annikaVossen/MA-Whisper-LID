import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
from transformers import WhisperProcessor, WhisperModel
import librosa
import os
import random
# from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn as nn
from mlp_model import LanguageMLP, train_model
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

print("imported")

model_id = "openai/whisper-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperModel.from_pretrained(model_id).to(device)
model.eval()

def get_embedding(audio_array):
    inputs = processor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt"
    )  # convert audio to log-mel specgram for model
    
    with torch.no_grad():
        encoder_outputs = model.encoder(inputs["input_features"].to(device)) #encoder forward pass
        # encoder applies several transformer layers to produce embeddings

    embedding = encoder_outputs.last_hidden_state.mean(dim=1) # mean pooling over time, very simple choice
    # could try other strategies of temporal aggregation, like maxpooling, attention pooling,statistical pooling, or some combo
    # currently only looking at final encoder hidden states, want to also considering looking at some intermediate representations

    return embedding.squeeze(0).cpu().numpy()


def load_audio(path):
    audio, sr = librosa.load(path, sr=16000, mono=True)
    return audio


X = []
y = []
X_test = []
y_test = []
MAX_PER_LANG = 5000
data_d = "fleurs_data"
data_train = os.path.join(data_d, "train")
data_test = os.path.join(data_d, "test")

for lang in os.listdir(data_train):
    lang_folder = os.path.join(data_train, lang)

    if os.path.isdir(lang_folder):

        files = os.listdir(os.path.join(data_train, lang))
        files = [f for f in files if f.endswith((".wav", ".mp3"))]

        selected = random.sample(files, min(MAX_PER_LANG, len(files)))

        for file in selected:
            path = os.path.join(lang_folder, file)

            try:
                audio = load_audio(path)
                emb = get_embedding(audio)

                X.append(emb)
                y.append(lang)

            except Exception as e:
                print(f"Skipping {path}: {e}")

for lang in os.listdir(data_test):
    lang_folder = os.path.join(data_test, lang)

    if os.path.isdir(lang_folder):

        files = os.listdir(os.path.join(data_test, lang))
        files = [f for f in files if f.endswith((".wav", ".mp3"))]

        for file in files:
            path = os.path.join(lang_folder, file)

            try:
                audio = load_audio(path)
                emb = get_embedding(audio)

                X_test.append(emb)
                y_test.append(lang)

            except Exception as e:
                print(f"Skipping {path}: {e}")


# train-test splits are provided in FLEURS dataset
X_train = np.array(X)
y_train = np.array(y)

X_test = np.array(X_test)
y_test = np.array(y_test)

le = LabelEncoder()

y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train_enc, dtype=torch.long)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test_enc, dtype=torch.long)


model = LanguageMLP(
    input_dim=X_train.shape[1],
    hidden_dim=128,
    num_classes=len(set(y_train))
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_model(
    model,
    optimizer,
    X_train_t,
    y_train_t,
    device,
    epochs=30
)

## predict
model.eval()

with torch.no_grad():
    logits = model(X_test_t.to(device))
    preds = torch.argmax(logits, dim=1).cpu().numpy()


# plot embeddings via PCA for visualization
X_arr = X_test
y_arr = y_test

X_2d = PCA(n_components=2).fit_transform(X_arr)

for lang in set(y_arr):
    idx = y_arr == lang
    plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=lang)

plt.legend()
plt.title("Whisper Embeddings by Language")
plt.savefig("whisper_embeddings_pca.png")
plt.close()


report = classification_report(
    y_test_enc,
    preds,
    target_names=le.classes_
)

with open("classification_report.txt", "w") as f:
    f.write(report)
print(report)

torch.save(model.state_dict(), "mlp_model_baseline.pt")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)


# confusion matrix

cm = confusion_matrix(y_test_enc, preds)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(10, 8))
plt.imshow(cm_norm, interpolation="nearest")
plt.title("Confusion Matrix (Whisper Embeddings + MLP)")
plt.colorbar()

tick_marks = np.arange(len(le.classes_))
plt.xticks(tick_marks, le.classes_, rotation=90)
plt.yticks(tick_marks, le.classes_)

plt.xlabel("Predicted label")
plt.ylabel("True label")

plt.tight_layout()
plt.savefig("confusion_matrix_baseline.png", dpi=300)
plt.close()