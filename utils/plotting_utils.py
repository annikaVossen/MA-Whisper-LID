from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import umap



def plot_pca(X, y, classes, title, save_path):
    X_2d = PCA(n_components=2).fit_transform(X)

    for lang in classes:
        idx = (y == lang)
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=lang)

    plt.legend()
    plt.title("Whisper Embeddings by Language")
    plt.savefig(save_path)
    plt.close()

def plot_cm(y_test_enc, preds, classes, title, save_path):
    labels = np.unique(classes)
    print("Labels for confusion matrix:", labels)
    print("Unique predicted labels:", np.unique(preds))
    print("Unique true labels:", np.unique(y_test_enc))
    cm = confusion_matrix(y_test_enc, preds, labels=labels)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_norm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_tsne(X, y, classes, title, save_path):


    colors = np.concatenate([plt.colormaps["tab20"](np.linspace(0,1,20)), plt.colormaps["tab20b"](np.linspace(0,1,20))[0::4]])

    tsne = TSNE(
    n_components=2,
    perplexity=10,
    learning_rate="auto",
    init="pca",
    random_state=42
    )

    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(12, 8))
    lang_id = 0
    for lang in classes:
        idx = (y == lang)
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=lang, color=colors[lang_id])
        lang_id += 1

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_umap(X, y, classes, title, save_path):

    colors = np.concatenate([plt.colormaps["tab20"](np.linspace(0,1,20)), plt.colormaps["tab20b"](np.linspace(0,1,20))[0::4]])

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    X_2d = reducer.fit_transform(X)

    plt.figure(figsize=(12, 8))
    lang_id = 0
    for lang in classes:
        idx = (y == lang)
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=lang, color=colors[lang_id])
        lang_id += 1

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()