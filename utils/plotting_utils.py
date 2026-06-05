from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE



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
    cm = confusion_matrix(y_test_enc, preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_norm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_tsne(X, y, classes, title, save_path):
    # base_colors = plt.cm.tab20.colors  # 20 colors
    # extra_colors = plt.cm.tab10.colors[:5]  # 5 more
    # all_colors = base_colors + extra_colors
    # cmap_25 = ListedColormap(all_colors)
    tsne = TSNE(
    n_components=2,
    perplexity=10,
    learning_rate="auto",
    init="pca",
    random_state=42
    )

    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(12, 8))
    for lang in classes:
        idx = (y == lang)
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=lang)

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title("Whisper tSNE Embeddings by Language")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

