# utils/metrics.py
from typing import List, Sequence
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

def per_class_accuracy(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> List[float]:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    accs = []
    for c in range(num_classes):
        mask = (y_true == c)
        accs.append(float((y_pred[mask] == y_true[mask]).mean()) if mask.sum() else float("nan"))
    return accs

def compute_confusion_matrices(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int):
    labels = list(range(num_classes))
    cm_counts = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = cm_counts.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm_counts, row_sums, where=row_sums!=0)
        cm_norm = np.nan_to_num(cm_norm)
    return cm_counts, cm_norm

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, save_path: str, cmap: str = "Blues"):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = ".2f" if np.issubdtype(cm.dtype, np.floating) else "d"
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def f1_summary(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    return {
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

# === 新增：每类 PRF 值 + 条形图 ===
def per_class_prf(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int):
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    return {"precision": p, "recall": r, "f1": f1, "support": support}

def plot_metric_bars(values: np.ndarray, class_names: List[str], metric_name: str, save_path: str, color: str = "#4C78A8"):
    # 横向条形图（“一条一条”）
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(class_names) + 1))
    y_pos = np.arange(len(class_names))
    ax.barh(y_pos, values, align="center", color=color)
    ax.set_yticks(y_pos); ax.set_yticklabels(class_names)
    ax.set_xlabel(metric_name); ax.set_xlim(0, 1)
    ax.set_title(f"{metric_name} per class")
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
