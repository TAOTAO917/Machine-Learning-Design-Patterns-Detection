# scripts/evaluate.py  —— Fixed tokenizer path selection + added robustness checks
import argparse, json, yaml, os
from datetime import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf

# Custom layers (if your model includes them)
from models.textcnn import ReverseTemporal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "textcnn.yaml"

from utils.data_io import (
    read_text_label_tsv, load_tokenizer, vectorize, get_vocab_size_for_model
)
from utils.metrics import (
    per_class_accuracy, compute_confusion_matrices, plot_confusion_matrix,
    f1_summary, per_class_prf, plot_metric_bars
)
from sklearn.metrics import classification_report


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise RuntimeError(f"Empty configuration: {CONFIG_PATH}")
    return cfg


def get_latest_run(ckpt_root: Path) -> Path:
    if not ckpt_root.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_root}")
    runs = [p for p in ckpt_root.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"Checkpoint directory is empty: {ckpt_root}")
    runs.sort(key=lambda p: p.name)  # YYYY-mm-dd_HH-MM-SS
    return runs[-1]


def load_label_names(labels_map_path: Path, num_classes: int):
    if labels_map_path.exists():
        with open(labels_map_path, "r", encoding="utf-8") as f:
            mp = json.load(f)
    else:
        mp = {}
    return [mp.get(str(i), str(i)) for i in range(num_classes)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="", help="checkpoint directory (contains model.h5 or .keras)")
    args = parser.parse_args()

    cfg = load_config()
    num_classes = int(cfg.get("num_classes", 11))
    data_cfg = cfg["data"]
    seq_cfg = cfg["sequence"]
    out_cfg = cfg.get("output", {})

    ckpt_root = PROJECT_ROOT / out_cfg.get("ckpt_dir", "checkpoints/textcnn")
    run_dir = Path(args.ckpt) if args.ckpt else get_latest_run(ckpt_root)

    # Model file (.keras preferred)
    model_path_keras = run_dir / "model.keras"
    model_path_h5 = run_dir / "model.h5"
    if model_path_keras.exists():
        model_path = model_path_keras
    elif model_path_h5.exists():
        model_path = model_path_h5
    else:
        raise FileNotFoundError(f"No model.h5 or model.keras found under {run_dir}")

    # === Tokenizer path selection: prefer HF, fallback to Keras ===
    tok_path_hf    = PROJECT_ROOT / "data" / "processed" / "hf_tokenizer"
    tok_path_keras = PROJECT_ROOT / "data" / "processed" / "tokenizer.json"
    if tok_path_hf.exists():
        tok_path = tok_path_hf
    elif tok_path_keras.exists():
        tok_path = tok_path_keras
    else:
        raise FileNotFoundError("Neither data/processed/hf_tokenizer/ nor data/processed/tokenizer.json exists")

    tokenizer = load_tokenizer(tok_path)
    print(f"[INFO] Using tokenizer: {tok_path}")

    # Read dev set and vectorize (internally handles HF/Keras tokenizer differences)
    dev_path = PROJECT_ROOT / data_cfg["dev_path"]
    x_texts, y_true = read_text_label_tsv(dev_path)
    max_len = int(seq_cfg.get("max_len", 120))
    x_dev = vectorize(tokenizer, x_texts, max_len)

    # Load model (compile=False prevents deserialization errors for custom losses)
    model = tf.keras.models.load_model(
        str(model_path),
        custom_objects={
            "ReverseTemporal": ReverseTemporal,
            "Custom>ReverseTemporal": ReverseTemporal
        },
        compile=False,
    )

    # —— Debug: compare model embedding vocab size vs tokenizer vocab size
    try:
        embed_input_dim = None
        for lyr in model.layers:
            if hasattr(lyr, "input_dim"):  # Embedding layer
                embed_input_dim = getattr(lyr, "input_dim")
                break
        tok_vocab = get_vocab_size_for_model(
            tokenizer, int(cfg.get("tokenizer", {}).get("max_vocab_size", 20000))
        )
        if embed_input_dim is not None:
            print(f"[DEBUG] model Embedding.input_dim={embed_input_dim}, tokenizer_vocab={tok_vocab}")
            if abs(embed_input_dim - tok_vocab) / max(1, embed_input_dim) > 0.3:
                print("[WARN] Model vocab size and tokenizer vocab differ significantly, "
                      "training and evaluation tokenizers may not match!")
    except Exception as e:
        print(f"[DEBUG] Skipped vocab alignment check: {e}")

    # Prediction
    probs = model.predict(x_dev, verbose=0)
    if np.isnan(probs).any():
        raise RuntimeError("NaN detected in prediction probabilities. Please check your input/model.")

    y_pred = np.argmax(probs, axis=1)

    # —— Debug: predicted class distribution
    uniq, cnt = np.unique(y_pred, return_counts=True)
    dist = {int(u): int(c) for u, c in zip(uniq, cnt)}
    print(f"[DEBUG] Predicted class distribution: {dist} / total={len(y_pred)}")
    if cnt.max() / len(y_pred) > 0.9:
        print("[WARN] Over 90% of samples predicted as one class. "
              "This may indicate tokenizer mismatch or vectorization/max_len issues.")

    # Class names
    class_names = load_label_names(PROJECT_ROOT / data_cfg.get("labels_map_path", ""), num_classes)

    # Text report
    report_txt = classification_report(
        y_true, y_pred, target_names=class_names, digits=2, zero_division=0
    )
    print("\n" + report_txt)

    # Metrics and confusion matrices
    per_cls_acc = per_class_accuracy(y_true, y_pred, num_classes)
    acc = float((np.array(y_true) == y_pred).mean())
    f1s = f1_summary(y_true, y_pred)
    cm_counts, cm_norm = compute_confusion_matrices(y_true, y_pred, num_classes)

    # Output paths
    figures_dir = PROJECT_ROOT / "results" / "figures"
    logs_dir = PROJECT_ROOT / "results" / "logs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Confusion matrices
    fig_counts_path = figures_dir / f"confusion_matrix_counts_{ts}.png"
    fig_norm_path   = figures_dir / f"confusion_matrix_normalized_{ts}.png"
    plot_confusion_matrix(cm_counts, class_names, "Confusion Matrix (Counts)", str(fig_counts_path), cmap="Blues")
    plot_confusion_matrix(cm_norm,   class_names, "Confusion Matrix (Row-normalized)", str(fig_norm_path), cmap="Purples")

    # Per-class P/R/F bar charts
    prf = per_class_prf(y_true, y_pred, num_classes)
    plot_metric_bars(prf["precision"], class_names, "Precision", str(figures_dir / f"per_class_precision_{ts}.png"), color="#4C78A8")
    plot_metric_bars(prf["recall"],    class_names, "Recall",    str(figures_dir / f"per_class_recall_{ts}.png"),    color="#54A24B")
    plot_metric_bars(prf["f1"],        class_names, "F1-score",  str(figures_dir / f"per_class_f1_{ts}.png"),        color="#F58518")

    # Save reports
    with open(logs_dir / f"classification_report_{ts}.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)
    with open(logs_dir / f"eval_{ts}.txt", "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {run_dir}\nDev size: {len(y_true)}\n")
        f.write(f"Overall accuracy: {acc:.4f}\n")
        f.write(f"F1 micro: {f1s['f1_micro']:.4f}, macro: {f1s['f1_macro']:.4f}, weighted: {f1s['f1_weighted']:.4f}\n")
        f.write("\nPer-class accuracy:\n")
        for i, a in enumerate(per_cls_acc):
            sup = int(np.sum(np.array(y_true) == i))
            f.write(f"  {i:2d} ({class_names[i]}): {'N/A' if np.isnan(a) else f'{a:.4f}'} (support={sup})\n")
        f.write("\nFigures:\n")
        f.write(f"  Counts CM: {fig_counts_path}\n")
        f.write(f"  Row-norm CM: {fig_norm_path}\n")
        f.write(f"  Precision bars: {figures_dir / f'per_class_precision_{ts}.png'}\n")
        f.write(f"  Recall bars: {figures_dir / f'per_class_recall_{ts}.png'}\n")
        f.write(f"  F1 bars: {figures_dir / f'per_class_f1_{ts}.png'}\n")

    print(f"[OK] Figures saved to {figures_dir}; text reports saved to {logs_dir}")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
