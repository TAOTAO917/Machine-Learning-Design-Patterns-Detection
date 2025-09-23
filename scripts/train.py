# scripts/train.py
import os, json, yaml
from datetime import datetime
from pathlib import Path
import tensorflow as tf
import numpy as np
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "textcnn.yaml"

from utils.seed import set_seed
from utils.logger import get_logger
from utils.data_io import read_text_label_tsv, load_tokenizer, vectorize, to_onehot, get_vocab_size_for_model
from models.textcnn import build_textcnn


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise RuntimeError(f"Empty configuration: {CONFIG_PATH}")
    return cfg

def ensure_dirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def _parse_class_weight(trn_cfg, y_train_ids, num_classes):
    """Supports:
       - 'balanced': automatically compute using the classic formula
       - dict: explicitly pass {class_id: weight}
       - class_weight_gamma: exponential scaling (default 1.0)
       - class_weight_clip: [lo, hi] clipping
       - others/none: return None
    """
    cw = trn_cfg.get("class_weight", "none")
    gamma = float(trn_cfg.get("class_weight_gamma", 1.0))
    clip = trn_cfg.get("class_weight_clip", None)

    if isinstance(cw, dict):
        # Allow YAML keys as strings
        w = {int(k): float(v) for k, v in cw.items()}
    elif isinstance(cw, str) and cw.lower() == "balanced":
        counts = Counter(y_train_ids)
        total = sum(counts.values())
        # Classic balanced formula: w_c = total / (num_classes * count_c)
        w = {c: total / (num_classes * counts.get(c, 1)) for c in range(num_classes)}
    else:
        return None

    # Exponential scaling (can strengthen imbalance handling)
    if gamma != 1.0:
        w = {c: (val ** gamma) for c, val in w.items()}

    # Clip to avoid extreme values
    if clip:
        lo, hi = float(clip[0]), float(clip[1])
        w = {c: max(lo, min(hi, val)) for c, val in w.items()}

    return w

# Define after imports, before main()
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=None, gamma=1.5, name="focal_loss"):
        super().__init__(name=name)
        self.alpha = (tf.constant(alpha, dtype=tf.float32)
                      if alpha is not None else None)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Numerical stability
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        # Per-class cross entropy
        ce = -y_true * tf.math.log(y_pred)  # [B, C]
        if self.alpha is not None:
            ce = ce * self.alpha             # Class weighting (alpha vector)
        # Focal modulation based on true class probability
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)  # [B]
        focal = tf.pow(1.0 - p_t, self.gamma)          # [B]
        loss = tf.reduce_sum(ce, axis=-1) * focal      # [B]
        return tf.reduce_mean(loss)


def main():
    log = get_logger("train")
    cfg = load_config()
    set_seed(int(cfg.get("seed", 2025)))

    data_cfg = cfg["data"]
    tok_cfg  = cfg["tokenizer"]
    seq_cfg  = cfg["sequence"]
    mdl_cfg  = cfg["model"]
    trn_cfg  = cfg["train"]
    out_cfg  = cfg.get("output", {})

    # Paths
    train_path = PROJECT_ROOT / data_cfg["train_path"]
    dev_path   = PROJECT_ROOT / data_cfg["dev_path"]
    tok_path_hf    = PROJECT_ROOT / "data" / "processed" / "hf_tokenizer"
    tok_path_keras = PROJECT_ROOT / "data" / "processed" / "tokenizer.json"
    tok_path = tok_path_hf if tok_path_hf.exists() else tok_path_keras
    tokenizer = load_tokenizer(tok_path)
    log.info(f"Using tokenizer: {tok_path}")

    # Load data and tokenizer
    log.info(f"Reading train/dev data: {train_path.name}, {dev_path.name}")
    x_train_texts, y_train_ids = read_text_label_tsv(train_path)
    x_dev_texts,   y_dev_ids   = read_text_label_tsv(dev_path)

    # Print class distribution for imbalance observation
    counts = Counter(y_train_ids)
    log.info(f"Training set class distribution: {dict(counts)}")
    
    # Actual vocab_size: min(configured limit, actual vocab+1)
    # For HF tokenizer: vocab_size + added_tokens
    vocab_size = get_vocab_size_for_model(
        tokenizer,
        max_vocab_size_keras=int(cfg.get("tokenizer", {}).get("max_vocab_size", 20000))
    )

    max_len = int(seq_cfg.get("max_len", 120))
    num_classes = int(cfg.get("num_classes", 11))

    x_train = vectorize(tokenizer, x_train_texts, max_len)
    x_dev   = vectorize(tokenizer, x_dev_texts,   max_len)
    y_train = to_onehot(y_train_ids, num_classes)
    y_dev   = to_onehot(y_dev_ids,   num_classes)

    log.info(f"vocab_size={vocab_size}, max_len={max_len}, num_classes={num_classes}")
    log.info(f"x_train: {x_train.shape}, x_dev: {x_dev.shape}")

    # === Build TextCNN ===
    model = build_textcnn(
        seq_len=max_len,
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_size=int(mdl_cfg.get("embedding_size", 50)),
        filter_sizes=tuple(mdl_cfg.get("filter_sizes", [2,3,4,5])),
        num_filters=int(mdl_cfg.get("num_filters", 64)),
        dropout=float(mdl_cfg.get("dropout", 0.5)),
        l2_reg=float(mdl_cfg.get("l2_reg", 0.001)),
    )

    # === Loss: optional label smoothing ===
    ls = float(trn_cfg.get("label_smoothing", 0.0) or 0.0)
    loss_obj = tf.keras.losses.CategoricalCrossentropy(label_smoothing=ls) \
               if ls > 0 else trn_cfg.get("loss", "categorical_crossentropy")

    model.compile(
        optimizer=trn_cfg.get("optimizer", "adam"),
        loss=loss_obj,
        metrics=trn_cfg.get("metrics", ["accuracy"])
    )
    model.summary(print_fn=lambda s: None)

    # === Output directories ===
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_root = PROJECT_ROOT / out_cfg.get("ckpt_dir", "checkpoints/textcnn")
    run_dir = ckpt_root / ts
    ensure_dirs(run_dir)

    results_root = PROJECT_ROOT / out_cfg.get("results_dir", "results")
    logs_dir = results_root / "logs"
    ensure_dirs(logs_dir)

    ckpt_path = run_dir / "model.keras"  # Native Keras format

    # === Callbacks ===
    save_best_only = bool(trn_cfg.get("save_best_only", True))

    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),          # If save_best_only=False, will be overwritten each epoch
            monitor="val_accuracy",
            mode="max",
            save_best_only=save_best_only,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(logs_dir / f"train_{ts}.csv")),
    ]

    # Optional: learning rate scheduler
    if str(trn_cfg.get("lr_scheduler", "none")).lower() == "reduce_on_plateau":
        rop = trn_cfg.get("reduce_on_plateau", {}) or {}
        cbs.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", mode="max",
            factor=float(rop.get("factor", 0.5)),
            patience=int(rop.get("patience", 2)),
            min_lr=float(rop.get("min_lr", 1e-5)),
            verbose=1
        ))

    # Optional: early stopping (patience=0 or not set => disabled)
    patience = int(trn_cfg.get("early_stopping_patience", 0) or 0)
    if patience > 0:
        cbs.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max",
            patience=patience, restore_best_weights=True
        ))

    # === Build class weights (used as alpha for focal loss) ===
    class_weight = _parse_class_weight(trn_cfg, y_train_ids, num_classes)
    alpha_vec = None
    if class_weight is not None:
        alpha_vec = [class_weight.get(i, 1.0) for i in range(num_classes)]

    # === Loss ===
    loss_name = str(trn_cfg.get("loss", "categorical_crossentropy")).lower()
    if loss_name == "focal":
        gamma = float(trn_cfg.get("focal_gamma", 1.5))
        loss_obj = FocalLoss(alpha=alpha_vec, gamma=gamma)
        # Avoid double-weighting: alpha in loss + class_weight in fit
        fit_class_weight = None
    else:
        ls = float(trn_cfg.get("label_smoothing", 0.0) or 0.0)
        loss_obj = (tf.keras.losses.CategoricalCrossentropy(label_smoothing=ls)
                    if ls > 0 else "categorical_crossentropy")
        fit_class_weight = class_weight  # Only used for non-focal loss

    model.compile(optimizer=trn_cfg.get("optimizer", "adam"),
                loss=loss_obj,
                metrics=trn_cfg.get("metrics", ["accuracy"]))


    # === Training ===
    history = model.fit(
        x=x_train, y=y_train,
        batch_size=int(trn_cfg.get("batch_size", 64)),
        epochs=int(trn_cfg.get("epochs", 15)),
        validation_data=(x_dev, y_dev),
        callbacks=cbs,
        class_weight=fit_class_weight,   # Key: may be None
        verbose=1
    )
    model.save(str(run_dir / "model_last.keras"))

    # Save metrics and config snapshot
    best_val_acc = float(np.max(history.history.get("val_accuracy", [0.0])))
    metrics = {"best_val_accuracy": best_val_acc}
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"\n[OK] Training complete. Best val_accuracy={best_val_acc:.4f}")
    print(f"[OK] Model saved to: {ckpt_path}")

if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
