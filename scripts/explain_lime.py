# scripts/explain_lime.py
"""
LIME Explainability: highlight the most influential tokens for a single text sample.

Usage examples:
  1) Explain the 3rd sample from the validation set:
     python -m scripts.explain_lime --index 3
  2) Explain a custom text:
     python -m scripts.explain_lime --text "cv2.CascadeClassifier ..."

Outputs:
  - results/explanations/lime_<ts>_idx-*_pred-<name>_topN.html
  The console will also print the top contributing tokens with weights.
"""

import argparse, json, yaml, os, re
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "textcnn.yaml"

from utils.data_io import read_text_label_tsv, load_tokenizer
from models.textcnn import ReverseTemporal  # Allow deserialization of custom layer

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise RuntimeError(f"Empty configuration: {CONFIG_PATH}")
    return cfg

def latest_run_dir(ckpt_root: Path) -> Path:
    runs = [p for p in ckpt_root.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"Checkpoint directory is empty: {ckpt_root}")
    runs.sort(key=lambda p: p.name)
    return runs[-1]

def load_label_names(labels_map_path: Path, num_classes: int):
    if labels_map_path.exists():
        with open(labels_map_path, "r", encoding="utf-8") as f:
            mp = json.load(f)
    else:
        mp = {}
    return [mp.get(str(i), str(i)) for i in range(num_classes)]

def make_predict_fn(model, tokenizer, max_len: int):
    def _predict(texts):
        seqs = tokenizer.texts_to_sequences(list(texts))
        X = pad_sequences(seqs, maxlen=max_len, padding="pre", truncating="pre")
        probs = model.predict(X, verbose=0)
        return probs
    return _predict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="", help="checkpoint directory (contains model.h5 or .keras)")
    parser.add_argument("--index", type=int, default=0, help="when --text is not provided, use this index from dev set")
    parser.add_argument("--text", type=str, default="", help="custom text to explain (takes precedence over --index)")
    parser.add_argument("--num-features", type=int, default=10, help="number of features to display")
    parser.add_argument("--top-labels", type=int, default=1, help="number of top predicted labels to explain")
    args = parser.parse_args()

    cfg = load_config()
    data_cfg = cfg["data"]
    seq_len = int(cfg["sequence"]["max_len"])
    num_classes = int(cfg.get("num_classes", 11))

    # Choose checkpoint
    ckpt_root = PROJECT_ROOT / cfg.get("output", {}).get("ckpt_dir", "checkpoints/textcnn")
    run_dir = Path(args.ckpt) if args.ckpt else latest_run_dir(ckpt_root)

    # Model (compatible with .keras / .h5)
    m_keras = run_dir / "model.keras"
    m_h5    = run_dir / "model.h5"
    model_path = m_keras if m_keras.exists() else m_h5
    if not model_path.exists():
        raise FileNotFoundError(f"No model.keras or model.h5 found under {run_dir}")
    model = tf.keras.models.load_model(
        str(model_path),
        custom_objects={
            "ReverseTemporal": ReverseTemporal,
            "Custom>ReverseTemporal": ReverseTemporal,
        },
    )

    # Tokenizer
    tok_json = PROJECT_ROOT / "data" / "processed" / "tokenizer.json"
    tokenizer = load_tokenizer(tok_json)

    # Class names
    class_names = load_label_names(PROJECT_ROOT / data_cfg.get("labels_map_path", ""), num_classes)

    # Text source: prefer --text, otherwise take from dev set by --index
    if args.text.strip():
        text = args.text
        true_label = None
        src_desc = "custom"
    else:
        dev_path = PROJECT_ROOT / data_cfg["dev_path"]
        dev_texts, dev_labels = read_text_label_tsv(dev_path)
        if not (0 <= args.index < len(dev_texts)):
            raise IndexError(f"--index out of range (0 ~ {len(dev_texts)-1})")
        text = dev_texts[args.index]
        true_label = int(dev_labels[args.index])
        src_desc = f"dev_idx-{args.index}"

    # Predict to get top predicted label
    predict_fn = make_predict_fn(model, tokenizer, seq_len)
    pred_probs = predict_fn([text])[0]
    pred_id = int(np.argmax(pred_probs))
    pred_name = class_names[pred_id]

    # LIME: split on whitespace, keep '.' and '_' (consistent with your tokenizer)
    explainer = LimeTextExplainer(
        class_names=class_names,
        split_expression=r"\s+",   # split only on whitespace
        bow=False                  # position-sensitive, suitable for sequence models
    )
    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_fn,
        top_labels=min(args.top_labels, len(class_names)),
        num_features=args.num_features
    )

    # Output directory
    out_dir = PROJECT_ROOT / "results" / "explanations"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    html_path = out_dir / f"lime_{ts}_{src_desc}_pred-{pred_name}_top{args.num_features}.html"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(exp.as_html())

    # Console summary
    print(f"[OK] Predicted: {pred_id} ({pred_name})  | True: {true_label if true_label is not None else '-'}")
    print(f"[OK] LIME HTML saved: {html_path}")

    # Print top feature contributions for predicted label
    label_to_show = pred_id
    weights = exp.as_list(label=label_to_show)
    print("\nTop features for predicted label:")
    for tok, w in weights[:args.num_features]:
        print(f"  {tok:<30} {w:+.4f}")

if __name__ == "__main__":
    # Reduce TF logging noise
    tf.get_logger().setLevel("ERROR")
    main()
