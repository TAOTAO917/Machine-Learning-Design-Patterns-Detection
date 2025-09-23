# scripts/build_tokenizer.py — Support CodeBERT(HF) and whitespace tokenization
import os, json, yaml
import numpy as np
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "textcnn.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level of {path} must be a mapping (key: value)")
    return data


def read_texts(path: Path):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" not in line:
                raise ValueError(f"[{path.name}:{i}] Missing tab separator: {line[:80]}")
            text, _label = line.split("\t", 1)
            if text:
                texts.append(text)
    if not texts:
        raise RuntimeError(f"No text found in {path}")
    return texts


def _save_stats_yaml(stats: dict, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(stats, f, allow_unicode=True, sort_keys=False)
    print(f"[OK] Saved statistics to: {out_path}")


def main():
    cfg = load_config()
    data_cfg = cfg.get("data", {})
    tok_cfg  = cfg.get("tokenizer", {}) or {}
    seq_cfg  = cfg.get("sequence", {}) or {}

    train_path = PROJECT_ROOT / data_cfg.get("train_path", "data/train_data.txt")
    dev_path   = PROJECT_ROOT / data_cfg.get("dev_path",   "data/val_data.txt")

    # Read all texts (train + dev for length distribution statistics)
    texts = read_texts(train_path)
    if dev_path.exists():
        texts += read_texts(dev_path)

    mode = str(tok_cfg.get("mode", "hf")).lower()  # 'hf' or 'whitespace'

    if mode == "whitespace":
        # === Tokenize by whitespace: no lowercasing, no filtering ===
        from tensorflow.keras.preprocessing.text import Tokenizer

        max_vocab_size = int(tok_cfg.get("max_vocab_size", 20000))
        oov_token = tok_cfg.get("oov_token", "[OOV]")

        # Normalize multiple whitespaces to a single space, keep tokens unchanged
        texts = [" ".join(t.split()) for t in texts]

        print("[INFO] Using whitespace tokenizer (split by space, no filtering, no lowercasing)")
        tokenizer = Tokenizer(
            num_words=max_vocab_size,
            oov_token=oov_token,
            lower=False,
            filters="",
            split=" "
        )
        tokenizer.fit_on_texts(texts)

        # Save to data/processed/tokenizer.json
        out_dir = PROJECT_ROOT / "data" / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        tok_json_path = out_dir / "tokenizer.json"
        with open(tok_json_path, "w", encoding="utf-8") as f:
            f.write(tokenizer.to_json())
        print(f"[OK] Tokenizer saved to: {tok_json_path}")

        # Compute sequence lengths and OOV (after whitespace tokenization)
        sequences = tokenizer.texts_to_sequences(texts)
        lengths = [len(s) for s in sequences]
        if not lengths:
            raise RuntimeError("Empty tokenization results, please check your input data.")

        # Count OOV tokens
        oov_index = tokenizer.word_index.get(oov_token, None)
        total_tokens = int(np.sum(lengths))
        oov_tokens = int(sum(sum(1 for t in seq if t == oov_index) for seq in sequences)) if oov_index else 0
        oov_ratio = float(oov_tokens / total_tokens) if total_tokens > 0 else 0.0

        raw_vocab_size = len(tokenizer.word_index)  # includes OOV
        effective_vocab_size = min(max_vocab_size, raw_vocab_size + 1)  # +1 for OOV

        p95 = int(np.percentile(lengths, 95))
        p99 = int(np.percentile(lengths, 99))
        max_obs = int(np.max(lengths))
        max_len_cfg = int(seq_cfg.get("max_len", 256))

        stats = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_docs": int(len(texts)),
            "tokenizer": {
                "type": "whitespace",
                "configured_max_vocab_size": max_vocab_size,
                "effective_vocab_size": effective_vocab_size,
                "oov_token": oov_token,
            },
            "coverage": {
                "total_tokens": total_tokens,
                "oov_tokens": oov_tokens,
                "oov_ratio": round(oov_ratio, 6),
            },
            "sequence": {
                "configured_max_len": max_len_cfg,
                "suggest_max_len_p95": p95,
                "suggest_max_len_p99": p99,
                "max_observed": max_obs
            }
        }

        stats_path = out_dir / "stats.yaml"
        _save_stats_yaml(stats, stats_path)

        print("\n[SUMMARY]")
        print(f"- Number of documents: {stats['num_docs']}")
        print(f"- Vocabulary (effective/configured limit): {effective_vocab_size} / {max_vocab_size}")
        print(f"- OOV coverage: {oov_tokens}/{total_tokens} = {oov_ratio:.4%}")
        print(f"- Length recommendation (max_len): p95={p95}, p99={p99}, max={max_obs}")

    else:
        # === HuggingFace tokenizer (CodeBERT / others) ===
        from transformers import AutoTokenizer

        hf_model_name = tok_cfg.get("hf_model_name", "microsoft/codebert-base")
        use_fast = bool(tok_cfg.get("use_fast", True))
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, use_fast=use_fast)

        # Add extra tokens (optional)
        extra_tokens = tok_cfg.get("extra_tokens", []) or []
        if extra_tokens:
            added = tokenizer.add_tokens(list(set(extra_tokens)))
            print(f"[INFO] Added {added} extra tokens to tokenizer.")

        # Save to data/processed/hf_tokenizer
        out_dir = PROJECT_ROOT / "data" / "processed" / "hf_tokenizer"
        out_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(out_dir))
        print(f"[OK] HF tokenizer saved to: {out_dir}")

        # Compute sequence lengths by HF tokenizer
        lens = []
        for t in texts:
            enc = tokenizer(
                t,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_attention_mask=False
            )
            lens.append(len(enc["input_ids"]))
        lens = np.array(lens)
        p95 = int(np.percentile(lens, 95))
        p99 = int(np.percentile(lens, 99))
        max_obs = int(lens.max())
        vocab_size = int(tokenizer.vocab_size + len(tokenizer.get_added_vocab()))
        max_len_cfg = int(seq_cfg.get("max_len", 256))

        stats = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_docs": int(len(texts)),
            "tokenizer": {
                "type": "hf",
                "hf_model_name": hf_model_name,
                "use_fast": use_fast,
                "base_vocab_size": int(tokenizer.vocab_size),
                "added_tokens": int(len(tokenizer.get_added_vocab())),
                "effective_vocab_size": vocab_size
            },
            "sequence": {
                "configured_max_len": max_len_cfg,
                "suggest_max_len_p95": p95,
                "suggest_max_len_p99": p99,
                "max_observed": max_obs
            }
        }

        stats_path = out_dir / "stats.yaml"
        _save_stats_yaml(stats, stats_path)

        print("\n[SUMMARY]")
        print(f"- Number of documents: {stats['num_docs']}")
        print(f"- Vocabulary (base/added/total): {stats['tokenizer']['base_vocab_size']} / "
              f"{stats['tokenizer']['added_tokens']} / {stats['tokenizer']['effective_vocab_size']}")
        print(f"- Length recommendation (max_len): p95={p95}, p99={p99}, max={max_obs}")
        print("\n[Hint] To add project-specific tokens, edit tokenizer.extra_tokens in configs/textcnn.yaml.")


if __name__ == "__main__":
    main()
