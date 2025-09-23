from typing import List, Tuple
import numpy as np
import tensorflow as tf
from pathlib import Path

# === Added: HF tokenizer support ===
from transformers import AutoTokenizer
import json

def read_text_label_tsv(path: Path) -> Tuple[List[str], List[int]]:
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line: 
                continue
            if "\t" not in line:
                raise ValueError(f"[{path}] Line {i} is missing a tab: {line[:80]}")
            text, label_str = line.split("\t", 1)
            texts.append(text)
            labels.append(int(label_str))
    return texts, labels

def _is_hf_tokenizer_dir(p: Path) -> bool:
    # HF tokenizer directories usually contain tokenizer.json or vocab.json/merges.txt
    if p.is_dir():
        if (p / "tokenizer.json").exists() or (p / "vocab.json").exists():
            return True
    return False

def load_tokenizer(path: Path):
    """
    If `path` is a directory, try loading as a HuggingFace tokenizer;
    If it is a .json file, fall back to the legacy Keras Tokenizer (compatibility mode).
    """
    if _is_hf_tokenizer_dir(path):
        tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
        tok._is_hf = True  # Mark it for later checks
        return tok

    # Compatibility: legacy Keras Tokenizer
    if path.suffix == ".json":
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
        with open(path, "r", encoding="utf-8") as f:
            json_str = f.read()
        tok = tokenizer_from_json(json_str)
        tok._is_hf = False
        return tok

    raise FileNotFoundError(f"Unrecognized tokenizer path: {path}")

def vectorize(tokenizer, texts: List[str], max_len: int):
    """
    If HF tokenizer: return input_ids of shape (N, max_len) as int32.
    If Keras tokenizer: use the legacy logic (not recommended, kept for compatibility).
    """
    if getattr(tokenizer, "_is_hf", False):
        enc = tokenizer(
            texts,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_attention_mask=False,
            return_tensors=None
        )
        ids = np.array(enc["input_ids"], dtype="int32")
        return ids
    else:
        # Legacy Keras compatibility (not recommended)
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seqs = tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=max_len, padding="pre", truncating="pre")

def to_onehot(labels: List[int], num_classes: int):
    return tf.keras.utils.to_categorical(np.array(labels), num_classes=num_classes)

def get_vocab_size_for_model(tokenizer, max_vocab_size_keras: int = None) -> int:
    """
    Determine Embedding input_dim during training:
      - HF: vocab_size + added_vocab
      - Keras: min(max_vocab_size, len(word_index)+1)
    """
    if getattr(tokenizer, "_is_hf", False):
        base = int(tokenizer.vocab_size)
        added = int(len(tokenizer.get_added_vocab()))
        return base + added
    else:
        assert max_vocab_size_keras is not None, "Keras Tokenizer requires max_vocab_size to be provided"
        return min(int(max_vocab_size_keras), len(tokenizer.word_index) + 1)
