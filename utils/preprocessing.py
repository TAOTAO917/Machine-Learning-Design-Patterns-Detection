# utils/preprocessing.py
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ← Added: for splitting CamelCase
_CAMEL_RE = re.compile(r'([a-z0-9])([A-Z])')

# ← Added: text augmentation (append subword views without breaking the original text)
def augment_text(text: str,
                 split_dot_underscore: bool = True,
                 split_camel: bool = True,
                 dedup: bool = True) -> str:
    """
    Returns: original text + (optional) view with ./_ replaced by space +
             view with CamelCase split, concatenated together.

    Example:
        'cv2.CascadeClassifier detectMultiScale'
        -> 'cv2.CascadeClassifier detectMultiScale cv2 CascadeClassifier detect Multi Scale'
    """
    parts = [text]
    if split_dot_underscore:
        parts.append(text.replace('.', ' ').replace('_', ' '))
    if split_camel:
        parts.append(_CAMEL_RE.sub(r'\1 \2', text))
    merged = " ".join(parts)
    if dedup:  # remove duplicates in order of appearance to control length
        toks = merged.split()
        merged = " ".join(dict.fromkeys(toks))
    return merged

# Keep the original name pad_to_len; ensure post/post padding and truncation
def pad_to_len(seqs, max_len: int):
    return pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")
