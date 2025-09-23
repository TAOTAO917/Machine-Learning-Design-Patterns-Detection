# scripts/clean_dataset.py
import argparse, re
from pathlib import Path

# Default “low-discriminative” common words to remove (case-insensitive)
DEFAULT_REMOVE = {
    "print","append","add","remove","read","write","open","close","range",
    "split","sorted","strip","join","len","max","min","sum","mean","type",
    "int","float","str","bool","time","sleep","copy","update","set","get",
    "imshow","waitkey","rectangle","resize","release","read","write"
}
# Must-preserve tokens (strongly indicative keywords)
PRESERVE = {
    # Cascade
    "CascadeClassifier","detectMultiScale","cv2.CascadeClassifier","detectmultiscale",
    # Rebalancing
    "SMOTE","fit_resample","RandomOverSampler","RandomUnderSampler","class_weight","focal_loss",
    # Reframing
    "VarianceThreshold","permutationimportance","maxnorm","LabelEncoder","get_support",
    # Other commonly seen and discriminative tokens
    "cv2","numpy","tensorflow","keras","img_to_array","expand_dims"
}

TOKEN_RE = re.compile(r"[A-Za-z0-9_.]+")  # Match tokens consisting of letters/numbers/._

SHORT_STOP = {"a","an","the","to","in","of","on","by","at","is","are"}  # Short stopwords for denoising

KEEP_SHORT = {"cv","cv2","np","tf","pd","plt","sk"}  # Short but meaningful abbreviations

def clean_text(text, remove_set):
    # Extract tokens (keep . and _), filter step by step
    tokens = TOKEN_RE.findall(text)
    out = []
    for tok in tokens:
        # Preserve list has highest priority
        if tok in PRESERVE or tok.lower() in {t.lower() for t in PRESERVE}:
            out.append(tok); continue
        lo = tok.lower()
        # Remove short stopwords & generic function/operation words
        if lo in SHORT_STOP: 
            continue
        if lo in remove_set:
            continue
        # Remove very short uninformative tokens (but keep meaningful abbreviations)
        if len(tok) <= 2 and tok.lower() not in KEEP_SHORT:
            continue
        out.append(tok)
    return " ".join(out)

def process_file(fin: Path, fout: Path, remove_extra):
    remove_set = {t.lower() for t in (DEFAULT_REMOVE | set(remove_extra))}
    kept, dropped_empty, changed = 0, 0, 0
    fout.parent.mkdir(parents=True, exist_ok=True)
    with fin.open("r", encoding="utf-8") as f_in, fout.open("w", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in, 1):
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            text, label = line.rsplit("\t", 1)
            new_text = clean_text(text, remove_set)
            if not new_text.strip():
                dropped_empty += 1
                continue
            if new_text != text:
                changed += 1
            f_out.write(f"{new_text}\t{label}\n")
            kept += 1
    return kept, dropped_empty, changed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_in", default="data/train_data_balanced.txt")
    ap.add_argument("--dev_in",   default="data/val_data.txt")
    ap.add_argument("--train_out", default="data/cleaned/train_clean.txt")
    ap.add_argument("--dev_out",   default="data/cleaned/val_clean.txt")
    ap.add_argument("--remove", default="", help="extra tokens to remove, comma-separated, e.g.: print,append,add")
    args = ap.parse_args()

    extra = [t.strip() for t in args.remove.split(",") if t.strip()]
    k1,d1,c1 = process_file(Path(args.train_in), Path(args.train_out), extra)
    k2,d2,c2 = process_file(Path(args.dev_in),   Path(args.dev_out),   extra)

    print(f"[CLEAN] train kept={k1}, dropped_empty={d1}, changed={c1} -> {args.train_out}")
    print(f"[CLEAN] dev   kept={k2}, dropped_empty={d2}, changed={c2} -> {args.dev_out}")
    print("[NOTE] The tokenizer needs to be re-fitted using the cleaned training data.")

if __name__ == "__main__":
    main()
