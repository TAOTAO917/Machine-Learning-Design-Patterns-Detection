import json
import re
import os
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ---------- Core Logic ----------

def extract_number(filepath: str) -> int:
    """Extract the number at the end of a filename: /path/10.py -> 10; if no match, send to the very end."""
    base = os.path.basename(filepath or "")
    m = re.search(r"(\d+)\.py$", base)
    return int(m.group(1)) if m else 10**9

def sort_json_entries(data):
    """Sort entries naturally by the number in 'File' (1,2,3,10…)."""
    return sorted(data, key=lambda x: extract_number(x.get("File", "")))

def tokens_one_item(item):
    """
    Extract Name + Attributes from one object and concatenate into a single token string (space-separated).
    - Remove newlines and surrounding whitespace
    - Keep only strings
    - Deduplicate while preserving first appearance order
    """
    names = item.get("Name", []) or []
    attrs = item.get("Attributes", []) or []
    seq = []
    seen = set()
    for s in list(names) + list(attrs):
        if isinstance(s, str):
            t = s.replace("\r", " ").replace("\n", " ").strip()
            if t and t not in seen:
                seen.add(t)
                seq.append(t)
    return " ".join(seq)

def convert_one_json(in_path: str, label: int) -> dict:
    """
    Convert one JSON:
    - Read and sort by File number
    - Output to same_dir/output/{base}_sorted.json
    - Generate same_dir/output/{base}_label{label}.txt
      Rule: each File → one line: `Name+Attributes (space-separated)\t<label>`
    """
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be an array (list).")

    data_sorted = sort_json_entries(data)

    base_dir = os.path.dirname(in_path)
    out_dir = os.path.join(base_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(in_path))[0]
    sorted_json_path = os.path.join(out_dir, f"{base_name}_sorted.json")
    txt_path = os.path.join(out_dir, f"{base_name}_label{label}.txt")

    # Write sorted JSON (preserve non-ASCII characters)
    with open(sorted_json_path, "w", encoding="utf-8") as f:
        json.dump(data_sorted, f, indent=2, ensure_ascii=False)

    # Write TXT: one line per File
    lines_written = 0
    with open(txt_path, "w", encoding="utf-8") as f:
        for item in data_sorted:
            line_body = tokens_one_item(item)
            if line_body:  # skip empty
                f.write(f"{line_body}\t{label}\n")
                lines_written += 1

    return {
        "sorted_json_path": sorted_json_path,
        "txt_path": txt_path,
        "count_lines": lines_written,  # number of lines = number of Files (may be fewer if some were skipped)
    }

# ---------- Tkinter UI ----------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("JSON → Corpus (one line per File)")
        self.geometry("740x500")
        self.minsize(740, 500)
        self.selected_files = []
        self.create_widgets()

    def create_widgets(self):
        toolbar = ttk.Frame(self, padding=10)
        toolbar.pack(fill="x")

        ttk.Button(toolbar, text="Select JSON (multi-select)", command=self.pick_files).pack(side="left")

        ttk.Label(toolbar, text="Label:").pack(side="right")
        self.label_var = tk.StringVar(value="0")
        self.spin = ttk.Spinbox(toolbar, from_=0, to=999, width=5,
                                textvariable=self.label_var, justify="center")
        self.spin.pack(side="right", padx=(0,10))

        mid = ttk.Frame(self, padding=(10,0,10,10))
        mid.pack(fill="both", expand=True)

        self.listbox = tk.Listbox(mid, selectmode="extended")
        self.listbox.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(mid, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=scrollbar.set)

        bottom = ttk.Frame(self, padding=10)
        bottom.pack(fill="x")

        self.status_var = tk.StringVar(value="Note: each File → one line: Name+Attributes (space-separated)\\t<label>")
        ttk.Label(bottom, textvariable=self.status_var).pack(side="left")

        ttk.Button(bottom, text="Start Conversion", command=self.start_convert).pack(side="right")

    def pick_files(self):
        paths = filedialog.askopenfilenames(
            title="Select JSON files",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not paths:
            return
        self.selected_files = list(paths)
        self.refresh_listbox()
        self.status_var.set(f"Selected {len(self.selected_files)} file(s)")

    def refresh_listbox(self):
        self.listbox.delete(0, tk.END)
        for p in self.selected_files:
            self.listbox.insert(tk.END, p)

    def start_convert(self):
        if not self.selected_files:
            messagebox.showwarning("Warning", "Please select at least one JSON file first.")
            return
        try:
            label = int(self.label_var.get())
            if not (0 <= label <= 999):
                raise ValueError
        except Exception:
            messagebox.showerror("Error", "Label must be an integer between 0 and 999.")
            return

        ok, msgs = 0, []
        for path in self.selected_files:
            try:
                res = convert_one_json(path, label)
                ok += 1
                msgs.append(
                    f"✅ {os.path.basename(path)}\n"
                    f"  - Sorted JSON: {res['sorted_json_path']}\n"
                    f"  - Corpus TXT: {res['txt_path']}\n"
                    f"  - Lines: {res['count_lines']}\n"
                )
            except Exception as e:
                msgs.append(f"❌ {os.path.basename(path)} Error: {e}\n{traceback.format_exc(limit=1)}")

        self.status_var.set(f"Completed: success {ok}/{len(self.selected_files)}")
        messagebox.showinfo("Conversion Done", "\n".join(msgs))

def main():
    App().mainloop()

if __name__ == "__main__":
    main()
