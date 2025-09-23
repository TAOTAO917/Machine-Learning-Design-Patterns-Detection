import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ========== Utility Functions ==========

def script_dir() -> str:
    """Directory of the script (output files are placed here)."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def safe_read_lines(path: str):
    """Read all lines from a file, normalize to '\n' endings, no deduplication or rewriting."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Normalize line endings
    norm = []
    for ln in lines:
        if ln.endswith("\n"):
            norm.append(ln)
        else:
            norm.append(ln + "\n")
    return norm

def parse_label_from_line(line: str):
    """
    Parse label from the end of a line (last segment separated by tab).
    Return (True, int_label) if success, otherwise (False, None).
    """
    raw = line[:-1] if line.endswith("\n") else line
    parts = raw.rsplit("\t", 1)
    if len(parts) != 2:
        return (False, None)
    label_str = parts[1].strip()
    try:
        return (True, int(label_str))
    except:
        return (False, None)

def unique_output_path(base_path: str) -> str:
    """If file exists, automatically add suffix (1), (2)… to avoid overwriting."""
    if not os.path.exists(base_path):
        return base_path
    root, ext = os.path.splitext(base_path)
    i = 1
    while True:
        cand = f"{root} ({i}){ext}"
        if not os.path.exists(cand):
            return cand
        i += 1

# ========== Merge Logic ==========

def merge_txt_files(paths):
    """
    Merge multiple TXT files:
    - Read all lines
    - Group by label at the end of each line (0,1,2,...)
    - Within each label, preserve file selection order and line order
    Return (merged_text, statistics_string)
    """
    # label -> list[lines]
    buckets = {}
    unknown = []  # lines with unparseable labels (put at the end)
    stats = []    # per-file statistics

    for idx, p in enumerate(paths, 1):
        lines = safe_read_lines(p)
        cnt_ok = 0
        cnt_bad = 0
        for ln in lines:
            ok, lab = parse_label_from_line(ln)
            if ok:
                buckets.setdefault(lab, []).append(ln)
                cnt_ok += 1
            else:
                unknown.append(ln)
                cnt_bad += 1
        stats.append(f"- {os.path.basename(p)}: parsed {cnt_ok} lines successfully, failed {cnt_bad} lines")

    # Assemble output (labels in ascending order)
    merged = []
    for lab in sorted(buckets.keys()):
        merged.extend(buckets[lab])

    # Append unknown label lines at the end
    if unknown:
        merged.extend(unknown)

    merged_text = "".join(merged)
    stat_text = "Merge statistics:\n" + "\n".join(stats)
    if buckets:
        stat_text += f"\nLabels covered: {', '.join(map(str, sorted(buckets.keys())))}"
    if unknown:
        stat_text += f"\n⚠️ Lines with unparseable labels: {len(unknown)}"

    return merged_text, stat_text

# ========== Tkinter UI ==========

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Merge TXT (sorted by labels 0,1,2…)")
        self.geometry("760x520")
        self.minsize(760, 520)

        self.selected_files = []
        self.create_widgets()

    def create_widgets(self):
        # Top toolbar
        bar = ttk.Frame(self, padding=10)
        bar.pack(fill="x")

        ttk.Button(bar, text="Select TXT (multi-select)", command=self.pick_files).pack(side="left")

        # Middle: file list
        mid = ttk.Frame(self, padding=(10,0,10,10))
        mid.pack(fill="both", expand=True)

        self.listbox = tk.Listbox(mid, selectmode="extended")
        self.listbox.pack(side="left", fill="both", expand=True)

        sc = ttk.Scrollbar(mid, orient="vertical", command=self.listbox.yview)
        sc.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=sc.set)

        # Bottom controls
        bottom = ttk.Frame(self, padding=10)
        bottom.pack(fill="x")

        self.status = tk.StringVar(value="Note: lines are sorted by their numeric labels at the end (0→1→2…), while preserving file and line order within each label.")
        ttk.Label(bottom, textvariable=self.status).pack(side="left")

        ttk.Button(bottom, text="Start Merge", command=self.start_merge).pack(side="right")

    def pick_files(self):
        paths = filedialog.askopenfilenames(
            title="Select TXT files",
            filetypes=[("TXT files", "*.txt"), ("All files", "*.*")]
        )
        if not paths:
            return
        self.selected_files = list(paths)
        self.refresh_list()
        self.status.set(f"Selected {len(self.selected_files)} file(s)")

    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        for p in self.selected_files:
            self.listbox.insert(tk.END, p)

    def start_merge(self):
        if not self.selected_files:
            messagebox.showwarning("Warning", "Please select at least one TXT file first.")
            return
        try:
            merged_text, stat_text = merge_txt_files(self.selected_files)
            out_dir = script_dir()
            out_path = unique_output_path(os.path.join(out_dir, "merged_corpus.txt"))
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(merged_text)
            messagebox.showinfo("Merge Completed", f"{stat_text}\n\nOutput file:\n{out_path}")
            self.status.set(f"Done, output: {out_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Merge failed: {e}")

def main():
    App().mainloop()

if __name__ == "__main__":
    main()
