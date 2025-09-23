# -*- coding: utf-8 -*-
"""
GUI: Select a folder -> scan all .py files -> parse function calls -> export as one JSON file (array).
- Supports alias restoration: import torch as th / from torch import nn as tnn
- Output fields:
  {
    "File": "file path",
    "Name": [deduplicated list of function end names, e.g. ["fit", "relu"]],
    "Attributes": [deduplicated list of full dotted names, e.g. ["torch.nn.functional.relu"]],
    "Torch": true/false,
    "Sklearn": true/false,
    "Tensorflow": true/false
  }
"""

import os
import json
import ast
from typing import Dict, List, Optional, Set, Tuple

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QPushButton, QLabel,
    QLineEdit, QVBoxLayout, QHBoxLayout, QCheckBox, QTextEdit, QProgressBar,
    QMessageBox
)

# ---------- AST Utilities ----------

def read_text(path: str) -> str:
    # Robust reading to avoid encoding errors
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def get_alias_map(tree: ast.AST) -> Dict[str, str]:
    """
    Collect import alias mappings:
      - import torch as th         -> {'th': 'torch'}
      - from torch import nn as n  -> {'n': 'torch.nn'}
    """
    alias_map: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                top = (name.name or "").split('.')[0]
                if name.asname:
                    alias_map[name.asname] = top
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for name in node.names:
                if name.asname:
                    full = module + (("." + name.name) if name.name else "")
                    alias_map[name.asname] = full
    return alias_map

def dotted_from_func(func: ast.AST) -> Optional[str]:
    """
    Try to extract a dotted string from Call.func.
    Handles Name / Attribute / Call(func) / Subscript / Lambda.
    """
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        left = dotted_from_func(func.value)
        if left:
            return f"{left}.{func.attr}"
        return func.attr  # fallback
    if isinstance(func, ast.Call):
        return dotted_from_func(func.func)
    if isinstance(func, ast.Subscript):
        return dotted_from_func(func.value)
    if isinstance(func, ast.Lambda):
        return "<lambda>"
    return None

def normalize_with_alias(dotted: str, alias_map: Dict[str, str]) -> str:
    """
    Replace alias prefixes with real modules:
    - e.g. 'tnn.ReLU' with alias tnn -> torch.nn, becomes 'torch.nn.ReLU'.
    """
    if not dotted:
        return dotted
    parts = dotted.split(".")
    if parts and parts[0] in alias_map:
        mapped = alias_map[parts[0]]
        mapped_parts = mapped.split(".") if mapped else []
        parts = mapped_parts + parts[1:]
    return ".".join(parts)

def collect_calls(tree: ast.AST) -> Tuple[Set[str], Set[str]]:
    """
    Traverse AST, collect function calls:
    - Returns (names, attributes)
      names: set of deduplicated function end names (e.g. 'fit', 'relu')
      attributes: set of deduplicated full dotted names (e.g. 'torch.nn.functional.relu')
    """
    names: Set[str] = set()
    attributes: Set[str] = set()

    alias_map = get_alias_map(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            dotted = dotted_from_func(node.func)
            if not dotted:
                continue
            dotted = normalize_with_alias(dotted, alias_map)
            attributes.add(dotted)
            # End name (last segment)
            names.add(dotted.split(".")[-1])

    return names, attributes

def detect_frameworks(import_alias_map: Dict[str, str], attributes: Set[str], source_text: str) -> Tuple[bool, bool, bool]:
    """
    More robust framework detection:
    - Prefer checking imports/aliases and call chains
    - Finally fallback to source text substring (may cause false positives)
    """
    top_imports = set(import_alias_map.values())
    top_aliases = set(import_alias_map.keys())

    def has_prefix(attrs: Set[str], prefix: str, aliases: Set[str]) -> bool:
        for a in attrs:
            if a == prefix or a.startswith(prefix + "."):
                return True
        for a in attrs:
            head = a.split(".", 1)[0]
            if head in aliases and import_alias_map.get(head, "").startswith(prefix):
                return True
        return False

    torch_ok = ("torch" in top_imports) or has_prefix(attributes, "torch", top_aliases) or ("torch" in source_text)
    sk_ok = ("sklearn" in top_imports) or has_prefix(attributes, "sklearn", top_aliases) or ("sklearn" in source_text)
    tf_ok = ("tensorflow" in top_imports) or has_prefix(attributes, "tensorflow", top_aliases) or ("tensorflow" in source_text)

    return bool(torch_ok), bool(sk_ok), bool(tf_ok)

def build_record(py_path: str, source_text: str) -> Optional[dict]:
    """
    Build one record for a single .py file.
    Skip and return None if syntax error occurs.
    """
    try:
        tree = ast.parse(source=source_text)
    except SyntaxError:
        return None

    alias_map = get_alias_map(tree)
    names, attrs = collect_calls(tree)
    torch_ok, sk_ok, tf_ok = detect_frameworks(alias_map, attrs, source_text)

    return {
        "File": os.path.abspath(py_path),
        "Name": sorted(names),            # dedup + sorted for easier diff
        "Attributes": sorted(attrs),
        "Torch": torch_ok,
        "Sklearn": sk_ok,
        "Tensorflow": tf_ok,
    }

def scan_folder(folder: str, recursive: bool) -> List[str]:
    """
    Return the list of .py files to process.
    By default, process all .py files.  
    If you only want 1.py..100.py, you can select "only numeric" in the GUI (see Worker).
    """
    files: List[str] = []
    if recursive:
        for root, _, filenames in os.walk(folder):
            for fn in filenames:
                if fn.lower().endswith(".py"):
                    files.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(folder):
            if fn.lower().endswith(".py"):
                files.append(os.path.join(folder, fn))
    files.sort()
    return files

# ---------- Background Worker ----------

class Worker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(int, int, str)
    message = pyqtSignal(str)

    def __init__(self, folder: str, recursive: bool, only_numeric: bool):
        super().__init__()
        self.folder = folder
        self.recursive = recursive
        self.only_numeric = only_numeric
        self.out_path = os.path.join(folder, "CodeFeatures.json")  # default output in same folder

    def run(self):
        all_files = scan_folder(self.folder, self.recursive)
        if self.only_numeric:
            all_files = [p for p in all_files if os.path.basename(p)[:-3].isdigit()]

        total = len(all_files)
        ok_count = 0
        results = []

        if total == 0:
            self.message.emit("No .py files found to process.")
            self.progress.emit(100, "")
            self.finished.emit(0, 0, self.out_path)
            return

        for idx, path in enumerate(all_files, start=1):
            try:
                src = read_text(path)
                rec = build_record(path, src)
                if rec is not None:
                    results.append(rec)
                    ok_count += 1
                else:
                    self.message.emit(f"[Skipped] Syntax error: {path}")
            except Exception as e:
                self.message.emit(f"[Failed] {path} => {e}")

            percent = int(idx * 100 / total)
            self.progress.emit(percent, os.path.basename(path))

        try:
            with open(self.out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.message.emit(f"Written: {self.out_path} ({ok_count}/{total} records)")
        except Exception as e:
            self.message.emit(f"[Write failed] {e}")

        self.finished.emit(total, ok_count, self.out_path)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Code Feature Extraction - JSON Generator (PyQt5)")
        self.setMinimumWidth(720)

        # Folder selection
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Please select a folder containing .py files")
        self.folder_edit.setReadOnly(True)
        btn_choose = QPushButton("Select Folder")
        btn_choose.clicked.connect(self.choose_folder)

        # Options
        self.cb_recursive = QCheckBox("Include subfolders")
        self.cb_numeric = QCheckBox("Only process numeric filenames (1.py ~ N.py)")
        self.cb_recursive.setChecked(True)

        # Run
        btn_run = QPushButton("Start Processing")
        btn_run.clicked.connect(self.start_process)

        # Progress and log
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.now_label = QLabel("Ready")
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        # Layout
        line1 = QHBoxLayout()
        line1.addWidget(self.folder_edit, 1)
        line1.addWidget(btn_choose)

        opts = QHBoxLayout()
        opts.addWidget(self.cb_recursive)
        opts.addWidget(self.cb_numeric)
        opts.addStretch(1)
        opts.addWidget(btn_run)

        v = QVBoxLayout()
        v.addLayout(line1)
        v.addLayout(opts)
        v.addWidget(self.progress)
        v.addWidget(self.now_label)
        v.addWidget(self.log)

        container = QWidget()
        container.setLayout(v)
        self.setCentralWidget(container)

        self.worker: Optional[Worker] = None

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_edit.setText(folder)

    def start_process(self):
        folder = self.folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Warning", "Please select a valid folder first.")
            return

        self.progress.setValue(0)
        self.log.clear()
        self.now_label.setText("Processing…")

        self.worker = Worker(
            folder=folder,
            recursive=self.cb_recursive.isChecked(),
            only_numeric=self.cb_numeric.isChecked(),
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.message.connect(self.append_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_progress(self, percent: int, filename: str):
        self.progress.setValue(percent)
        self.now_label.setText(f"Processing: {filename} … {percent}%")

    def append_log(self, text: str):
        self.log.append(text)

    def on_finished(self, total: int, ok: int, out_path: str):
        self.now_label.setText("Done")
        QMessageBox.information(self, "Done", f"Processing finished: {ok}/{total} records.\nOutput: {out_path}")

def main():
    import sys
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
