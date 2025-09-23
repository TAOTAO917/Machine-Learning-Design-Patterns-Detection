# utils/callbacks.py
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

class MacroF1Callback(tf.keras.callbacks.Callback):
    def __init__(self, x_dev, y_dev, name="val_macro_f1"):
        super().__init__()
        self.x_dev = x_dev
        self.y_dev = y_dev
        self.name = name

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        probs = self.model.predict(self.x_dev, verbose=0)
        y_pred = probs.argmax(axis=1)
        y_true = self.y_dev.argmax(axis=1)
        m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        logs[self.name] = m
        print(f"\n[Metric] {self.name}: {m:.4f}")
