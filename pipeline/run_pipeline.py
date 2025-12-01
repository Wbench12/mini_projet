"""QSAR Pipeline Runner

Placeholder end-to-end steps:
1. Load raw features.
2. Simple normalization (min-max per column excluding id).
3. Save processed features.
4. Run dummy predictions via services.model and save outputs.

Replace with real semi-supervised training / inference logic.
"""
from __future__ import annotations

from pathlib import Path
import csv

from app.services.model import predict_batch, get_model

RAW_PATH = Path("data/raw/features.csv")
PROCESSED_PATH = Path("data/processed/features_processed.csv")
PREDICTIONS_PATH = Path("data/predictions/predictions.csv")


def load_raw():
    rows = []
    with RAW_PATH.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def normalize(rows):
    # Collect numeric columns except molecule_id
    numeric_cols = [c for c in rows[0].keys() if c != "molecule_id"]
    mins = {c: min(float(r[c]) for r in rows) for c in numeric_cols}
    maxs = {c: max(float(r[c]) for r in rows) for c in numeric_cols}
    for r in rows:
        for c in numeric_cols:
            val = float(r[c])
            rng = maxs[c] - mins[c]
            r[c] = (val - mins[c]) / rng if rng != 0 else 0.0
    return rows, numeric_cols


def save_processed(rows, cols):
    with PROCESSED_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["molecule_id", *cols])
        for r in rows:
            writer.writerow([r["molecule_id"], *(r[c] for c in cols)])


def run_predictions(rows, cols):
    feature_matrix = [[float(r[c]) for r in cols] for r in rows]
    preds = predict_batch(feature_matrix)
    with PREDICTIONS_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["molecule_id", "prediction", "model_version"])
        model = get_model()
        for r, p in zip(rows, preds):
            writer.writerow([r["molecule_id"], p, model.version if model else "unknown"])


def main():
    if not RAW_PATH.exists():
        raise SystemExit(f"Raw features file not found: {RAW_PATH}")
    rows = load_raw()
    rows_norm, cols = normalize(rows)
    save_processed(rows_norm, cols)
    run_predictions(rows_norm, cols)
    print("Pipeline complete:")
    print(f" - Processed: {PROCESSED_PATH}")
    print(f" - Predictions: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
