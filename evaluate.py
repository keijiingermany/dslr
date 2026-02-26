"""
evaluate.py
-----------
Self-evaluation script for the DSLR logistic regression classifier.

Two modes:

1. Cross-validation on training data (default):
   python3 evaluate.py [dataset_train.csv] [weights.json]
   Splits dataset_train.csv (80/20) and reports accuracy.

2. Compare houses.csv against a ground-truth CSV:
   python3 evaluate.py --compare <ground_truth.csv> <houses.csv>
   Uses the "Hogwarts House" column of ground_truth as labels.

The accuracy metric matches scikit-learn's accuracy_score definition:
  accuracy = number of correct predictions / total predictions
"""

import sys
import csv
import math
import os
import json
import random
from lib.utils import (
    resolve_dataset_path, read_csv_dicts, numeric_feature_names,
    safe_float, DROP_COLUMNS, HOUSE_COL, die
)
from lib.preprocess import preprocess_fit_transform, preprocess_transform
from lib.logreg import train_ovr, predict_ovr_one


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

REDUNDANT_FEATURES = {"Defense Against the Dark Arts"}


def load_train_data(path: str):
    fieldnames, rows = read_csv_dicts(path)
    if HOUSE_COL not in fieldnames:
        die(f"'{HOUSE_COL}' not found in {path}")
    features = [
        f for f in numeric_feature_names(fieldnames, DROP_COLUMNS)
        if f not in REDUNDANT_FEATURES
    ]
    X, y = [], []
    for r in rows:
        house = r.get(HOUSE_COL, "").strip()
        if not house or house.lower() == "nan":
            continue
        X.append([safe_float(r.get(f, "")) for f in features])
        y.append(house)
    return X, y, features


def confusion_matrix_str(y_true, y_pred, classes):
    """Returns a formatted confusion matrix string."""
    n = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            matrix[idx[t]][idx[p]] += 1
    col_w = max(len(c) for c in classes) + 2
    header = " " * col_w + "".join(f"{c:>{col_w}}" for c in classes) + "  ← predicted"
    lines = [header]
    for i, c in enumerate(classes):
        row = f"{c:>{col_w}}" + "".join(f"{matrix[i][j]:>{col_w}}" for j in range(n))
        lines.append(row)
    return "\n".join(lines)


def per_class_stats(y_true, y_pred, classes):
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"  {c:<30s}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}")


# ──────────────────────────────────────────────
# Mode 1: cross-validation
# ──────────────────────────────────────────────

def cross_validate(train_path: str, weights_path: str | None):
    X, y, features = load_train_data(train_path)
    m = len(X)

    # Stratified 80/20 split
    random.seed(42)
    indices = list(range(m))
    random.shuffle(indices)
    split = int(0.8 * m)
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_val   = [X[i] for i in val_idx]
    y_val   = [y[i] for i in val_idx]

    # Train
    X_train_copy = [row[:] for row in X_train]
    Xb_train, means, stds = preprocess_fit_transform(X_train_copy)
    classes = sorted(set(y_train))
    thetas = train_ovr(Xb_train, y_train, classes, lr=0.1, iters=5000)

    # Validate
    X_val_copy = [row[:] for row in X_val]
    Xb_val = preprocess_transform(X_val_copy, means[:], stds)
    y_pred = [predict_ovr_one(xb, thetas) for xb in Xb_val]

    correct = sum(1 for t, p in zip(y_val, y_pred) if t == p)
    acc = correct / len(y_val) * 100

    print("=" * 55)
    print("  DSLR — Cross-Validation Evaluation (80/20 split)")
    print("=" * 55)
    print(f"  Train samples : {len(X_train)}")
    print(f"  Val   samples : {len(X_val)}")
    print(f"  Features used : {len(features)}")
    print(f"  Accuracy      : {correct}/{len(y_val)} = {acc:.2f}%")
    status = "✓ PASS (≥ 98%)" if acc >= 98.0 else "✗ FAIL (< 98%)"
    print(f"  Status        : {status}")
    print()
    print("Per-class metrics (validation set):")
    all_classes = sorted(set(y_val) | set(y_pred))
    per_class_stats(y_val, y_pred, all_classes)
    print()
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix_str(y_val, y_pred, all_classes))
    print()

    # Also show full-training accuracy if weights exist
    if weights_path and os.path.isfile(weights_path):
        with open(weights_path) as f:
            model = json.load(f)
        saved_features = model["features"]
        saved_means = model["means"]
        saved_stds = model["stds"]
        saved_thetas = model["thetas"]

        X_full = [row[:] for row in X]
        Xb_full = preprocess_transform(X_full, saved_means[:], saved_stds)
        y_full_pred = [predict_ovr_one(xb, saved_thetas) for xb in Xb_full]
        correct_full = sum(1 for t, p in zip(y, y_full_pred) if t == p)
        print(f"Full-training accuracy (from {weights_path}): "
              f"{correct_full}/{len(y)} = {correct_full/len(y)*100:.2f}%")


# ──────────────────────────────────────────────
# Mode 2: compare houses.csv to ground-truth
# ──────────────────────────────────────────────

def compare_to_ground_truth(gt_path: str, pred_path: str):
    # Load ground truth
    if not os.path.isfile(gt_path):
        die(f"ground truth file not found: {gt_path}")
    gt_map = {}
    with open(gt_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = row.get("Index", "").strip()
            house = row.get(HOUSE_COL, "").strip()
            if idx != "" and house != "":
                gt_map[idx] = house

    # Load predictions
    if not os.path.isfile(pred_path):
        die(f"prediction file not found: {pred_path}")
    pred_map = {}
    with open(pred_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = row.get("Index", "").strip()
            house = row.get(HOUSE_COL, "").strip()
            if idx != "":
                pred_map[idx] = house

    common = sorted(set(gt_map) & set(pred_map), key=lambda x: int(x) if x.isdigit() else x)
    if not common:
        die("no matching Index rows between ground truth and predictions")

    y_true = [gt_map[i] for i in common]
    y_pred = [pred_map[i] for i in common]
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = correct / len(common) * 100

    print("=" * 55)
    print("  DSLR — Prediction vs Ground Truth Evaluation")
    print("=" * 55)
    print(f"  Ground truth  : {gt_path}")
    print(f"  Predictions   : {pred_path}")
    print(f"  Samples       : {len(common)}")
    print(f"  Accuracy      : {correct}/{len(common)} = {acc:.2f}%")
    status = "✓ PASS (≥ 98%)" if acc >= 98.0 else "✗ FAIL (< 98%)"
    print(f"  Status        : {status}")
    print()

    all_classes = sorted(set(y_true) | set(y_pred))
    print("Per-class metrics:")
    per_class_stats(y_true, y_pred, all_classes)
    print()
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix_str(y_true, y_pred, all_classes))
    print()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if args and args[0] == "--compare":
        # python3 evaluate.py --compare <gt.csv> <houses.csv>
        if len(args) < 3:
            die("Usage: python3 evaluate.py --compare <ground_truth.csv> <houses.csv>")
        compare_to_ground_truth(args[1], args[2])
    else:
        # python3 evaluate.py [dataset_train.csv] [weights.json]
        train_path = args[0] if args else "datasets/dataset_train.csv"
        if not os.path.isfile(train_path):
            train_path = resolve_dataset_path(sys.argv, prefer="datasets/dataset_train.csv")
        weights_path = args[1] if len(args) >= 2 else "weights.json"
        cross_validate(train_path, weights_path)


if __name__ == "__main__":
    main()
