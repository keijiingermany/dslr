"""
evaluate.py — Self-evaluation for the DSLR classifier.

  python3 evaluate.py [dataset] [--optimizer batch|sgd|minibatch]
  python3 evaluate.py --all     [dataset]   # compare all 3 optimizers
  python3 evaluate.py --compare <ground_truth.csv> <houses.csv>
"""
import sys
import csv
import os
import json
import random
from lib.utils import (
    resolve_dataset_path, read_csv_dicts,
    numeric_feature_names, safe_float,
    DROP_COLUMNS, HOUSE_COL, die,
)
from lib.preprocess import (
    preprocess_fit_transform, preprocess_transform,
)
from lib.logreg import (
    train_ovr, predict_ovr_one, OPTIMIZERS,
)

REDUNDANT_FEATURES = {"Defense Against the Dark Arts"}
DEFAULTS = {
    "batch":    {"lr": 0.1,  "iters": 5000},
    "sgd":      {"lr": 0.05, "iters": 500},
    "minibatch": {"lr": 0.1, "iters": 500, "bs": 32},
}


def load_train_data(path):
    fnames, rows = read_csv_dicts(path)
    if HOUSE_COL not in fnames:
        die(f"'{HOUSE_COL}' not found in {path}")
    feats = [
        f for f in numeric_feature_names(fnames, DROP_COLUMNS)
        if f not in REDUNDANT_FEATURES
    ]
    X, y = [], []
    for r in rows:
        h = r.get(HOUSE_COL, "").strip()
        if h and h.lower() != "nan":
            X.append([safe_float(r.get(f, "")) for f in feats])
            y.append(h)
    return X, y, feats


def cross_validate(train_path, weights_path, optimizer="batch"):
    X, y, feats = load_train_data(train_path)
    m = len(X)

    # Shuffled 80/20 split (seed for reproducibility)
    random.seed(42)
    idx = list(range(m))
    random.shuffle(idx)
    split = int(0.8 * m)
    ti, vi = idx[:split], idx[split:]

    Xt = [X[i][:] for i in ti]
    yt = [y[i] for i in ti]
    Xv = [X[i][:] for i in vi]
    yv = [y[i] for i in vi]

    Xb, means, stds = preprocess_fit_transform(Xt)
    classes = sorted(set(yt))

    d = DEFAULTS[optimizer]
    lr, iters = d["lr"], d["iters"]
    bs = d.get("bs", 32)

    thetas = train_ovr(
        Xb, yt, classes,
        lr=lr, iters=iters,
        optimizer=optimizer, batch_size=bs,
    )

    Xvb = preprocess_transform(Xv, means[:], stds)
    yp = [predict_ovr_one(xb, thetas) for xb in Xvb]
    correct = sum(1 for t, p in zip(yv, yp) if t == p)
    acc = correct / len(yv) * 100

    print("=" * 55)
    print("  DSLR — Cross-Validation (80/20 split)")
    print("=" * 55)
    print(f"  Optimizer : {optimizer}")
    print(f"  Train     : {len(Xt)}  Val: {len(Xv)}")
    print(f"  Features  : {len(feats)}")
    print(f"  Accuracy  : {correct}/{len(yv)} = {acc:.2f}%")
    ok = "PASS" if acc >= 98.0 else "FAIL"
    print(f"  Status    : {ok} (>= 98%)")

    # Full-training accuracy from saved weights
    if weights_path and os.path.isfile(weights_path):
        model = json.load(open(weights_path))
        Xf = [row[:] for row in X]
        Xfb = preprocess_transform(
            Xf, model["means"][:], model["stds"],
        )
        cf = sum(
            1 for xi, yi in zip(Xfb, y)
            if predict_ovr_one(xi, model["thetas"]) == yi
        )
        print(f"  Full-train: {cf}/{len(y)}"
              f" = {cf / len(y) * 100:.2f}%")
    print()
    return acc


def compare(gt_path, pred_path):
    if not os.path.isfile(gt_path):
        die(f"not found: {gt_path}")
    if not os.path.isfile(pred_path):
        die(f"not found: {pred_path}")

    def load_map(p):
        out = {}
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                idx = r.get("Index", "").strip()
                h = r.get(HOUSE_COL, "").strip()
                if idx and h:
                    out[idx] = h
        return out

    gt, pr = load_map(gt_path), load_map(pred_path)
    common = sorted(set(gt) & set(pr),
                    key=lambda x: int(x) if x.isdigit() else x)
    if not common:
        die("no matching Index rows")

    yt = [gt[i] for i in common]
    yp = [pr[i] for i in common]
    c = sum(1 for t, p in zip(yt, yp) if t == p)
    acc = c / len(common) * 100

    print("=" * 55)
    print("  DSLR — Prediction vs Ground Truth")
    print("=" * 55)
    print(f"  Samples  : {len(common)}")
    print(f"  Accuracy : {c}/{len(common)} = {acc:.2f}%")
    ok = "PASS" if acc >= 98.0 else "FAIL"
    print(f"  Status   : {ok} (>= 98%)")
    print()


def compare_all(train_path):
    results = []
    for opt in ["batch", "sgd", "minibatch"]:
        print(f"[{opt}] training...", flush=True)
        acc = cross_validate(train_path, None, opt)
        results.append((opt, acc))

    print("=" * 55)
    print("  Summary — All Optimizers")
    print("=" * 55)
    print(f"  {'Optimizer':<12} {'Val Acc':>8}  {'Status'}")
    print(f"  {'-'*12}  {'-'*7}  {'-'*6}")
    for opt, acc in results:
        ok = "PASS" if acc >= 98.0 else "FAIL"
        tag = "(mandatory)" if opt == "batch" else "(bonus)"
        print(f"  {opt:<12} {acc:>7.2f}%  {ok}  {tag}")
    print()


def main():
    raw = sys.argv[1:]
    if raw and raw[0] == "--compare":
        if len(raw) < 3:
            die("Usage: evaluate.py --compare <gt> <pred>")
        compare(raw[1], raw[2])
        return

    if raw and raw[0] == "--all":
        tp = raw[1] if len(raw) >= 2 else None
        if not tp or not os.path.isfile(tp):
            tp = resolve_dataset_path(
                [sys.argv[0]], prefer="datasets/dataset_train.csv",
            )
        compare_all(tp)
        return

    optimizer = "batch"
    positional = []
    i = 0
    while i < len(raw):
        if raw[i] == "--optimizer":
            if i + 1 >= len(raw):
                die("--optimizer requires a value")
            optimizer = raw[i + 1].lower()
            if optimizer not in OPTIMIZERS:
                die(f"Unknown optimizer '{optimizer}'")
            i += 2
        else:
            positional.append(raw[i])
            i += 1

    tp = positional[0] if positional else None
    if not tp or not os.path.isfile(tp):
        tp = resolve_dataset_path(
            [sys.argv[0]],
            prefer="datasets/dataset_train.csv",
        )
    wp = positional[1] if len(positional) >= 2 else "weights.json"
    cross_validate(tp, wp, optimizer)


if __name__ == "__main__":
    main()
