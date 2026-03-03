import sys
import json
from lib.utils import (
    resolve_dataset_path, read_csv_dicts,
    numeric_feature_names, safe_float,
    DROP_COLUMNS, HOUSE_COL, die,
)
from lib.preprocess import preprocess_fit_transform
from lib.logreg import train_ovr, predict_ovr_one, OPTIMIZERS

# Astronomy ≈ -DADA (r=-1.0); drop one to avoid redundancy
REDUNDANT_FEATURES = {"Defense Against the Dark Arts"}

DEFAULTS = {
    "batch":    {"lr": 0.1,  "iters": 5000},
    "sgd":      {"lr": 0.05, "iters": 500},
    "minibatch": {"lr": 0.1, "iters": 500, "bs": 32},
}


def parse_args():
    dataset = None
    optimizer = "batch"
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--optimizer":
            if i + 1 >= len(args):
                die("--optimizer requires: batch|sgd|minibatch")
            optimizer = args[i + 1].lower()
            if optimizer not in OPTIMIZERS:
                die(f"Unknown optimizer '{optimizer}'")
            i += 2
        else:
            dataset = dataset or args[i]
            i += 1
    if dataset is None:
        dataset = resolve_dataset_path(
            [sys.argv[0]],
            prefer="datasets/dataset_train.csv",
        )
    return dataset, optimizer


def main():
    path, opt = parse_args()
    fnames, rows = read_csv_dicts(path)
    if HOUSE_COL not in fnames:
        die(f"'{HOUSE_COL}' column not found")

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

    Xb, means, stds = preprocess_fit_transform(X)
    classes = sorted(set(y))

    d = DEFAULTS[opt]
    lr, iters = d["lr"], d["iters"]
    bs = d.get("bs", 32)
    print(f"Optimizer: {opt}  (lr={lr}, iters={iters}"
          + (f", bs={bs}" if opt == "minibatch" else "")
          + ")")

    thetas = train_ovr(
        Xb, y, classes,
        lr=lr, iters=iters, optimizer=opt, batch_size=bs,
    )

    # Training accuracy
    correct = sum(
        1 for xi, yi in zip(Xb, y)
        if predict_ovr_one(xi, thetas) == yi
    )
    print(f"Training accuracy: {correct}/{len(y)}"
          f" = {correct / len(y) * 100:.2f}%")

    json.dump({
        "features": feats, "means": means, "stds": stds,
        "classes": classes, "thetas": thetas,
    }, open("weights.json", "w"))
    print("OK: weights.json generated")


if __name__ == "__main__":
    main()
