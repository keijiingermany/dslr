#!/usr/bin/env python3
import sys
import json
from lib.utils import resolve_dataset_path, read_csv_dicts, numeric_feature_names, safe_float, DROP_COLUMNS, HOUSE_COL, die
from lib.preprocess import preprocess_fit_transform
from lib.logreg import train_ovr


def build_X_y(path: str):
    fieldnames, rows = read_csv_dicts(path)
    if HOUSE_COL not in fieldnames:
        die(f"'{HOUSE_COL}' column not found in train dataset: {path}")

    features = numeric_feature_names(fieldnames, DROP_COLUMNS)

    X = []
    y = []
    for r in rows:
        y.append(r.get(HOUSE_COL, "").strip())
        row = [safe_float(r.get(f, "")) for f in features]
        X.append(row)

    # basic sanity: remove empty labels (shouldn't happen in train)
    filtered_X, filtered_y = [], []
    for xi, yi in zip(X, y):
        if yi != "" and yi.lower() != "nan":
            filtered_X.append(xi)
            filtered_y.append(yi)
    return filtered_X, filtered_y, features


def main():
    train_path = resolve_dataset_path(sys.argv, prefer="datasets/dataset_train.csv")
    X, y, features = build_X_y(train_path)

    Xb, means, stds = preprocess_fit_transform(X)

    classes = sorted(list(set(y)))

    # start values (good default)
    lr = 0.05
    iters = 3000

    thetas = train_ovr(Xb, y, classes, lr=lr, iters=iters)

    model = {
        "features": features,
        "means": means,
        "stds": stds,
        "classes": classes,
        "thetas": thetas,
        "hyperparams": {"lr": lr, "iters": iters},
    }

    with open("weights.json", "w") as f:
        json.dump(model, f)

    print("OK: weights.json generated")


if __name__ == "__main__":
    main()