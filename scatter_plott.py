#!/usr/bin/env python3
import sys
import math
import matplotlib.pyplot as plt
from lib.utils import resolve_dataset_path, read_csv_dicts, DROP_COLUMNS, HOUSE_COL, safe_float, die


def main():
    train_path = resolve_dataset_path(sys.argv, prefer="datasets/dataset_train.csv")
    fieldnames, rows = read_csv_dicts(train_path)

    features = [c for c in fieldnames if c not in DROP_COLUMNS and c != HOUSE_COL]
    if len(features) < 2:
        die("not enough numeric features")

    # choose pair
    x_name = features[0]
    y_name = features[1]
    if len(sys.argv) >= 4:
        x_name = sys.argv[2].strip()
        y_name = sys.argv[3].strip()
        if x_name not in features or y_name not in features:
            die("feature name not found")

    xs, ys = [], []
    for r in rows:
        x = safe_float(r.get(x_name, ""))
        y = safe_float(r.get(y_name, ""))
        if (not math.isnan(x)) and (not math.isnan(y)):
            xs.append(x)
            ys.append(y)

    plt.figure()
    plt.scatter(xs, ys, s=10)
    plt.title(f"Scatter: {x_name} vs {y_name}")
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


if __name__ == "__main__":
    main()