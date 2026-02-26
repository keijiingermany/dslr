#!/usr/bin/env python3
import sys
import math
import matplotlib.pyplot as plt
from lib.utils import resolve_dataset_path, read_csv_dicts, DROP_COLUMNS, HOUSE_COL, safe_float


def main():
    train_path = resolve_dataset_path(sys.argv, prefer="datasets/dataset_train.csv")
    fieldnames, rows = read_csv_dicts(train_path)

    features = [c for c in fieldnames if c not in DROP_COLUMNS and c != HOUSE_COL]

    # allow limiting number of features to avoid huge grid
    k = 6
    if len(sys.argv) >= 3 and sys.argv[2].strip() != "":
        k = int(sys.argv[2].strip())
    features = features[:max(2, min(k, len(features)))]

    # build data dict
    data = {f: [] for f in features}
    for r in rows:
        for f in features:
            v = safe_float(r.get(f, ""))
            data[f].append(v)

    n = len(features)
    fig = plt.figure(figsize=(3*n, 3*n))

    for i in range(n):
        for j in range(n):
            ax = fig.add_subplot(n, n, i*n + j + 1)
            fi, fj = features[i], features[j]

            if i == j:
                # histogram on diagonal (ignore NaN)
                vals = [v for v in data[fi] if not math.isnan(v)]
                ax.hist(vals, bins=20)
            else:
                xs, ys = [], []
                for a, b in zip(data[fj], data[fi]):
                    if (not math.isnan(a)) and (not math.isnan(b)):
                        xs.append(a)
                        ys.append(b)
                ax.scatter(xs, ys, s=5)

            if i == n - 1:
                ax.set_xlabel(fj, rotation=90)
            else:
                ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(fi)
            else:
                ax.set_yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()