"""
pair_plot.py
------------
Displays a scatter plot matrix (pair plot) with per-house colouring.
Usage:
  python3 pair_plot.py [dataset] [max_features=13]
"""

import sys
import math
import matplotlib.pyplot as plt
from lib.utils import resolve_dataset_path, read_csv_dicts, DROP_COLUMNS, HOUSE_COL, safe_float

HOUSE_COLORS = {
    "Gryffindor": "#e63946",
    "Hufflepuff": "#f4a261",
    "Ravenclaw": "#457b9d",
    "Slytherin": "#2a9d8f",
}
DEFAULT_COLOR = "#aaaaaa"
HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]


def main():
    train_path = resolve_dataset_path(sys.argv, prefer="datasets/dataset_train.csv")
    fieldnames, rows = read_csv_dicts(train_path)

    features = [c for c in fieldnames if c not in DROP_COLUMNS and c != HOUSE_COL]

    # allow limiting number of features to avoid huge grid (default: all 13)
    k = len(features)
    if len(sys.argv) >= 3 and sys.argv[2].strip() != "":
        k = int(sys.argv[2].strip())
    features = features[:max(2, min(k, len(features)))]

    # build data dict and house list (aligned per row)
    data = {f: [safe_float(r.get(f, "")) for r in rows] for f in features}
    houses_list = [r.get(HOUSE_COL, "").strip() for r in rows]
    nrows = len(rows)

    n = len(features)
    fig = plt.figure(figsize=(2.5 * n, 2.5 * n))

    for i in range(n):
        for j in range(n):
            ax = fig.add_subplot(n, n, i * n + j + 1)
            fi, fj = features[i], features[j]

            if i == j:
                # diagonal: per-house histogram
                for h in HOUSES:
                    vals = [
                        data[fi][k_]
                        for k_ in range(nrows)
                        if houses_list[k_] == h and not math.isnan(data[fi][k_])
                    ]
                    ax.hist(vals, bins=20, alpha=0.5,
                            color=HOUSE_COLORS.get(h, DEFAULT_COLOR), label=h)
            else:
                # off-diagonal: per-house scatter
                for h in HOUSES:
                    xs_, ys_ = [], []
                    for k_ in range(nrows):
                        if houses_list[k_] != h:
                            continue
                        a, b = data[fj][k_], data[fi][k_]
                        if not math.isnan(a) and not math.isnan(b):
                            xs_.append(a)
                            ys_.append(b)
                    ax.scatter(xs_, ys_, s=3, alpha=0.5,
                               color=HOUSE_COLORS.get(h, DEFAULT_COLOR), label=h)

            if i == n - 1:
                ax.set_xlabel(fj, fontsize=7, rotation=45, ha="right")
            else:
                ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(fi, fontsize=7)
            else:
                ax.set_yticks([])

    # single shared legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=HOUSE_COLORS[h], markersize=7, label=h)
        for h in HOUSES
    ]
    fig.legend(handles=handles, title="House", loc="upper right",
               bbox_to_anchor=(1.0, 1.0), fontsize=8)

    plt.suptitle("Pair Plot — Hogwarts features by House", y=1.01, fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()