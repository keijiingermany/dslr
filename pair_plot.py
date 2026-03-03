"""
pair_plot.py  –  scatter plot matrix with per-house colouring.

From this visualization, which features will you use for logistic regression?
Answer: features that show clear house separation.

Usage:
  python3 pair_plot.py [dataset]
"""

import sys
import math
import matplotlib.pyplot as plt

from lib.utils import (
    resolve_dataset_path, read_csv_dicts,
    DROP_COLUMNS, HOUSE_COL, safe_float,
)

HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
COLORS = ["#e63946", "#f4a261", "#457b9d", "#2a9d8f"]

# Drop "Defense Against the Dark Arts" (r≈-1.0 with Astronomy → redundant)
SKIP = {"Defense Against the Dark Arts"}


def main():
    path = resolve_dataset_path(sys.argv, prefer="datasets/dataset_train.csv")
    _, rows = read_csv_dicts(path)

    features = [
        c for c in rows[0].keys()
        if c not in DROP_COLUMNS and c != HOUSE_COL and c not in SKIP
    ]
    n = len(features)

    per_house = {h: {f: [] for f in features} for h in HOUSES}
    for r in rows:
        h = r.get(HOUSE_COL, "").strip()
        if h not in per_house:
            continue
        for f in features:
            v = safe_float(r.get(f, ""))
            if not math.isnan(v):
                per_house[h][f].append(v)

    fig, axes = plt.subplots(n, n, figsize=(2.5 * n, 2.5 * n))
    fig.suptitle("Pair Plot — which features separate the houses?",
                 fontsize=12, y=1.002)

    for i, fi in enumerate(features):
        for j, fj in enumerate(features):
            ax = axes[i][j]
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            if i == j:
                for h, c in zip(HOUSES, COLORS):
                    ax.hist(per_house[h][fi], bins=20, alpha=0.5, color=c)
            else:
                for h, c in zip(HOUSES, COLORS):
                    xs, ys = per_house[h][fj], per_house[h][fi]
                    mn = min(len(xs), len(ys))
                    ax.scatter(xs[:mn], ys[:mn], s=2, alpha=0.3, color=c)

            if i == n - 1:
                ax.set_xlabel(fj, fontsize=7, rotation=30, ha="right")
            if j == 0:
                ax.set_ylabel(fi, fontsize=7)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=c, markersize=8, label=h)
        for h, c in zip(HOUSES, COLORS)
    ]
    fig.legend(handles=handles, title="House",
               loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
