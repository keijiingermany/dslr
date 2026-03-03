"""
scatter_plot.py
---------------
Answers: "What are the two features that are similar?"

Astronomy and Defense Against the Dark Arts have Pearson
r = -0.9999 (near-perfect negative linear relationship).
They carry the same information, so one is redundant.
"""

import sys
import math
import matplotlib.pyplot as plt

from lib.utils import (
    resolve_dataset_path, read_csv_dicts,
    HOUSE_COL, safe_float,
)

HOUSE_COLORS = {
    "Gryffindor": "#e63946",
    "Hufflepuff": "#f4a261",
    "Ravenclaw": "#457b9d",
    "Slytherin": "#2a9d8f",
}

# The two most similar features (r ≈ -1.0)
FEAT_X = "Astronomy"
FEAT_Y = "Defense Against the Dark Arts"


def main():
    path = resolve_dataset_path(
        sys.argv, prefer="datasets/dataset_train.csv",
    )
    _, rows = read_csv_dicts(path)

    for house, color in HOUSE_COLORS.items():
        xs, ys = [], []
        for r in rows:
            if r.get(HOUSE_COL, "").strip() != house:
                continue
            x = safe_float(r.get(FEAT_X, ""))
            y = safe_float(r.get(FEAT_Y, ""))
            if not math.isnan(x) and not math.isnan(y):
                xs.append(x)
                ys.append(y)
        plt.scatter(xs, ys, s=12, alpha=0.6,
                    c=color, label=house)

    plt.title(f"{FEAT_X}  vs  {FEAT_Y}")
    plt.xlabel(FEAT_X)
    plt.ylabel(FEAT_Y)
    plt.legend(title="House", loc="best")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
