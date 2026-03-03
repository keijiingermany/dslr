import sys
import math
import matplotlib.pyplot as plt

from lib.utils import (
    resolve_dataset_path,
    read_csv_dicts,
    DROP_COLUMNS,
    HOUSE_COL,
    safe_float,
    die,
)

HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
COLORS = ["#e74c3c", "#f1c40f", "#3498db", "#2ecc71"]


def main():
    path = resolve_dataset_path(sys.argv, prefer="datasets/dataset_train.csv")
    fieldnames, rows = read_csv_dicts(path)

    if HOUSE_COL not in fieldnames:
        die(f"'{HOUSE_COL}' not found in dataset: {path}")

    features = [
        c for c in fieldnames
        if c not in DROP_COLUMNS and c != HOUSE_COL
    ]

    # collect data per house per feature
    data = {f: {h: [] for h in HOUSES} for f in features}
    for r in rows:
        h = r.get(HOUSE_COL, "").strip()
        if h not in HOUSES:
            continue
        for f in features:
            v = safe_float(r.get(f, ""))
            if not math.isnan(v):
                data[f][h].append(v)

    ncols = 3
    nrows = math.ceil(len(features) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2.5))
    fig.suptitle(
        "Score distributions by house\n"
        "→ Which course is homogeneous across all houses?",
        fontsize=13,
    )

    ANSWER = "Care of Magical Creatures"
    for idx, feat in enumerate(features):
        ax = axes[idx // ncols][idx % ncols]
        for h, color in zip(HOUSES, COLORS):
            ax.hist(data[feat][h], bins=20, alpha=0.5, label=h, color=color)
        if feat == ANSWER:
            ax.set_title(feat, fontsize=8, fontweight="bold", color="green")
            for spine in ax.spines.values():
                spine.set_edgecolor("green")
                spine.set_linewidth(2.5)
        else:
            ax.set_title(feat, fontsize=8)
        ax.tick_params(labelsize=6)

    # hide unused subplots
    for idx in range(len(features), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.5) for c in COLORS
    ]
    fig.legend(handles, HOUSES, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
