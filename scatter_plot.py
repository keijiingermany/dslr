"""
scatter_plot.py
---------------
Answers: "What are the two features that are similar?"

By default, automatically finds the pair of features with the highest
absolute Pearson correlation and plots them with per-house colouring.
You can also specify the pair manually:
  python3 scatter_plot.py [dataset] [feature_x] [feature_y]
"""

import sys
import math
import matplotlib.pyplot as plt
from lib.utils import resolve_dataset_path, read_csv_dicts, DROP_COLUMNS, HOUSE_COL, safe_float, die

HOUSE_COLORS = {
    "Gryffindor": "#e63946",
    "Hufflepuff": "#f4a261",
    "Ravenclaw": "#457b9d",
    "Slytherin": "#2a9d8f",
}
DEFAULT_COLOR = "#888888"


def pearson_correlation(xs: list, ys: list) -> float:
    """Pearson r between two lists of equal length (no NaN)."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    return num / (sx * sy)


def find_most_similar_pair(features: list, data: dict) -> tuple:
    """Return (feat_a, feat_b, r) with highest |Pearson r| (excluding perfect 1.0)."""
    best = (features[0], features[1], 0.0)
    n = len(features)
    for i in range(n):
        for j in range(i + 1, n):
            fi, fj = features[i], features[j]
            pairs = [
                (data[fi][k], data[fj][k])
                for k in range(len(data[fi]))
                if not math.isnan(data[fi][k]) and not math.isnan(data[fj][k])
            ]
            if len(pairs) < 2:
                continue
            xs_, ys_ = zip(*pairs)
            r = pearson_correlation(list(xs_), list(ys_))
            # exclude r == ±1.0 (e.g. linear duplicates) to find genuinely "similar"
            if abs(r) > abs(best[2]) and abs(r) < 0.9999:
                best = (fi, fj, r)
    # if everything is < threshold, just take the best we found
    return best


def main():
    train_path = resolve_dataset_path(sys.argv, prefer="datasets/dataset_train.csv")
    fieldnames, rows = read_csv_dicts(train_path)

    features = [c for c in fieldnames if c not in DROP_COLUMNS and c != HOUSE_COL]
    if len(features) < 2:
        die("not enough numeric features")

    # Build flat data dict (all rows, NaN included for alignment)
    data = {f: [safe_float(r.get(f, "")) for r in rows] for f in features}
    houses = [r.get(HOUSE_COL, "").strip() for r in rows]

    # Choose pair
    if len(sys.argv) >= 4:
        x_name = sys.argv[2].strip()
        y_name = sys.argv[3].strip()
        if x_name not in features or y_name not in features:
            die(f"feature not found. Available: {features}")
        pairs = [(x, y) for x, y in zip(data[x_name], data[y_name])]
        xs_all = [p[0] for p in pairs]
        ys_all = [p[1] for p in pairs]
        r = pearson_correlation(
            [v for v in xs_all if not math.isnan(v)],
            [v for v in ys_all if not math.isnan(v)],
        )
    else:
        x_name, y_name, r = find_most_similar_pair(features, data)
        print(f"Most similar pair: '{x_name}' vs '{y_name}'  (r = {r:.4f})")

    # Plot with house colouring
    fig, ax = plt.subplots(figsize=(8, 6))

    # collect per-house
    plotted_houses = set()
    for xi, yi, h in zip(data[x_name], data[y_name], houses):
        if math.isnan(xi) or math.isnan(yi):
            continue
        color = HOUSE_COLORS.get(h, DEFAULT_COLOR)
        label = h if h not in plotted_houses else "_nolegend_"
        ax.scatter(xi, yi, s=20, c=color, alpha=0.7, label=label)
        plotted_houses.add(h)

    ax.set_title(
        f"Scatter: {x_name}  vs  {y_name}\n(Pearson r = {r:.4f})", fontsize=11
    )
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.legend(title="House", loc="best")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()