import sys
import math
import matplotlib.pyplot as plt
from lib.utils import resolve_dataset_path, read_csv_dicts, DROP_COLUMNS, HOUSE_COL, safe_float, die


HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]


def main():
    train_path = resolve_dataset_path(sys.argv, prefer="datasets/dataset_train.csv")
    fieldnames, rows = read_csv_dicts(train_path)

    if HOUSE_COL not in fieldnames:
        die(f"'{HOUSE_COL}' not found in dataset: {train_path}")

    # numeric features (including Arithmancy etc), drop meta columns but keep HOUSE_COL for split
    features = [c for c in fieldnames if c not in DROP_COLUMNS and c != HOUSE_COL]

    # pick a feature from argv[2] optionally
    feature = features[0]
    if len(sys.argv) >= 3 and sys.argv[2].strip() != "":
        feature = sys.argv[2].strip()
        if feature not in features:
            die(f"feature not found: {feature}")

    per_house = {h: [] for h in HOUSES}
    for r in rows:
        h = r.get(HOUSE_COL, "").strip()
        if h in per_house:
            v = safe_float(r.get(feature, ""))
            if not math.isnan(v):
                per_house[h].append(v)

    plt.figure()
    for h in HOUSES:
        plt.hist(per_house[h], bins=30, alpha=0.5, label=h)
    plt.title(f"Histogram: {feature}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()