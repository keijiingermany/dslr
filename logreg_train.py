import sys
import json
from lib.utils import resolve_dataset_path, read_csv_dicts, numeric_feature_names, safe_float, DROP_COLUMNS, HOUSE_COL, die
from lib.preprocess import preprocess_fit_transform
from lib.logreg import train_ovr

# Features that are near-perfect linear combinations of another feature.
# Keeping one from each duplicate pair avoids multicollinearity and
# improves numerical stability without losing information.
# (Astronomy vs Defense Against the Dark Arts → r = -1.0000)
REDUNDANT_FEATURES = {"Defense Against the Dark Arts"}


def build_X_y(path: str):
    fieldnames, rows = read_csv_dicts(path)
    if HOUSE_COL not in fieldnames:
        die(f"'{HOUSE_COL}' column not found in train dataset: {path}")

    features = [
        f for f in numeric_feature_names(fieldnames, DROP_COLUMNS)
        if f not in REDUNDANT_FEATURES
    ]

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

    lr = 0.1
    iters = 5000

    thetas = train_ovr(Xb, y, classes, lr=lr, iters=iters)

    # --- training accuracy ---
    from lib.logreg import predict_ovr_one
    correct = sum(1 for xi, yi in zip(Xb, y) if predict_ovr_one(xi, thetas) == yi)
    acc = correct / len(y) * 100
    print(f"Training accuracy: {correct}/{len(y)} = {acc:.2f}%")

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