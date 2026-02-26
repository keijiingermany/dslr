import sys
import json
from lib.utils import resolve_dataset_path, read_csv_dicts, numeric_feature_names, safe_float, DROP_COLUMNS, HOUSE_COL, die
from lib.preprocess import preprocess_transform
from lib.logreg import predict_ovr_one


def build_X(path: str, features):
    fieldnames, rows = read_csv_dicts(path)
    X = []
    for r in rows:
        row = [safe_float(r.get(f, "")) for f in features]
        X.append(row)
    return X


def main():
    # usage:
    # python3 logreg_predict.py datasets/dataset_test.csv weights.json
    test_path = resolve_dataset_path(sys.argv, prefer="datasets/dataset_test.csv")

    if len(sys.argv) >= 3 and str(sys.argv[2]).strip() != "":
        weights_path = str(sys.argv[2]).strip()
    else:
        weights_path = "weights.json"

    with open(weights_path) as f:
        model = json.load(f)

    features = model["features"]
    means = model["means"]
    stds = model["stds"]
    thetas = model["thetas"]

    X = build_X(test_path, features)
    Xb = preprocess_transform(X, means, stds)

    with open("houses.csv", "w") as f:
        f.write("Index,Hogwarts House\n")
        for i, xb in enumerate(Xb):
            pred = predict_ovr_one(xb, thetas)
            f.write(f"{i},{pred}\n")

    print("OK: houses.csv generated")


if __name__ == "__main__":
    main()