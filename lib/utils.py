import os
import sys
import csv
import math

DEFAULT_CANDIDATES = [
    os.path.join("datasets", "dataset_train.csv"),
    os.path.join("datasets", "dataset_test.csv"),
    "dataset_train.csv",
    "dataset_test.csv",
]

DROP_COLUMNS = {
    "Index", "Hogwarts House",
    "First Name", "Last Name", "Birthday", "Best Hand",
}

HOUSE_COL = "Hogwarts House"


def die(msg, code=1):
    print(f"ERROR: {msg}")
    sys.exit(code)


def is_number(x):
    if x is None:
        return False
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return False
    try:
        float(s)
        return True
    except Exception:
        return False


def resolve_dataset_path(argv, prefer=None):
    if len(argv) >= 2 and str(argv[1]).strip() != "":
        return str(argv[1]).strip()
    if prefer is not None and os.path.isfile(prefer):
        return prefer
    for c in DEFAULT_CANDIDATES:
        if os.path.isfile(c):
            return c
    die(
        "dataset not found. Usage:\n"
        "  python3 <script>.py datasets/dataset_train.csv"
    )
    return ""


def read_csv_dicts(path):
    if not os.path.isfile(path):
        die(f"file not found: {path}")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            die(f"empty CSV: {path}")
        rows = list(reader)
        return list(reader.fieldnames), rows


def numeric_feature_names(fieldnames, drop):
    return [c for c in fieldnames if c not in drop]


def safe_float(s):
    return float(str(s).strip()) if is_number(s) else float("nan")


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def sigmoid(z):
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)
