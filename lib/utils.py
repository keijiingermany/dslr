import os
import sys
import csv
import math
from typing import Dict, List, Tuple, Optional


DEFAULT_CANDIDATES = [
    os.path.join("datasets", "dataset_train.csv"),
    os.path.join("datasets", "dataset_test.csv"),
    "dataset_train.csv",
    "dataset_test.csv",
]


DROP_COLUMNS = {
    "Index",
    "Hogwarts House",
    "First Name",
    "Last Name",
    "Birthday",
    "Best Hand",
}


HOUSE_COL = "Hogwarts House"


def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}")
    sys.exit(code)


def info(msg: str) -> None:
    print(msg)


def is_number(x) -> bool:
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


def resolve_dataset_path(argv, prefer: Optional[str] = None) -> str:
    """
    Priority:
      1) argv[1] if provided
      2) prefer if provided and exists
      3) common defaults in DEFAULT_CANDIDATES
    """
    if len(argv) >= 2 and str(argv[1]).strip() != "":
        return str(argv[1]).strip()

    if prefer is not None and os.path.isfile(prefer):
        return prefer

    for c in DEFAULT_CANDIDATES:
        if os.path.isfile(c):
            return c

    die(
        "dataset path not provided and default files were not found.\n"
        "Tried:\n  - " + "\n  - ".join(DEFAULT_CANDIDATES) + "\n\n"
        "Usage:\n  python3 <script>.py datasets/dataset_train.csv"
    )
    return ""  # unreachable


def read_csv_dicts(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    if not os.path.isfile(path):
        die(f"file not found: {path}")

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            die(f"CSV header not found (empty file?): {path}")
        rows = [row for row in reader]
        return list(reader.fieldnames), rows


def numeric_feature_names(fieldnames: List[str], drop: set) -> List[str]:
    return [c for c in fieldnames if c not in drop]


def safe_float(s: str) -> float:
    # caller should check is_number; if not, returns NaN
    if is_number(s):
        return float(str(s).strip())
    return float("nan")


def dot(a: List[float], b: List[float]) -> float:
    acc = 0.0
    for x, y in zip(a, b):
        acc += x * y
    return acc


def sigmoid(z: float) -> float:
    # numerically safer sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)
