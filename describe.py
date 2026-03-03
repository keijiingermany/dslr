import sys
from lib.utils import resolve_dataset_path, read_csv_dicts, is_number
from lib.stats import (
    mean, var_sample, std_sample,
    compute_min, compute_max, compute_range,
    percentile_linear, iqr,
    skewness, kurtosis_excess, count_unique,
)

MANDATORY = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
BONUS = ["Var", "Range", "IQR", "Skew", "Kurt", "Unique", "NaN"]
MAX_WIDTH = 120   # target terminal width


def compute(data, nans):
    mu = mean(data)
    ds = sorted(data)
    return {
        "Count":  f"{len(data):.6f}",
        "Mean":   f"{mu:.6f}",
        "Std":    f"{std_sample(data, mu):.6f}",
        "Min":    f"{compute_min(data):.6f}",
        "25%":    f"{percentile_linear(ds, 0.25):.6f}",
        "50%":    f"{percentile_linear(ds, 0.50):.6f}",
        "75%":    f"{percentile_linear(ds, 0.75):.6f}",
        "Max":    f"{compute_max(data):.6f}",
        "Var":    f"{var_sample(data, mu):.6f}",
        "Range":  f"{compute_range(data):.6f}",
        "IQR":    f"{iqr(ds):.6f}",
        "Skew":   f"{skewness(data, mu):.6f}",
        "Kurt":   f"{kurtosis_excess(data, mu):.6f}",
        "Unique": f"{count_unique(data)}",
        "NaN":    f"{nans}",
    }


def print_block(title, stat_labels, all_cols, stats):
    print(title)
    lw = max(len(s) for s in stat_labels)
    val_w = 13  # "1234567.000000" → width for numeric values

    # group columns so the total line width stays within MAX_WIDTH
    # line = label(lw) + "  " + N*(val_w + 2) + name_overhead
    # simplest: each col takes max(len(col_name), val_w) + 2 separator
    chunks = []
    chunk = []
    used = lw + 2
    for c in all_cols:
        cw = max(len(c), val_w)
        if chunk and used + cw + 2 > MAX_WIDTH:
            chunks.append(chunk)
            chunk = [c]
            used = lw + 2 + cw + 2
        else:
            chunk.append(c)
            used += cw + 2
    if chunk:
        chunks.append(chunk)

    for cols in chunks:
        cws = [max(len(c), max(len(stats[c][s]) for s in stat_labels))
               for c in cols]
        print(f"{'':>{lw}}  " + "  ".join(
            f"{c:>{w}}" for c, w in zip(cols, cws)
        ))
        for s in stat_labels:
            print(f"{s:>{lw}}  " + "  ".join(
                f"{stats[c][s]:>{w}}" for c, w in zip(cols, cws)
            ))
        print()


def describe(path):
    fieldnames, rows = read_csv_dicts(path)

    cols_data = {k: [] for k in fieldnames}
    nan_count = {k: 0 for k in fieldnames}
    for r in rows:
        for k in fieldnames:
            v = r.get(k, "")
            if is_number(v):
                cols_data[k].append(float(v))
            else:
                nan_count[k] += 1

    num_cols = [k for k in fieldnames if cols_data[k]]
    if not num_cols:
        print(f"ERROR: no numeric columns in {path}")
        sys.exit(1)

    stats = {c: compute(cols_data[c], nan_count[c]) for c in num_cols}

    print(f"Dataset: {path}\n")
    print_block("[Mandatory]", MANDATORY, num_cols, stats)
    print_block("[Bonus]",     BONUS,     num_cols, stats)


def main():
    path = resolve_dataset_path(
        sys.argv, prefer="datasets/dataset_train.csv",
    )
    describe(path)


if __name__ == "__main__":
    main()
